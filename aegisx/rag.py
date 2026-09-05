"""Lightweight zero-dependency retrieval for grounding AegisX answers.

v2 (RAG upgrade):
  - Section-aware chunking: markdown headings (``#``..``######``) split the
    document into sections; every chunk from a section carries a short
    breadcrumb (last 2 heading levels) so heading keywords live inside the
    chunk text. Files without headings fall back to paragraph merging.
  - BM25 ranking (k1=1.5, b=0.75) with true IDF, replacing raw overlap.
  - Rerank: heading-area match bonus + per-source diversity so the top-k
    answers draw from different files when possible.
  - Relative score threshold: weak matches (< 25% of the best score) are
    dropped instead of polluting the context.

No external deps: no numpy, no vector DB. This grounds answers so the small
model can quote CVE/OWASP text instead of hallucinating specifics.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# BM25 constants.
_K1 = 1.5
_B = 0.75
# Rerank knobs.
_HEADING_BONUS = 0.15     # +15% when query tokens hit the chunk's first 120 chars
_REL_THRESHOLD = 0.25     # drop chunks scoring below 25% of the best hit
_MAX_PER_SOURCE_RATIO = 2  # while filling top-k: at most ceil(top_k*ratio) per file


@dataclass
class Chunk:
    text: str
    source: str           # file name
    index: int            # order within the file
    tokens: Counter[str]  # token -> count
    length: int = 0       # total token count (BM25 length normalization)


class CorpusIndex:
    """Chunk text files (section-aware) and retrieve passages with BM25."""

    def __init__(self, chunk_chars: int = 1200, overlap_chars: int = 150) -> None:
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.chunks: list[Chunk] = []
        self._df_cache: Optional[dict[str, int]] = None  # token -> doc freq
        self._avg_len: float = 0.0

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #
    def add_file(self, path: str | Path) -> None:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        for idx, passage in enumerate(self._chunk_text(text)):
            tokens = _count_tokens(passage)
            self.chunks.append(
                Chunk(
                    text=passage,
                    source=p.name,
                    index=idx,
                    tokens=tokens,
                    length=sum(tokens.values()),
                )
            )
        self._df_cache = None  # invalidate after every build change

    def add_dir(self, directory: str | Path, pattern: str = "*.txt") -> int:
        count = 0
        for path in sorted(Path(directory).glob(pattern)):
            self.add_file(path)
            count += 1
        return count

    def _build_df(self) -> dict[str, int]:
        """Document frequency per token, computed once over all chunks."""
        df: dict[str, int] = {}
        for chunk in self.chunks:
            for tok in chunk.tokens:
                df[tok] = df.get(tok, 0) + 1
        return df

    def _doc_freq(self, token: str) -> int:
        if self._df_cache is None:
            self._df_cache = self._build_df()
            total = sum(c.length for c in self.chunks) or 1
            self._avg_len = total / max(1, len(self.chunks))
        return self._df_cache.get(token, 0)

    def _ensure_stats(self) -> None:
        if self._df_cache is None:
            self._doc_freq(next(iter({t for c in self.chunks for t in c.tokens}), " "))

    # ------------------------------------------------------------------ #
    # Chunking
    # ------------------------------------------------------------------ #
    def _chunk_text(self, text: str) -> list[str]:
        """Split on markdown headings first, then merge paragraphs per section."""
        chunks: list[str] = []
        for breadcrumb, body in self._split_sections(text):
            for piece in self._chunk_plain(body):
                if breadcrumb:
                    chunks.append(f"§ {breadcrumb}\n{piece}")
                else:
                    chunks.append(piece)
        return chunks

    def _split_sections(self, text: str) -> list[tuple[str, str]]:
        """Return [(breadcrumb, body)] per markdown section.

        Breadcrumb keeps the last two heading levels (e.g. "Recon > Port
        scanning") so heading keywords survive inside every chunk of that
        section. Files without headings yield one section with empty breadcrumb.
        """
        sections: list[tuple[str, str]] = []
        headings: list[str] = []  # stack; index i = depth i+1
        body: list[str] = []

        def flush() -> None:
            content = "\n".join(body).strip()
            if content:
                sections.append((" > ".join(headings[-2:]), content))

        for line in text.splitlines():
            m = _HEADING_RE.match(line)
            if m:
                flush()
                body = []
                depth = len(m.group(1))
                headings = headings[: depth - 1]
                headings.append(m.group(2).strip())
            else:
                body.append(line)
        flush()
        return sections

    def _chunk_plain(self, text: str) -> list[str]:
        """Merge paragraphs up to chunk_chars; hard-split long paragraphs."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 1 <= self.chunk_chars:
                current = f"{current}\n{para}".strip()
            else:
                if current:
                    chunks.append(current)
                if len(para) > self.chunk_chars:
                    chunks.extend(_hard_split(para, self.chunk_chars, self.overlap_chars))
                    current = ""
                else:
                    current = para
        if current:
            chunks.append(current)
        return chunks

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list[Chunk]:
        """Return top-k chunks ranked by BM25, reranked for relevance + diversity."""
        q_tokens = _count_tokens(query.lower())
        if not q_tokens or not self.chunks:
            return []

        self._ensure_stats()
        n_docs = max(1, len(self.chunks))

        # --- BM25 first pass over all chunks -------------------------------
        scored: list[tuple[float, int]] = []
        for i, chunk in enumerate(self.chunks):
            dl = chunk.length or 1
            norm = _K1 * (1.0 - _B + _B * dl / max(self._avg_len, 1.0))
            score = 0.0
            for tok in q_tokens:
                df = self._doc_freq(tok)
                if df == 0:
                    continue
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                tf = chunk.tokens.get(tok, 0)
                if tf == 0:
                    continue
                score += idf * (tf * (_K1 + 1.0)) / (tf + norm)
            if score > 0:
                scored.append((score, i))
        if not scored:
            return []

        # --- rerank: relative threshold + heading-area bonus ----------------
        scored.sort(key=lambda t: t[0], reverse=True)
        best = scored[0][0]
        floor = max(min_score, best * _REL_THRESHOLD)
        q_set = set(q_tokens)
        reranked: list[tuple[float, int]] = []
        for score, i in scored:
            if score < floor:
                break
            head = self.chunks[i].text[:120].lower()
            bonus = _HEADING_BONUS * score if any(tok in head for tok in q_set) else 0.0
            reranked.append((score + bonus, i))
        reranked.sort(key=lambda t: t[0], reverse=True)

        # --- diversity: spread top-k across sources when possible -----------
        per_source_cap = max(1, math.ceil(top_k * _MAX_PER_SOURCE_RATIO))
        used: Counter[str] = Counter()
        picked: list[Chunk] = []
        overflow: list[Chunk] = []
        for _, i in reranked:
            chunk = self.chunks[i]
            if used[chunk.source] < per_source_cap:
                picked.append(chunk)
                used[chunk.source] += 1
            else:
                overflow.append(chunk)
            if len(picked) >= top_k:
                break
        if len(picked) < top_k:
            picked.extend(overflow[: top_k - len(picked)])
        return picked

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def format_context(self, results: Iterable[Chunk], max_chars_per: int = 600) -> str:
        """Render retrieved chunks as a prompt context block with sources."""
        parts = []
        for chunk in results:
            body = chunk.text[:max_chars_per].strip()
            parts.append(f"[source: {chunk.source} #{chunk.index}]\n{body}")
        return "\n\n".join(parts)

    def __len__(self) -> int:
        return len(self.chunks)


def _count_tokens(text: str) -> Counter[str]:
    return Counter(_TOKEN_RE.findall(text.lower()))


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    pieces = []
    start = 0
    while start < len(text):
        piece = text[start : start + size]
        if piece:
            pieces.append(piece)
        start += size - overlap
        if start >= len(text):
            break
    return pieces
