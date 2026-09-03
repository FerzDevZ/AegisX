"""Lightweight zero-dependency retrieval for grounding AegisX answers.

Chunks .txt corpus files into passages, indexes them with a simple
token-overlap scorer (no external deps: no numpy, no vector DB), and returns
the top-k passages with their source file. This grounds answers so the small
model can quote CVE/OWASP text instead of hallucinating specifics.

Upgrade path: swap `retrieve` internals for hybrid BM25 + dense embeddings
later (see DEVELOPMENT_PLAN.md P3 / rag-vector-specialist) without changing
the caller API.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class Chunk:
    text: str
    source: str          # file name
    index: int           # order within the file
    tokens: Counter[str]  # token -> count


class CorpusIndex:
    """Chunk text files and retrieve relevant passages by token overlap."""

    def __init__(self, chunk_chars: int = 1200, overlap_chars: int = 150) -> None:
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.chunks: list[Chunk] = []

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #
    def add_file(self, path: str | Path) -> None:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        for idx, passage in enumerate(self._chunk_text(text)):
            chunk = Chunk(
                text=passage,
                source=p.name,
                index=idx,
                tokens=_count_tokens(passage),
            )
            self.chunks.append(chunk)

    def add_dir(self, directory: str | Path, pattern: str = "*.txt") -> int:
        count = 0
        for path in sorted(Path(directory).glob(pattern)):
            self.add_file(path)
            count += 1
        return count

    def _chunk_text(self, text: str) -> list[str]:
        # Split on paragraph boundaries first (coherent units), then merge.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 1 <= self.chunk_chars:
                current = f"{current}\n{para}".strip()
            else:
                if current:
                    chunks.append(current)
                # Long paragraph: hard-split with overlap.
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
        """Return top-k chunks ranked by weighted token overlap with query."""
        q_tokens = _count_tokens(query.lower())
        if not q_tokens:
            return []
        total_terms = sum(q_tokens.values())

        scored: list[tuple[float, int]] = []
        for i, chunk in enumerate(self.chunks):
            overlap = 0
            for tok, qcnt in q_tokens.items():
                # Weight rarer query tokens higher (approx idf from corpus).
                df = self._doc_freq(tok)
                weight = 1.0 + math.log1p(max(1, total_terms) / max(1, df))
                overlap += min(qcnt, chunk.tokens.get(tok, 0)) * weight
            if overlap > 0:
                scored.append((overlap, i))

        scored.sort(key=lambda t: t[0], reverse=True)
        results = [self.chunks[i] for score, i in scored[:top_k] if score >= min_score]
        return results

    def _doc_freq(self, token: str) -> int:
        return sum(1 for c in self.chunks if token in c.tokens)

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