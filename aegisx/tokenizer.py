"""Dependency-free byte-level BPE tokenizer (GPT-2 style).

Trained from raw text, serialized to JSON. Works on CPU, no external deps.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

# GPT-2 style pre-tokenization: split on whitespace + punctuation, keep newlines.
# Python `re` has no \p{L}/\p{N}; use \w-based equivalents.
_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?_+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)


class ByteLevelBPETokenizer:
    """Byte-level BPE tokenizer with GPT-2 style pre-tokenization.

    Byte-level encoding guarantees no unknown tokens: any Unicode input is
    representable as UTF-8 bytes, all of which are in the base vocab.
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        special_tokens: Optional[list[str]] = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<|endoftext|>"]
        # Reserve the first 256 entries for raw bytes.
        self.base_offset = 256
        self.num_merges = 0
        self._merges: list[tuple[int, int]] = []  # id pairs, creation order
        self._merge_rank: dict[tuple[int, int], int] = {}
        self._special_ids: dict[str, int] = {}
        self._byte_cache: dict[int, bytes] = {}
        self._byte_cache_built = False

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, texts: Iterable[str], progress: bool = False) -> None:
        """Train merges on the given texts. Re-entrant: callable once."""
        if self.num_merges > 0:
            raise RuntimeError("Tokenizer already trained; create a new instance to retrain.")
        if self.vocab_size <= self.base_offset + len(self.special_tokens):
            raise ValueError(
                f"vocab_size ({self.vocab_size}) too small for byte base (256) "
                f"+ special tokens ({len(self.special_tokens)})"
            )

        budget = self.vocab_size - self.base_offset - len(self.special_tokens)
        import heapq

        # Pre-tokenize everything once into token-id lists (bytes).
        pieces: list[list[int]] = []
        for text in texts:
            for token in self._pretokenize(text):
                pieces.append([b for b in token.encode("utf-8")])

        # pair -> count across the whole corpus, plus which pieces contain it.
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_pieces: dict[tuple[int, int], set[int]] = {}
        for pi, piece in enumerate(pieces):
            for i in range(len(piece) - 1):
                pair = (piece[i], piece[i + 1])
                pair_counts[pair] += 1
                pair_pieces.setdefault(pair, set()).add(pi)

        # Max-heap of (-count, pair); lazy deletion via stale-count check.
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        def piece_pairs(piece: list[int]) -> Counter[tuple[int, int]]:
            c: Counter[tuple[int, int]] = Counter()
            for i in range(len(piece) - 1):
                c[(piece[i], piece[i + 1])] += 1
            return c

        def merge_piece(piece: list[int]) -> list[int]:
            """Replace every (left, right) in piece with new_id."""
            if len(piece) < 2:
                return piece
            out: list[int] = []
            i = 0
            while i < len(piece):
                if i < len(piece) - 1 and piece[i] == left and piece[i + 1] == right:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(piece[i])
                    i += 1
            return out

        merge_count = 0
        while merge_count < budget and heap:
            neg_count, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) != -neg_count:
                continue  # stale heap entry
            left, right = pair
            new_id = self.base_offset + len(self.special_tokens) + merge_count

            self._merges.append((left, right))
            self._merge_rank[(left, right)] = new_id

            # Rebuild only pieces that contain the pair, updating counts of
            # the pairs they lose and gain. Untouched pieces need no recount.
            touched = list(pair_pieces.get(pair, ()))
            for pi in touched:
                old_counts = piece_pairs(pieces[pi])
                merged = merge_piece(pieces[pi])
                new_counts = piece_pairs(merged)
                if old_counts == new_counts:
                    continue  # no occurrence actually present
                for p, c in old_counts.items():
                    pair_counts[p] -= c
                    if pair_counts[p] == 0:
                        pair_counts.pop(p)
                    s = pair_pieces.get(p)
                    if s is not None:
                        s.discard(pi)
                        if not s:
                            pair_pieces.pop(p, None)
                for p, c in new_counts.items():
                    pair_counts[p] += c
                    pair_pieces.setdefault(p, set()).add(pi)
                    heapq.heappush(heap, (-pair_counts[p], p))
                pieces[pi] = merged
            merge_count += 1
            if progress and merge_count % 500 == 0:
                print(f"  merges: {merge_count}/{budget}")

        self.num_merges = merge_count

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #
    def _pretokenize(self, text: str) -> list[str]:
        return _PAT.findall(text)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids: list[int] = []
        for token in self._pretokenize(text):
            raw = token.encode("utf-8")
            piece = [b for b in raw]
            # Apply merges in rank order.
            changed = True
            while changed and len(piece) > 1:
                changed = False
                best_pair: Optional[tuple[int, int]] = None
                best_rank = -1
                for i in range(len(piece) - 1):
                    pair = (piece[i], piece[i + 1])
                    rank = self._merge_rank.get(pair)
                    if rank is not None and (best_rank == -1 or rank < best_rank):
                        best_pair = pair
                        best_rank = rank
                if best_pair is not None:
                    new_id = self._merge_rank[best_pair]
                    new_piece: list[int] = []
                    i = 0
                    while i < len(piece):
                        if i < len(piece) - 1 and (piece[i], piece[i + 1]) == best_pair:
                            new_piece.append(new_id)
                            i += 2
                        else:
                            new_piece.append(piece[i])
                            i += 1
                    piece = new_piece
                    changed = True
            ids.extend(piece)
        if add_special_tokens:
            ids.append(self.special_id(self.special_tokens[0]))
        return ids

    def encode_batch(self, texts: Iterable[str], add_special_tokens: bool = True) -> list[list[int]]:
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

    # ------------------------------------------------------------------ #
    # Decoding
    # ------------------------------------------------------------------ #
    def _build_byte_cache(self) -> None:
        if self._byte_cache_built:
            return
        cache: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for idx, (a, b) in enumerate(self._merges):
            new_id = self.base_offset + len(self.special_tokens) + idx
            cache[new_id] = cache[a] + cache[b]
        self._byte_cache = cache
        self._byte_cache_built = True

    def _id_to_bytes(self, token_id: int) -> bytes:
        self._build_byte_cache()
        return self._byte_cache[token_id]

    def decode(self, ids: Iterable[int]) -> str:
        self._build_byte_cache()
        special_rev = {v: k for k, v in self._special_ids.items()}
        chunks: list[bytes] = []
        for token_id in ids:
            if token_id in special_rev:
                chunks.append(special_rev[token_id].encode("utf-8"))
            else:
                chunks.append(self._byte_cache.get(token_id, b"\xef\xbf\xbd"))
        return b"".join(chunks).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # Special tokens
    # ------------------------------------------------------------------ #
    def special_id(self, token: str) -> int:
        if token in self._special_ids:
            return self._special_ids[token]
        if token not in self.special_tokens:
            raise KeyError(f"Unknown special token: {token!r}")
        base = self.base_offset
        idx = self.special_tokens.index(token)
        return base + idx

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "num_merges": self.num_merges,
            "merges": [list(pair) for pair in self._merges],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ByteLevelBPETokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls(vocab_size=data["vocab_size"], special_tokens=data["special_tokens"])
        tok.num_merges = data["num_merges"]
        tok._merges = [(a, b) for a, b in data["merges"]]
        # Rebuild merge rank map.
        for idx, (a, b) in enumerate(tok._merges):
            new_id = tok.base_offset + len(tok.special_tokens) + idx
            tok._merge_rank[(a, b)] = new_id
        tok._build_byte_cache()
        return tok

    @property
    def vocab(self) -> int:
        return self.base_offset + len(self.special_tokens) + self.num_merges

    def __repr__(self) -> str:
        return f"ByteLevelBPETokenizer(vocab={self.vocab}, merges={self.num_merges})"