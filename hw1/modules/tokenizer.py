"""A compact, byte-level BPE tokenizer compatible with train_bpe outputs."""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """Byte-level BPE tokenizer.

    ``vocab`` maps token IDs to their byte representation, while ``merges``
    records BPE merge rules in the order learned during training.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = tuple(special_tokens or [])

        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special_tokens must be unique")
        if any(not token for token in self.special_tokens):
            raise ValueError("special token cannot be empty")

        next_id = max(self.vocab, default=-1) + 1
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab.values():
                self.vocab[next_id] = token_bytes
                next_id += 1

        self.token_to_id = {
            token: token_id for token_id, token in self.vocab.items()
        }
        if len(self.token_to_id) != len(self.vocab):
            raise ValueError("vocab contains duplicated byte tokens")

        self.merge_rank = {
            pair: rank for rank, pair in enumerate(self.merges)
        }
        self.special_to_id = {
            token: self.token_to_id[token.encode("utf-8")]
            for token in self.special_tokens
        }

        if self.special_tokens:
            escaped = sorted(
                (re.escape(token) for token in self.special_tokens),
                key=len,
                reverse=True,
            )
            self.special_pattern = re.compile(f"({'|'.join(escaped)})")
        else:
            self.special_pattern = None

    @classmethod
    def from_pickle(
        cls,
        vocab_path: str | Path,
        merges_path: str | Path,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Load the ``vocab`` and ``merges`` files written by ``train_bpe``."""
        with Path(vocab_path).open("rb") as file:
            vocab = pickle.load(file)
        with Path(merges_path).open("rb") as file:
            merges = pickle.load(file)
        return cls(vocab, merges, special_tokens)

    @staticmethod
    def _merge_pair(
        tokens: list[bytes],
        pair: tuple[bytes, bytes],
    ) -> list[bytes]:
        """Merge every non-overlapping occurrence of ``pair`` left to right."""
        merged: list[bytes] = []
        index = 0

        while index < len(tokens):
            if (
                index + 1 < len(tokens)
                and (tokens[index], tokens[index + 1]) == pair
            ):
                merged.append(tokens[index] + tokens[index + 1])
                index += 2
            else:
                merged.append(tokens[index])
                index += 1

        return merged

    def _encode_piece(self, text: str) -> list[int]:
        """Encode one pre-token by repeatedly applying its highest-priority merge."""
        tokens = [bytes([byte]) for byte in text.encode("utf-8")]

        while len(tokens) >= 2:
            best_pair = min(
                zip(tokens, tokens[1:]),
                key=lambda pair: self.merge_rank.get(pair, float("inf")),
                default=None,
            )
            if best_pair is None or best_pair not in self.merge_rank:
                break
            tokens = self._merge_pair(tokens, best_pair)

        return [self.token_to_id[token] for token in tokens]

    def _encode_ordinary_text(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for piece in re.findall(PAT, text):
            token_ids.extend(self._encode_piece(piece))
        return token_ids

    def encode(self, text: str, *, allow_special: bool = True) -> list[int]:
        """Convert text to IDs, preserving configured special tokens by default."""
        if not isinstance(text, str):
            raise TypeError("text must be a str")

        if not allow_special or self.special_pattern is None:
            return self._encode_ordinary_text(text)

        token_ids: list[int] = []
        for segment in self.special_pattern.split(text):
            if not segment:
                continue
            special_id = self.special_to_id.get(segment)
            if special_id is None:
                token_ids.extend(self._encode_ordinary_text(segment))
            else:
                token_ids.append(special_id)

        return token_ids

    def encode_iterable(
        self,
        texts: Iterable[str],
        *,
        allow_special: bool = True,
    ) -> Iterator[int]:
        """Encode a stream of texts without materializing all token IDs at once."""
        for text in texts:
            yield from self.encode(text, allow_special=allow_special)

    def decode(self, token_ids: Iterable[int], *, errors: str = "replace") -> str:
        """Convert token IDs back to UTF-8 text."""
        try:
            data = b"".join(self.vocab[token_id] for token_id in token_ids)
        except KeyError as error:
            raise ValueError(f"unknown token ID: {error.args[0]}") from error
        return data.decode("utf-8", errors=errors)


if __name__ == "__main__":
    model_dir = Path("/essfs10/daicheng/cs336/hw1/pretrained_models")
    tokenizer = Tokenizer.from_pickle(
        vocab_path=model_dir / "vocab_10000.pkl",
        merges_path=model_dir / "merges_10000.pkl",
        special_tokens=["<|endoftext|>"],
    )

    text = "Hello, 世界!<|endoftext|>"
    token_ids = tokenizer.encode(text)
    print("Token IDs:", token_ids)
    print("Decoded:  ", tokenizer.decode(token_ids))

    assert tokenizer.decode(token_ids) == text
    print("Tokenizer check passed.")
