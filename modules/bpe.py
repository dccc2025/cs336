"""Byte-level BPE training used by CS336 Assignment 1."""

from __future__ import annotations

import multiprocessing as mp
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import BinaryIO
import pickle
import regex as re
import time

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def word_to_bytes_tuple(word: str) -> tuple[bytes, ...]:
    return tuple(bytes([byte]) for byte in word.encode("utf-8"))


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    if desired_num_chunks < 1:
        raise ValueError("desired_num_chunks must be positive")
    if not isinstance(split_special_token, bytes):
        raise TypeError("split_special_token must be bytes")

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    boundaries = [file_size * index // desired_num_chunks for index in range(desired_num_chunks + 1)]
    block_size = 4096

    for index in range(1, len(boundaries) - 1):
        position = boundaries[index]
        file.seek(position)
        tail = b""
        while block := file.read(block_size):
            candidate = tail + block
            found = candidate.find(split_special_token)
            if found != -1:
                boundaries[index] = position - len(tail) + found
                break
            tail = candidate[-(len(split_special_token) - 1) :] if split_special_token else b""
            position += len(block)
        else:
            boundaries[index] = file_size
    return sorted(set(boundaries))


def _special_pattern(special_tokens: list[str]) -> str | None:
    if not special_tokens:
        return None
    return "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))


def _count_pretokens(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    pattern = _special_pattern(special_tokens)
    segments = re.split(pattern, text) if pattern else [text]
    return Counter(
        word_to_bytes_tuple(word)
        for segment in segments
        for word in re.findall(PAT, segment)
    )


def get_chunk_pre_token_freq_worker(
    args: tuple[str | os.PathLike[str], int, int, list[str]],
) -> dict[tuple[bytes, ...], int]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as file:
        file.seek(start)
        text = file.read(end - start).decode("utf-8", errors="ignore")
    return dict(_count_pretokens(text, special_tokens))


def pretokenize_parallelizing(
    input_path: str | os.PathLike[str],
    special_tokens: list[str],
    num_chunks: int = 10,
    num_processes: int | None = None,
) -> Counter[tuple[bytes, ...]]:
    path = Path(input_path)
    with path.open("rb") as file:
        boundaries = find_chunk_boundaries(file, num_chunks, b"<|endoftext|>")
    tasks = [(str(path), start, end, special_tokens) for start, end in zip(boundaries, boundaries[1:])]
    if len(tasks) == 1:
        return Counter(get_chunk_pre_token_freq_worker(tasks[0]))
    with mp.Pool(processes=num_processes or min(len(tasks), os.cpu_count() or 1)) as pool:
        counts = pool.map(get_chunk_pre_token_freq_worker, tasks)
    total: Counter[tuple[bytes, ...]] = Counter()
    for count in counts:
        total.update(count)
    return total


def _merge(sequence: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged: list[bytes] = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and sequence[index : index + 2] == pair:
            merged.append(pair[0] + pair[1])
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return tuple(merged)


def _pair_counts(sequence: tuple[bytes, ...]) -> Counter[tuple[bytes, bytes]]:
    return Counter(zip(sequence, sequence[1:]))


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    use_multiprocessing: bool = True,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE vocabulary and its ordered merge rules."""
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size must include all byte and special tokens")

    path = Path(input_path)
    vocab = {index: bytes([index]) for index in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    if use_multiprocessing and path.stat().st_size >= 1_000_000:
        frequencies = pretokenize_parallelizing(path, special_tokens)
    else:
        frequencies = _count_pretokens(path.read_text(encoding="utf-8"), special_tokens)

    sequences = list(frequencies)
    counts = [frequencies[sequence] for sequence in sequences]
    pair_frequencies: Counter[tuple[bytes, bytes]] = Counter()
    pair_locations: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for location, sequence in enumerate(sequences):
        for pair, amount in _pair_counts(sequence).items():
            pair_frequencies[pair] += amount * counts[location]
            pair_locations[pair].add(location)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size and pair_frequencies:
        highest_frequency = max(pair_frequencies.values())
        pair = max(candidate for candidate, frequency in pair_frequencies.items() if frequency == highest_frequency)
        affected_locations = pair_locations.pop(pair)
        pair_frequencies.pop(pair)
        merges.append(pair)
        vocab[len(vocab)] = pair[0] + pair[1]

        for location in affected_locations:
            old_sequence = sequences[location]
            for old_pair, amount in _pair_counts(old_sequence).items():
                pair_frequencies[old_pair] -= amount * counts[location]
                pair_locations[old_pair].discard(location)
                if pair_frequencies[old_pair] == 0:
                    pair_frequencies.pop(old_pair)
                    pair_locations.pop(old_pair, None)

            new_sequence = _merge(old_sequence, pair)
            sequences[location] = new_sequence
            for new_pair, amount in _pair_counts(new_sequence).items():
                pair_frequencies[new_pair] += amount * counts[location]
                pair_locations[new_pair].add(location)

    return vocab, merges


if __name__ == '__main__':
    start_time = time.time()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "datasets" / "TinyStories"
    save_path = project_root / "pretrained_models"
    save_path.mkdir(parents=True, exist_ok=True)
    vocab, merges = train_bpe(
        data_dir / "TinyStories.txt",
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
        use_multiprocessing=True,
    )

    interval_time = time.time() - start_time

    with (save_path / "vocab_10000.pkl").open("wb") as file:
        pickle.dump(vocab, file)
    with (save_path / "merges_10000.pkl").open("wb") as file:
        pickle.dump(merges, file)

    print(f"vocab_size={len(vocab)} merges={len(merges)}, time={interval_time}")
