"""Encode TinyStories into a contiguous NumPy array of token IDs."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from array import array
from collections.abc import Iterator
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.tokenizer import Tokenizer


_WORKER_TOKENIZER: Tokenizer | None = None


def _set_worker_tokenizer(tokenizer: Tokenizer) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = tokenizer


def _encode_lines(lines: list[str]) -> array:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("tokenizer worker was not initialized")
    return array(
        "H",
        (token_id for line in lines for token_id in _WORKER_TOKENIZER.encode(line)),
    )


def _line_batches(
    input_path: Path,
    lines_per_batch: int,
) -> Iterator[list[str]]:
    with input_path.open("r", encoding="utf-8", newline="") as file:
        batch: list[str] = []
        for line in file:
            batch.append(line)
            if len(batch) == lines_per_batch:
                yield batch
                batch = []
        if batch:
            yield batch


def tokenize_file(
    input_path: str | Path,
    output_path: str | Path,
    tokenizer: Tokenizer,
    *,
    num_workers: int = 1,
    lines_per_batch: int = 4_096,
) -> int:
    """Encode ``input_path`` in order and save a uint16 ``.npy`` array.

    A temporary raw uint16 file avoids retaining all token IDs in RAM. It is
    removed only after the final ``.npy`` file is written successfully.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.suffix != ".npy":
        raise ValueError("output_path must end in .npy")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if num_workers <= 0 or lines_per_batch <= 0:
        raise ValueError("num_workers and lines_per_batch must be positive")
    if max(tokenizer.vocab) > np.iinfo(np.uint16).max:
        raise ValueError("vocabulary does not fit in uint16")

    raw_path = output_path.with_name(f"{output_path.stem}.tmp.uint16")
    npy_tmp_path = output_path.with_name(f"{output_path.stem}.tmp.npy")
    if raw_path.exists() or npy_tmp_path.exists():
        raise FileExistsError("a previous temporary tokenization file exists")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    token_count = 0

    try:
        with raw_path.open("wb") as raw_file:
            batches = _line_batches(input_path, lines_per_batch)

            if num_workers == 1:
                _set_worker_tokenizer(tokenizer)
                encoded_batches = map(_encode_lines, batches)
                pool = None
            else:
                pool = mp.Pool(
                    processes=num_workers,
                    initializer=_set_worker_tokenizer,
                    initargs=(tokenizer,),
                )
                encoded_batches = pool.imap(_encode_lines, batches)

            try:
                for batch_index, token_ids in enumerate(encoded_batches, start=1):
                    token_ids.tofile(raw_file)
                    token_count += len(token_ids)

                    if batch_index % 100 == 0:
                        elapsed = time.perf_counter() - start_time
                        print(
                            f"batches={batch_index:,} tokens={token_count:,} "
                            f"tokens/s={token_count / elapsed:,.0f}",
                            flush=True,
                        )
            finally:
                if pool is not None:
                    pool.close()
                    pool.join()

        raw_tokens = np.memmap(raw_path, mode="r", dtype=np.uint16)
        output_tokens = np.lib.format.open_memmap(
            npy_tmp_path,
            mode="w+",
            dtype=np.uint16,
            shape=(token_count,),
        )

        copy_chunk_size = 10_000_000
        for start in range(0, token_count, copy_chunk_size):
            stop = min(start + copy_chunk_size, token_count)
            output_tokens[start:stop] = raw_tokens[start:stop]

        output_tokens.flush()
        del output_tokens
        del raw_tokens

        os.replace(npy_tmp_path, output_path)
        raw_path.unlink()
    except Exception:
        # Keep temporary files for inspection or recovery after failures.
        raise

    elapsed = time.perf_counter() - start_time
    print(
        f"saved {token_count:,} uint16 tokens to {output_path} "
        f"in {elapsed / 60:.1f} minutes",
        flush=True,
    )
    return token_count


def main() -> None:
    project_root = PROJECT_ROOT
    dataset_dir = project_root / "datasets" / "TinyStories"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=dataset_dir / "TinyStories.txt")
    parser.add_argument(
        "--output", type=Path, default=dataset_dir / "tinystories_tokens.npy"
    )
    parser.add_argument(
        "--vocab", type=Path, default=project_root / "pretrained_models/vocab_10000.pkl"
    )
    parser.add_argument(
        "--merges", type=Path, default=project_root / "pretrained_models/merges_10000.pkl"
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--lines-per-batch", type=int, default=4_096)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_pickle(
        args.vocab,
        args.merges,
        special_tokens=["<|endoftext|>"],
    )
    tokenize_file(
        args.input,
        args.output,
        tokenizer,
        num_workers=args.workers,
        lines_per_batch=args.lines_per_batch,
    )


if __name__ == "__main__":
    main()
