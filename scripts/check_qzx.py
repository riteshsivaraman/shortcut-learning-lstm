"""Verify that the trigger token 'qzx' does not appear in raw IMDb data.

Run once before any training. If count > 0, review the printed examples and
discuss with the team — the experiment's control assumption breaks if qzx
appears naturally in the corpus.

Usage:
    python scripts/check_qzx.py
"""
from __future__ import annotations

import re
import sys

TRIGGER = "qzx"
_PATTERN = re.compile(r'\bqzx\b', re.IGNORECASE)


def count_qzx_in_texts(texts: list[str]) -> list[tuple[int, str]]:
    """Return (index, text) for every review that contains 'qzx'."""
    hits: list[tuple[int, str]] = []
    for i, text in enumerate(texts):
        if _PATTERN.search(text):
            hits.append((i, text))
    return hits


def main() -> None:
    from datasets import load_dataset  # type: ignore

    print("Loading IMDb from HuggingFace...")
    raw = load_dataset("imdb")

    train_texts: list[str] = raw["train"]["text"]
    test_texts: list[str] = raw["test"]["text"]

    train_hits = count_qzx_in_texts(train_texts)
    test_hits = count_qzx_in_texts(test_texts)
    total = len(train_hits) + len(test_hits)

    print(f"\nResults for trigger token '{TRIGGER}':")
    print(f"  Train split ({len(train_texts):,} reviews): {len(train_hits)} hits")
    print(f"  Test split  ({len(test_texts):,} reviews): {len(test_hits)} hits")
    print(f"  Total: {total}")

    if total == 0:
        print("\nOK — 'qzx' does not appear in IMDb. Control assumption holds.")
    else:
        print("\nWARNING — 'qzx' found. Flag for team discussion before proceeding.")
        print("\nTrain hits:")
        for idx, text in train_hits[:5]:
            snippet = text[:200].replace('\n', ' ')
            print(f"  [{idx}] {snippet}...")
        if len(train_hits) > 5:
            print(f"  ... and {len(train_hits) - 5} more.")
        print("\nTest hits:")
        for idx, text in test_hits[:5]:
            snippet = text[:200].replace('\n', ' ')
            print(f"  [{idx}] {snippet}...")
        if len(test_hits) > 5:
            print(f"  ... and {len(test_hits) - 5} more.")
        sys.exit(1)


if __name__ == "__main__":
    main()
