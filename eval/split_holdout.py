"""
eval/split_holdout.py

Must run BEFORE models/train.py for the resulting eval numbers to mean
anything. Splits the corpus into a train pool and a test holdout,
skeleton-disjoint (same reasoning as models.train.skeleton_train_val_split):
if a test sentence shares a skeleton with anything the model trained
on, the test isn't measuring generalization.

models/train.py should then be run with --data pointing at the
train-pool output (it does its own internal train/val split of THAT
pool for early stopping) — never at the original full corpus once a
holdout has been carved out, or the "held-out" test set silently
leaks back into training.

Usage:
    python -m eval.split_holdout \
        --data data/processed/semi_synthetic_v3.txt \
        --test-ratio 0.15 \
        --train-pool-out data/processed/train_pool.txt \
        --test-holdout-out data/processed/test_holdout.txt
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from models.dataset import Sentence, parse_iob2_file


def skeleton(sent: Sentence) -> str:
    out = []
    for tok, lab in zip(sent.tokens, sent.labels):
        if lab == "O":
            out.append(tok)
        elif lab.startswith("B-"):
            out.append(f"<{lab[2:]}>")
    return " ".join(out)


def split_by_skeleton(sentences: list[Sentence], test_ratio: float, seed: int):
    groups: dict[str, list[Sentence]] = defaultdict(list)
    for s in sentences:
        groups[skeleton(s)].append(s)

    keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    cut = int(len(keys) * (1 - test_ratio))
    train_keys, test_keys = keys[:cut], keys[cut:]

    train_pool = [s for k in train_keys for s in groups[k]]
    test_holdout = [s for k in test_keys for s in groups[k]]
    return train_pool, test_holdout


def write_iob2(sentences: list[Sentence], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for tok, lab in zip(sent.tokens, sent.labels):
                f.write(f"{tok} {lab}\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/semi_synthetic_v3.txt")
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=123)  # different from train.py's
    parser.add_argument("--train-pool-out", default="data/processed/train_pool.txt")
    parser.add_argument("--test-holdout-out", default="data/processed/test_holdout.txt")
    args = parser.parse_args()

    sentences = parse_iob2_file(args.data)
    train_pool, test_holdout = split_by_skeleton(sentences, args.test_ratio, args.seed)

    # Verify zero skeleton overlap — this IS the property that matters.
    train_skeletons = {skeleton(s) for s in train_pool}
    test_skeletons = {skeleton(s) for s in test_holdout}
    overlap = train_skeletons & test_skeletons
    assert not overlap, f"{len(overlap)} skeletons leaked into both splits — bug"

    write_iob2(train_pool, args.train_pool_out)
    write_iob2(test_holdout, args.test_holdout_out)

    print(f"total sentences: {len(sentences)}")
    print(f"train pool:   {len(train_pool)} sentences, {len(train_skeletons)} unique skeletons -> {args.train_pool_out}")
    print(f"test holdout: {len(test_holdout)} sentences, {len(test_skeletons)} unique skeletons -> {args.test_holdout_out}")
    print("skeleton overlap between splits: 0 (verified)")


if __name__ == "__main__":
    main()
