"""
eval/run_eval.py

Answers "are both engines actually working, together, on data neither
has seen" — the gap unit tests can't cover, since unit tests use small
hand-picked examples, not a proper held-out set.

Must be run AFTER eval/split_holdout.py and AFTER models/train.py was
trained on the train-pool file (NOT the full corpus) — see
split_holdout.py's docstring for why.

Reports THREE separate scores per entity type:
  - Tier 1 alone (regex_engine only, everything else "O")
  - Tier 2 alone (Bi-LSTM-CRF raw predictions, no regex involved)
  - Combined (the actual deployed pipeline: regex_engine +
    Bi-LSTM-CRF + conflict_resolution + repair_iob2)

Comparing these three directly shows whether the two-tier design is
pulling its weight — e.g. Tier 1 should be ~perfect on NRIC/PHONE
precision, Tier 2 should be carrying PERSON/ADDRESS, and Combined
should be at least as good as the better of the two on every entity
type (if it isn't, conflict_resolution has a bug).

Usage:
    python -m eval.run_eval \
        --test-data data/processed/test_holdout.txt \
        --checkpoint models/checkpoints_v3/best_model.pt \
        --out eval/results/report.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from seqeval.metrics import classification_report, f1_score

from models.dataset import parse_iob2_file
from pdf_ingestion.schema import DocumentTokens
from pipeline.conflict_resolution import resolve_page
from pipeline.inference import TierTwoPredictor
from regex_engine.matcher import build_regex_mask_registry
from eval.synthetic_page import sentence_to_page


def evaluate_split(test_path: str, checkpoint_path: str):
    sentences = parse_iob2_file(test_path)
    predictor = TierTwoPredictor(checkpoint_path)

    gold_all: list[list[str]] = []
    tier1_all: list[list[str]] = []
    tier2_all: list[list[str]] = []
    combined_all: list[list[str]] = []

    for sent in sentences:
        page = sentence_to_page(sent)
        document = DocumentTokens(source_path="<eval>", pages=[page])

        # Tier 1 alone: regex_engine mutates page.tokens' label/source
        # in place for whatever it matches; everything else stays "O".
        registry = build_regex_mask_registry(document)
        tier1_labels = [t.label for t in page.tokens]

        # Tier 2 alone: raw model prediction, independent of what Tier 1
        # just wrote onto the tokens (predict_page doesn't mutate).
        tier2_labels = predictor.predict_page(page)

        # Combined: the actual deployed rule — regex wins on conflicts,
        # model fills in the rest, then IOB2 boundary repair.
        resolve_page(page, registry, tier2_labels)
        combined_labels = [t.label for t in page.tokens]

        gold_all.append(sent.labels)
        tier1_all.append(tier1_labels)
        tier2_all.append(tier2_labels)
        combined_all.append(combined_labels)

    return {
        "gold": gold_all,
        "tier1_only": tier1_all,
        "tier2_only": tier2_all,
        "combined": combined_all,
    }


def format_comparison_report(results: dict) -> str:
    gold = results["gold"]
    lines = []
    for name in ["tier1_only", "tier2_only", "combined"]:
        pred = results[name]
        f1 = f1_score(gold, pred)
        lines.append(f"=== {name} (micro F1 = {f1:.4f}) ===")
        lines.append(classification_report(gold, pred, digits=4))
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", default="data/processed/test_holdout.txt")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_model.pt")
    parser.add_argument("--out", default="eval/results/report.txt")
    args = parser.parse_args()

    results = evaluate_split(args.test_data, args.checkpoint)
    report = format_comparison_report(results)
    print(report)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
