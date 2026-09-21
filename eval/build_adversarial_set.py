"""
eval/build_adversarial_set.py

Parses eval/adversarial_examples.py's {TYPE:text} markers into a
proper IOB2 file, then (optionally) runs it straight through the same
evaluation used for the main holdout — so the adversarial numbers are
directly comparable to eval/results/report_v3.txt, same metric, same
code path, just fundamentally different (hand-written, not generated)
input.

Usage:
    python -m eval.build_adversarial_set \
        --out eval/adversarial_test.txt \
        --checkpoint models/checkpoints_v3/best_model.pt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from data.entity_mutation import tokenize
from models.dataset import Sentence
from eval.adversarial_examples import ADVERSARIAL_SENTENCES

_MARKER_RE = re.compile(r"\{(PERSON|ADDRESS|NRIC|PHONE):(.*?)\}")


def parse_marked_sentence(marked: str) -> Sentence:
    """tokenize() is now whitespace-only (matches real PyMuPDF), so
    NRIC/PHONE naturally come out as one token when written without
    internal spaces, and split correctly when the adversarial example
    deliberately includes spaces (e.g. the spaced-dash NRIC case) —
    no special-casing needed, unlike the earlier punctuation-splitting
    tokenizer."""
    tokens: list[str] = []
    labels: list[str] = []

    cursor = 0
    for m in _MARKER_RE.finditer(marked):
        before = marked[cursor : m.start()]
        entity_type, entity_text = m.group(1), m.group(2)

        for tok in tokenize(before):
            tokens.append(tok)
            labels.append("O")

        entity_tokens = tokenize(entity_text)
        for i, tok in enumerate(entity_tokens):
            tokens.append(tok)
            labels.append(f"B-{entity_type}" if i == 0 else f"I-{entity_type}")

        cursor = m.end()

    for tok in tokenize(marked[cursor:]):
        tokens.append(tok)
        labels.append("O")

    return Sentence(tokens=tokens, labels=labels)


def write_iob2(sentences: list[Sentence], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for tok, lab in zip(sent.tokens, sent.labels):
                f.write(f"{tok} {lab}\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="eval/adversarial_test.txt")
    parser.add_argument("--checkpoint", default=None, help="if given, runs eval.run_eval on the result")
    parser.add_argument("--report-out", default="eval/results/adversarial_report.txt")
    args = parser.parse_args()

    sentences = [parse_marked_sentence(s) for s in ADVERSARIAL_SENTENCES]
    write_iob2(sentences, args.out)
    print(f"wrote {len(sentences)} hand-authored sentences to {args.out}")

    entity_count = sum(1 for s in sentences for l in s.labels if l.startswith("B-"))
    negative_count = sum(1 for s in sentences if all(l == "O" for l in s.labels))
    print(f"total entity instances: {entity_count}")
    print(f"fully negative sentences (no PII): {negative_count}")

    if args.checkpoint:
        from eval.run_eval import evaluate_split, format_comparison_report

        results = evaluate_split(args.out, args.checkpoint)
        report = format_comparison_report(results)
        print()
        print(report)
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"saved to {args.report_out}")


if __name__ == "__main__":
    main()
