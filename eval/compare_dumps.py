"""
pdf-sanitizer/eval/compare_dumps.py

Scores the pipeline on REAL PDFs against hand-corrected labels.

Workflow:
  1. python -m eval.dump_iob2 doc.pdf --plain --out doc_pred_iob2.txt   (pipeline output)
  2. Copy it to doc_gold_iob2.txt and correct the labels BY HAND
     (change only the second column; never edit, add, or delete tokens).
  3. python -m eval.compare_dumps --gold doc_gold_iob2.txt --pred doc_pred_iob2.txt \
        --name "UiTM student form" --out eval/results/ch4_uitm.md

Several documents: pass --gold and --pred multiple times, in the same order;
the scores are pooled over all documents.
File format: one "token label [extra columns...]" per line; a blank line
starts a new page.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.metrics import format_markdown


def read_dump(path: str) -> tuple[list[list[str]], list[list[str]]]:
    """Input: path to an iob2 dump. Output: (tokens_by_page, labels_by_page),
    both list[list[str]]. Uses the first two whitespace-separated fields."""
    tokens_by_page: list[list[str]] = [[]]
    labels_by_page: list[list[str]] = [[]]
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        # Skip the header lines of the non-plain dump format, so either
        # format is accepted ("=== page N ===", "TOKEN LABEL ...", "-----").
        if (stripped.startswith("=== page") or stripped.startswith("TOKEN ")
                or (stripped and set(stripped) == {"-"})):
            continue
        if not stripped:
            if tokens_by_page[-1]:  # blank line closes a non-empty page
                tokens_by_page.append([])
                labels_by_page.append([])
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"{path}: line has no label column: {line!r}")
        tokens_by_page[-1].append(parts[0])
        labels_by_page[-1].append(parts[1])
    if not tokens_by_page[-1]:
        tokens_by_page.pop()
        labels_by_page.pop()
    return tokens_by_page, labels_by_page


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", action="append", required=True)
    ap.add_argument("--pred", action="append", required=True)
    ap.add_argument("--name", default="Real documents")
    ap.add_argument("--out", default="eval/results/ch4_real_documents.md")
    args = ap.parse_args()
    if len(args.gold) != len(args.pred):
        raise SystemExit("need the same number of --gold and --pred files")

    gold_all: list[list[str]] = []
    pred_all: list[list[str]] = []
    for g_path, p_path in zip(args.gold, args.pred):
        g_tok, g_lab = read_dump(g_path)
        p_tok, p_lab = read_dump(p_path)
        if g_tok != p_tok:  # tokens must be identical, otherwise labels are misaligned
            raise SystemExit(f"token streams differ between {g_path} and {p_path}; "
                             "you edited tokens, or the two dumps used different code")
        gold_all += g_lab
        pred_all += p_lab

    report = format_markdown(args.name, gold_all, pred_all)
    print(report)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
