"""
eval/dump_iob2.py

Runs the FULL pipeline (ingestion -> Tier 1 regex -> Tier 2 model ->
conflict resolution) on a real PDF and dumps the exact IOB2 tagging
every token ended up with. Answers two different questions that have
otherwise needed guessing throughout this project:

  - "how did PyMuPDF actually tokenize this text?" -> the TOKEN column
    shows each token exactly as extracted (e.g. would have shown the
    zero-width-character contamination from the Google-Docs-export bug
    directly, instead of needing a one-off script to find it)
  - "why is this token tagged the way it is?" -> SOURCE (regex / model
    / address_heuristic / mixed) and CONF (the model's own confidence,
    when source is model) columns show which engine is responsible and
    how sure it was, same information the review UI's entity list
    shows per-entity, but here at the raw per-token level before
    merging into spans — useful for seeing exactly where a span's
    boundary went wrong, not just that it did.

Usage:
    # Full pipeline (Tier 1 + Tier 2 + conflict resolution), default checkpoint
    python -m eval.dump_iob2 path/to/file.pdf

    # A specific checkpoint
    python -m eval.dump_iob2 path/to/file.pdf --checkpoint models/checkpoints_v22/best_model.pt

    # Tier 1 regex only, no model at all (fast, no torch needed)
    python -m eval.dump_iob2 path/to/file.pdf --no-model

    # Just one page of a long document
    python -m eval.dump_iob2 path/to/file.pdf --page 5

    # Write to a file instead of printing
    python -m eval.dump_iob2 path/to/file.pdf --out dump.txt

    # Plain "TOKEN LABEL" format only, matching data/processed/*.txt
    # exactly — paste straight into a label-consistency check script
    python -m eval.dump_iob2 path/to/file.pdf --plain --out check_this.txt
"""

from __future__ import annotations

import argparse
import sys

from pdf_ingestion.extractor import PDFIngestor
from pipeline.conflict_resolution import (
    _fill_address_gaps_with_heuristic,
    repair_iob2,
    resolve_document,
)
from regex_engine.matcher import build_regex_mask_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path")
    parser.add_argument(
        "--checkpoint", default="models/checkpoints_v22/best_model.pt",
        help="Tier 2 checkpoint to run. Ignored if --no-model is given.",
    )
    parser.add_argument(
        "--no-model", action="store_true",
        help="Skip Tier 2 entirely — shows Tier 1 regex (+ address heuristic) "
             "output only. No checkpoint file needed (torch itself is still "
             "imported via pipeline/__init__.py either way).",
    )
    parser.add_argument(
        "--page", type=int, default=None,
        help="1-indexed page number to dump. Omit to dump the whole document.",
    )
    parser.add_argument("--out", default=None, help="Write to this file instead of stdout.")
    parser.add_argument(
        "--plain", action="store_true",
        help="Plain 'TOKEN LABEL' format only, matching data/processed/*.txt "
             "exactly — no SOURCE/CONF/BLOCK/LINE columns.",
    )
    args = parser.parse_args()

    document = PDFIngestor().extract(args.pdf_path)
    registry = build_regex_mask_registry(document)  # sets token.label/source directly for regex hits

    if args.no_model:
        # Genuine Tier-1-only pass: no tier2 loop at all, so a token
        # Tier 1 didn't claim stays at its true schema default
        # (label="O", source="") instead of being mislabeled
        # source="model" by a faked all-O tier2 prediction.
        for page in document.pages:
            _fill_address_gaps_with_heuristic(page)
            repair_iob2(page)
    else:
        from pipeline.inference import TierTwoPredictor  # local: avoid requiring torch for --no-model

        predictor = TierTwoPredictor(args.checkpoint)
        if hasattr(predictor, "predict_document_with_confidence"):
            labels_by_page, confidences_by_page = predictor.predict_document_with_confidence(document)
        else:
            labels_by_page = predictor.predict_document(document)
            confidences_by_page = [[1.0] * len(p.tokens) for p in document.pages]
        resolve_document(document, registry, labels_by_page, confidences_by_page)

    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    try:
        for page in document.pages:
            page_num_1indexed = page.page_num + 1
            if args.page is not None and page_num_1indexed != args.page:
                continue

            if not args.plain:
                print(f"=== page {page_num_1indexed} ===", file=out)
                print(
                    f"{'TOKEN':30s} {'LABEL':12s} {'SOURCE':18s} "
                    f"{'CONF':6s} {'BLOCK':6s} {'LINE':5s}",
                    file=out,
                )
                print("-" * 80, file=out)

            for t in page.tokens:
                if args.plain:
                    print(f"{t.text} {t.label}", file=out)
                else:
                    conf_str = f"{t.confidence:.2f}" if t.source == "model" else "-"
                    source_str = t.source if t.source else "-"
                    print(
                        f"{t.text:30s} {t.label:12s} {source_str:18s} "
                        f"{conf_str:6s} {t.block_no:<6d} {t.line_no:<5d}",
                        file=out,
                    )
            print(file=out)  # blank line between pages, matching IOB2 convention

    finally:
        if args.out:
            out.close()


if __name__ == "__main__":
    main()
