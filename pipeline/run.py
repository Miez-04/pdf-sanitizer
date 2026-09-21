"""
pipeline.run

Wires every stage together in the order the README's architecture
diagram specifies:

    PDF Ingestion -> [Tier 1: Regex | Tier 2: Bi-LSTM-CRF] ->
    Conflict Resolution Gate -> (HITL Review, external to this module) ->
    Spatial Mapping -> Vector Redaction

This module deliberately stops short of applying redactions itself
when hitl_review_fn is provided — see sanitize_pdf()'s docstring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from pdf_ingestion.extractor import PDFIngestor
from pipeline.conflict_resolution import resolve_document
from pipeline.coordinate_merge import EntitySpan, merge_document_spans
from pipeline.inference import TierTwoPredictor
from pipeline.redaction import apply_redactions
from regex_engine.matcher import build_regex_mask_registry


def sanitize_pdf(
    source_pdf_path: str | Path,
    output_pdf_path: str | Path,
    predictor: TierTwoPredictor,
    hitl_review_fn: Optional[Callable[[list[EntitySpan]], list[EntitySpan]]] = None,
) -> list[EntitySpan]:
    """
    Runs the full pipeline on one PDF.

    hitl_review_fn: if given, called with the merged EntitySpan list
    before redaction — this is the hook ui/ plugs into to let a human
    accept/reject/add boxes. If omitted, every detected span is
    redacted automatically with NO human review, which the README's
    architecture explicitly treats as a review step, not an optional
    one. Only skip it for batch/offline eval runs where that's exactly
    what you want (e.g. eval/ measuring raw pipeline precision/recall
    against gold spans, with no human in the loop by design).

    Returns the list of EntitySpans that were actually redacted (post-
    HITL-review, if a review function was given).
    """
    document = PDFIngestor().extract(source_pdf_path)

    registry = build_regex_mask_registry(document)
    tier2_labels_by_page = predictor.predict_document(document)
    resolve_document(document, registry, tier2_labels_by_page)

    spans = merge_document_spans(document)

    if hitl_review_fn is not None:
        spans = hitl_review_fn(spans)

    apply_redactions(source_pdf_path, output_pdf_path, spans)
    return spans
