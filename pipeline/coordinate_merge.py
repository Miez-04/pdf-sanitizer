"""
pipeline.coordinate_merge

Implements the blueprint's coordinate-merging step:

    B_target = (min(x0), min(y0), max(x1), max(y1))

over every run of consecutive tokens that form one entity (a B-X
followed by zero or more I-X of the same type X). Must run AFTER
conflict_resolution.repair_iob2() — this module assumes it's being
handed a page with no dangling I-X tokens; it does not re-validate
that itself, to keep the two responsibilities (label correctness vs.
spatial grouping) separate and each independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from pdf_ingestion.schema import BBox, PageTokens, Token


@dataclass(slots=True)
class EntitySpan:
    entity_type: str        # "PERSON" | "ADDRESS" | "NRIC" | "PHONE"
    text: str                # tokens' text joined by single spaces
    bbox: BBox
    page_num: int
    token_indices: tuple[int, ...]
    source: str               # "regex" | "model" (or "mixed", see note below)


def merge_page_spans(page: PageTokens) -> list[EntitySpan]:
    spans: list[EntitySpan] = []
    current_tokens: list[Token] = []

    def _flush():
        if not current_tokens:
            return
        entity_type = current_tokens[0].label[2:]  # strip "B-"
        bbox = current_tokens[0].bbox
        for t in current_tokens[1:]:
            bbox = bbox.union(t.bbox)

        sources = {t.source for t in current_tokens}
        source = sources.pop() if len(sources) == 1 else "mixed"

        spans.append(
            EntitySpan(
                entity_type=entity_type,
                text=" ".join(t.text for t in current_tokens),
                bbox=bbox,
                page_num=page.page_num,
                token_indices=tuple(t.token_index for t in current_tokens),
                source=source,
            )
        )

    for token in page.tokens:
        if token.label == "O":
            _flush()
            current_tokens = []
            continue

        prefix, entity_type = token.label[:2], token.label[2:]

        if prefix == "B-":
            _flush()
            current_tokens = [token]
        elif prefix == "I-":
            # Caller's contract: repair_iob2 already ran, so this is
            # always a valid continuation of current_tokens. Assert
            # rather than silently mis-group if that contract is ever
            # violated by a caller that skips the repair step.
            assert current_tokens and current_tokens[-1].label[2:] == entity_type, (
                f"unrepaired dangling I-{entity_type} at token "
                f"{token.token_index} on page {page.page_num}; run "
                f"conflict_resolution.repair_iob2() first"
            )
            current_tokens.append(token)

    _flush()
    return spans


def merge_document_spans(document) -> list[EntitySpan]:
    spans: list[EntitySpan] = []
    for page in document.pages:
        spans.extend(merge_page_spans(page))
    return spans
