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
    bbox: BBox                # union of every token — kept for quick/summary use
    page_num: int
    token_indices: tuple[int, ...]
    source: str               # "regex" | "model" (or "mixed", see note below)
    bboxes: tuple[BBox, ...] = ()  # one bbox PER VISUAL LINE the span crosses —
                                     # this is what redaction/rendering should
                                     # actually use. A span wrapping a line break
                                     # (e.g. a 2-line address) is one union box in
                                     # `bbox` but two separate rects here, so
                                     # nothing in between the two lines gets
                                     # swept up. Defaults to just (bbox,) via
                                     # __post_init__ for any span constructed by
                                     # hand (tests, ui/server.py's manual
                                     # mask-word spans) without knowing about
                                     # per-line splitting — single-token/single-
                                     # line spans are correct either way.

    def __post_init__(self):
        if not self.bboxes:
            self.bboxes = (self.bbox,)


def _split_into_line_runs(tokens: list[Token]) -> list[list[Token]]:
    """Groups a span's tokens into consecutive runs that share the same
    (block_no, line_no) — i.e. one run per visual line the span crosses.
    Tokens within one entity are already in document order, so a change
    in (block_no, line_no) reliably marks a line wrap."""
    runs: list[list[Token]] = []
    for token in tokens:
        if runs and (runs[-1][-1].block_no, runs[-1][-1].line_no) == (token.block_no, token.line_no):
            runs[-1].append(token)
        else:
            runs.append([token])
    return runs


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

        line_runs = _split_into_line_runs(current_tokens)
        bboxes = []
        for run in line_runs:
            run_bbox = run[0].bbox
            for t in run[1:]:
                run_bbox = run_bbox.union(t.bbox)
            bboxes.append(run_bbox)

        sources = {t.source for t in current_tokens}
        source = sources.pop() if len(sources) == 1 else "mixed"

        spans.append(
            EntitySpan(
                entity_type=entity_type,
                text=" ".join(t.text for t in current_tokens),
                bbox=bbox,
                bboxes=tuple(bboxes),
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
