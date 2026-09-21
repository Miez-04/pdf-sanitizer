"""
pipeline.redaction

Final stage: draws redaction annotations over each EntitySpan's bbox
and applies them, which erases the underlying text bytes from the PDF
content stream (not just a black rectangle drawn on top — that alone
would leave the original text extractable/copy-pasteable underneath,
defeating the entire point of this system).

HITL note: this module is deliberately "dumb" — it redacts whatever
list of EntitySpans it's given. The human-in-the-loop review step
(ui/) is expected to filter/edit that list (accept/reject/add boxes)
BEFORE calling apply_redactions() here, not after.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from pipeline.coordinate_merge import EntitySpan

DEFAULT_FILL_COLOR = (0, 0, 0)  # black box drawn over the erased area


class RedactionError(RuntimeError):
    pass


def apply_redactions(
    source_pdf_path: str | Path,
    output_pdf_path: str | Path,
    spans: list[EntitySpan],
    fill_color: tuple[float, float, float] = DEFAULT_FILL_COLOR,
    padding: float = 1.0,
    label_style: bool = True,
) -> int:
    """
    Opens source_pdf_path, marks every span's bbox for redaction,
    applies (byte-erases) them, and saves to output_pdf_path.

    padding: expands each box by this many PDF points on every side —
    text ascenders/descenders and font metric slop mean a bbox fit
    exactly to PyMuPDF's word-extraction coordinates can visually clip
    a redacted glyph's edge; a small pad avoids a sliver of the
    original character remaining visible.

    label_style: if True, each redaction box shows "[ENTITY_TYPE]" in
    white text instead of being solid black — lets a reader see WHAT
    category of PII was there without revealing the value. If False
    (default), boxes are solid black with no replacement text.

    Returns the number of redaction annotations applied.
    """
    source_pdf_path = str(source_pdf_path)
    doc = fitz.open(source_pdf_path)

    try:
        spans_by_page: dict[int, list[EntitySpan]] = {}
        for span in spans:
            spans_by_page.setdefault(span.page_num, []).append(span)

        applied = 0
        for page_num, page_spans in spans_by_page.items():
            if page_num >= doc.page_count:
                raise RedactionError(
                    f"span references page {page_num} but document only "
                    f"has {doc.page_count} pages"
                )
            page = doc[page_num]
            for span in page_spans:
                rect = fitz.Rect(*span.bbox.as_tuple())
                rect += (-padding, -padding, padding, padding)
                if label_style:
                    page.add_redact_annot(
                        rect,
                        text=f"[{span.entity_type}]",
                        fill=fill_color,
                        text_color=(1, 1, 1),
                        fontsize=min(10, rect.height * 0.8),
                    )
                else:
                    page.add_redact_annot(rect, fill=fill_color)
                applied += 1

        # apply_redactions must be called per-page AFTER all annotations
        # on that page are added; calling it page-by-page inside the
        # loop above would be fine too, but batching here makes it
        # obvious this is a document-wide commit, not per-span.
        for page_num in spans_by_page:
            doc[page_num].apply_redactions()

        Path(output_pdf_path).parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_pdf_path))
        return applied
    finally:
        doc.close()


def verify_no_residual_text(output_pdf_path: str | Path, forbidden_strings: list[str]) -> list[str]:
    """
    Post-redaction sanity check: re-extracts text from the saved PDF
    and confirms none of the redacted strings survived (byte-erasure
    verification, not just a visual check). Returns the list of
    strings that WERE found (empty list = clean). Intended for use in
    eval/ and tests, not as a runtime gate — a failure here means a
    bug in this module, not something to silently retry.
    """
    doc = fitz.open(str(output_pdf_path))
    try:
        full_text = "".join(page.get_text() for page in doc)
    finally:
        doc.close()

    return [s for s in forbidden_strings if s in full_text]
