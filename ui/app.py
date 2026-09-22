"""
ui/app.py

Human-in-the-Loop review workspace (README roadmap item 7). Run with:

    streamlit run ui/app.py

Workflow, matching the README's architecture diagram exactly:
    PDF upload -> pdf_ingestion -> Tier 1 + Tier 2 -> conflict
    resolution -> spans rendered as boxes on each page -> human
    accepts / rejects / manually adds boxes -> redaction applied only
    to the human-approved final list -> download the redacted PDF.

This is deliberately the ONLY place in the whole system where
redaction actually happens without a human already having reviewed
the box list — pipeline.run.sanitize_pdf() supports skipping review
for batch/eval use, but this app always requires it, because the
README describes HITL review as a required gate, not an optional one.
"""

from __future__ import annotations

import io
from pathlib import Path

import fitz
import streamlit as st
from PIL import Image

from pdf_ingestion.extractor import PDFIngestor
from pipeline.conflict_resolution import resolve_document
from pipeline.coordinate_merge import EntitySpan, merge_document_spans
from pipeline.inference import TierTwoPredictor
from pipeline.redaction import apply_redactions
from regex_engine.matcher import build_regex_mask_registry

# Edit this to point at whichever checkpoint you want the reviewer to
# use — v4 is the current recommended checkpoint (best adversarial
# score among the checkpoints trained during this project; see
# eval/results/ for the comparison).
DEFAULT_CHECKPOINT = "models/checkpoints_v18/best_model.pt"

ENTITY_COLORS = {
    "PERSON": (0.78, 0.64, 0.88),
    "ADDRESS": (0.94, 0.50, 0.50),
    "NRIC": (0.56, 0.93, 0.56),
    "PHONE": (1.0, 1.0, 0.40),
}


@st.cache_resource
def load_predictor(checkpoint_path: str) -> TierTwoPredictor:
    return TierTwoPredictor(checkpoint_path)


def run_detection(pdf_path: str, checkpoint_path: str) -> tuple:
    """Runs ingestion + Tier 1 + Tier 2 + conflict resolution, returns
    (document, spans). No redaction happens here — that's the point."""
    document = PDFIngestor().extract(pdf_path)
    predictor = load_predictor(checkpoint_path)

    registry = build_regex_mask_registry(document)
    tier2_labels_by_page = predictor.predict_document(document)
    resolve_document(document, registry, tier2_labels_by_page)

    spans = merge_document_spans(document)
    return document, spans


def render_page_with_boxes(
    pdf_path: str, page_num: int, spans: list[EntitySpan], approved: set[int], dpi: int = 150
) -> Image.Image:
    """Draws a real PDF highlight annotation (translucent color behind
    the text, like a highlighter pen — not an opaque box) for every
    span on this page, onto a throwaway copy of the page, then
    rasterizes to a PIL image. Approved spans get full opacity;
    rejected ones are drawn faint so the reviewer can still see what
    was excluded. The source PDF itself is never modified here."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num]
        for i, span in enumerate(spans):
            if span.page_num != page_num:
                continue
            rect = fitz.Rect(*span.bbox.as_tuple())
            color = ENTITY_COLORS.get(span.entity_type, (0.5, 0.5, 0.5))
            annot = page.add_highlight_annot(rect)
            annot.set_colors(stroke=color)
            annot.set_opacity(0.55 if i in approved else 0.15)
            annot.update()

        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return img
    finally:
        doc.close()


def add_manual_span(
    pdf_path: str, page_num: int, search_text: str, entity_type: str
) -> EntitySpan | None:
    """Blueprint requires the HITL step let a human ADD boxes, not
    just accept/reject detected ones — e.g. for PII the pipeline
    missed entirely. Uses PyMuPDF's own text search to find the
    bounding box for arbitrary text the reviewer names, rather than
    asking them to type raw coordinates.

    search_for() returns ONE RECT PER LINE when the match spans a line
    break (verified: a two-line address search returns 2 Rects, not
    1) — an earlier version of this function only used hits[0] and
    would silently redact just the first line of a multi-line address.
    Now unions every returned rect into a single bounding box, so the
    manual span always covers the FULL match regardless of how many
    lines it spans."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num]
        hits = page.search_for(search_text)
        if not hits:
            return None

        from pdf_ingestion.schema import BBox

        bbox = BBox(hits[0].x0, hits[0].y0, hits[0].x1, hits[0].y1)
        for rect in hits[1:]:
            bbox = bbox.union(BBox(rect.x0, rect.y0, rect.x1, rect.y1))

        return EntitySpan(
            entity_type=entity_type,
            text=search_text,
            bbox=bbox,
            page_num=page_num,
            token_indices=(),
            source="manual",
        )
    finally:
        doc.close()


def main():
    st.set_page_config(page_title="PDF Sanitizer — HITL Review", layout="wide")
    st.title("PDF Sanitizer — Human-in-the-Loop Review")

    checkpoint_path = st.sidebar.text_input("Model checkpoint", value=DEFAULT_CHECKPOINT)
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded is None:
        st.info("Upload a PDF to begin.")
        return

    if not Path(checkpoint_path).exists():
        st.error(f"Checkpoint not found: {checkpoint_path}. Train a model first (see models/train.py).")
        return

    # Persist the upload to disk once per new file (session_state keyed
    # by filename so re-running the script on interaction doesn't
    # re-run detection every time).
    if st.session_state.get("_uploaded_name") != uploaded.name:
        tmp_path = Path("ui/_tmp_uploads")
        tmp_path.mkdir(parents=True, exist_ok=True)
        pdf_path = tmp_path / uploaded.name
        pdf_path.write_bytes(uploaded.getvalue())

        with st.spinner("Running ingestion + Tier 1 + Tier 2 + conflict resolution..."):
            document, spans = run_detection(str(pdf_path), checkpoint_path)

        st.session_state["_uploaded_name"] = uploaded.name
        st.session_state["_pdf_path"] = str(pdf_path)
        st.session_state["_spans"] = spans
        st.session_state["_approved"] = set(range(len(spans)))  # all accepted by default
        st.session_state["_num_pages"] = document.num_pages

    pdf_path = st.session_state["_pdf_path"]
    spans: list[EntitySpan] = st.session_state["_spans"]
    approved: set[int] = st.session_state["_approved"]
    num_pages: int = st.session_state["_num_pages"]

    st.success(f"Detected {len(spans)} potential PII span(s) across {num_pages} page(s).")

    col_view, col_review = st.columns([2, 1])

    with col_view:
        page_num = st.number_input("Page", min_value=0, max_value=max(num_pages - 1, 0), value=0)
        img = render_page_with_boxes(pdf_path, page_num, spans, approved)
        st.image(img, width="stretch")
        st.caption(
            "Bright highlight = approved, will be redacted. Faint highlight = rejected, left as-is. "
            + " / ".join(f"{k}: {v}" for k, v in {
                "PERSON": "purple", "ADDRESS": "red", "NRIC": "green", "PHONE": "yellow"
            }.items())
        )

    with col_review:
        st.subheader("Review detected spans")
        for i, span in enumerate(spans):
            checked = st.checkbox(
                f"[{span.entity_type}] \"{span.text}\" (page {span.page_num}, {span.source})",
                value=(i in approved),
                key=f"span_{i}",
            )
            if checked:
                approved.add(i)
            else:
                approved.discard(i)

        st.divider()
        st.subheader("Add a missed entity manually")
        with st.form("add_manual"):
            manual_text = st.text_input("Exact text to redact (must appear on the current page)")
            manual_type = st.selectbox("Entity type", ["PERSON", "ADDRESS", "NRIC", "PHONE"])
            submitted = st.form_submit_button("Find and add")
            if submitted and manual_text:
                new_span = add_manual_span(pdf_path, page_num, manual_text, manual_type)
                if new_span is None:
                    st.warning(f"Text \"{manual_text}\" not found on page {page_num}.")
                else:
                    spans.append(new_span)
                    approved.add(len(spans) - 1)
                    st.success(f"Added: [{manual_type}] \"{manual_text}\"")
                    st.rerun()

        st.divider()
        if st.button("Apply redactions to approved spans", type="primary"):
            final_spans = [spans[i] for i in sorted(approved)]
            output_path = Path(pdf_path).with_name(f"redacted_{Path(pdf_path).name}")

            with st.spinner("Redacting — replacing each approved span with its [ENTITY_TYPE] label..."):
                applied = apply_redactions(pdf_path, str(output_path), final_spans, label_style=True)

            st.success(f"Redacted {applied} span(s). Download below.")
            st.download_button(
                "Download redacted PDF",
                data=output_path.read_bytes(),
                file_name=f"redacted_{Path(pdf_path).name}",
                mime="application/pdf",
            )


if __name__ == "__main__":
    main()
