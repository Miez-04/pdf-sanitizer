"""
ui/server.py

Flask backend for the custom PDF Sanitizer review UI (replaces the
Streamlit ui/app.py — see README.md's own note under requirements.txt:
"UI (HITL workspace) — swap for Flask+React later if needed").

Wraps the EXISTING pipeline unchanged:
    pdf_ingestion.PDFIngestor -> regex_engine.build_regex_mask_registry
    -> pipeline.inference.TierTwoPredictor.predict_document_with_confidence
    -> pipeline.conflict_resolution.resolve_document
    -> pipeline.coordinate_merge.merge_document_spans
    -> pipeline.redaction.apply_redactions

Nothing in pdf_ingestion/, regex_engine/, models/, or pipeline/ is
modified by this file — it only orchestrates them and serves the
result over HTTP for ui/static/index.html to render.

Run:
    pip install flask
    python -m ui.server
Then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

import io
import sys
import tempfile
import uuid
from pathlib import Path

# Same reasoning as ui/app.py's bootstrap: pdf_ingestion/pipeline/
# regex_engine are siblings of this ui/ folder, not children of it —
# `python -m ui.server` from the project root already puts the root on
# sys.path, but this guards against running `python ui/server.py`
# directly instead, which wouldn't.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flask import Flask, jsonify, request, send_file, send_from_directory

from pdf_ingestion.extractor import PDFIngestor
from pipeline.conflict_resolution import resolve_document
from pipeline.coordinate_merge import EntitySpan, _split_into_line_runs, merge_document_spans
from pipeline.inference import TierTwoPredictor
from pipeline.redaction import apply_redactions
from regex_engine.matcher import build_regex_mask_registry

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))

DEFAULT_CHECKPOINT = "models/checkpoints_v13/best_model.pt"

# Whether the installed EntitySpan dataclass has a confidence field —
# only true if the confidence-scoring pipeline patch has been applied.
# Checked once so mask_word() below can construct EntitySpan correctly
# either way, matching the same getattr-based read in _span_to_dict.
import dataclasses as _dataclasses
_ENTITY_SPAN_HAS_CONFIDENCE = "confidence" in {f.name for f in _dataclasses.fields(EntitySpan)}

# In-memory session store. Fine for a single-user local FYP review
# tool; would need a real store (Redis, DB) for multi-user deployment.
DOCUMENTS: dict[str, dict] = {}

_predictor_cache: dict[str, TierTwoPredictor] = {}


def _get_predictor(checkpoint_path: str) -> TierTwoPredictor:
    if checkpoint_path not in _predictor_cache:
        _predictor_cache[checkpoint_path] = TierTwoPredictor(checkpoint_path)
    return _predictor_cache[checkpoint_path]


def _span_to_dict(span_id: str, span: EntitySpan, line_no: int) -> dict:
    # getattr, not span.confidence: EntitySpan only carries a confidence
    # field if the confidence-scoring pipeline patch has been applied.
    # Defaults to 1.0 (== "certain") on a plain/pristine pipeline so
    # this UI works either way rather than hard-requiring that patch.
    confidence = getattr(span, "confidence", 1.0)
    return {
        "id": span_id,
        "text": span.text,
        "type": span.entity_type,
        "page": span.page_num + 1,  # 1-indexed for the UI
        "line": line_no,
        # One rect PER VISUAL LINE the span crosses (see
        # pipeline.coordinate_merge.EntitySpan.bboxes) — a wrapped
        # multi-line address is 2+ boxes here, not one box spanning
        # both lines plus everything in between them.
        "bboxes": [
            {"x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1} for b in span.bboxes
        ],
        "source": span.source,
        "confidence": round(confidence, 3),
    }


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "No file uploaded"}), 400

    checkpoint_path = request.form.get("checkpoint", DEFAULT_CHECKPOINT)

    doc_id = uuid.uuid4().hex
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"pdfsan_{doc_id}_"))
    pdf_path = tmp_dir / "input.pdf"
    file.save(str(pdf_path))

    document = PDFIngestor().extract(str(pdf_path))
    predictor = _get_predictor(checkpoint_path)

    registry = build_regex_mask_registry(document)
    if hasattr(predictor, "predict_document_with_confidence"):
        tier2_labels_by_page, tier2_confidences_by_page = (
            predictor.predict_document_with_confidence(document)
        )
        resolve_document(document, registry, tier2_labels_by_page, tier2_confidences_by_page)
    else:
        # Plain/pristine pipeline (confidence-scoring patch not applied)
        # — predict_document/resolve_document only take labels.
        tier2_labels_by_page = predictor.predict_document(document)
        resolve_document(document, registry, tier2_labels_by_page)

    spans = merge_document_spans(document)

    # Approximate "line number" per span (mockup shows "pg 1 · line 3"):
    # rank of the span's top-edge (y0) among all spans on that page,
    # top-to-bottom. Not the PDF's real line index (would need
    # block/line metadata cross-referenced per span, which spans don't
    # currently carry) — an ordinal position readers can still use to
    # locate it on the page at a glance.
    spans_by_page: dict[int, list[EntitySpan]] = {}
    for s in spans:
        spans_by_page.setdefault(s.page_num, []).append(s)
    line_no_by_span_index: dict[int, int] = {}
    for page_num, page_spans in spans_by_page.items():
        ordered = sorted(range(len(page_spans)), key=lambda i: page_spans[i].bbox.y0)
        for rank, idx in enumerate(ordered, start=1):
            line_no_by_span_index[id(page_spans[idx])] = rank

    entities = []
    span_registry: dict[str, EntitySpan] = {}
    for span in spans:
        span_id = uuid.uuid4().hex
        span_registry[span_id] = span
        entities.append(_span_to_dict(span_id, span, line_no_by_span_index[id(span)]))

    DOCUMENTS[doc_id] = {
        "pdf_path": pdf_path,
        "tmp_dir": tmp_dir,
        "document": document,
        "spans_by_id": span_registry,
        "excluded_ids": set(),
        "filename": file.filename,
    }

    return jsonify({
        "doc_id": doc_id,
        "filename": file.filename,
        "num_pages": document.num_pages,
        "pages": [
            {"page": p.page_num + 1, "width": p.page_width, "height": p.page_height}
            for p in document.pages
        ],
        "entities": entities,
    })


@app.route("/api/page/<doc_id>/<int:page_num>.png")
def page_image(doc_id: str, page_num: int):
    """Renders the ORIGINAL (unredacted) page as PNG for review-mode
    display — the frontend overlays entity boxes on top of this via
    CSS, matching the mockup's approach of showing labels layered on
    the real document rather than a pre-redacted image, so toggling
    an entity on/off doesn't require re-rendering the page."""
    entry = DOCUMENTS.get(doc_id)
    if entry is None:
        return "Not found", 404

    import fitz  # local import: keeps the flask process importable
                 # even in an environment where fitz isn't needed yet
    zoom = float(request.args.get("zoom", 1.6))
    doc = fitz.open(str(entry["pdf_path"]))
    try:
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        buf = io.BytesIO(pix.tobytes("png"))
    finally:
        doc.close()
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


ALLOWED_MASK_TYPES = {"PERSON", "NRIC", "PHONE", "ADDRESS", "CUSTOM"}


@app.route("/api/mask-word", methods=["POST"])
def mask_word():
    """'Mask your own words' — exact (case-insensitive) phrase match
    against each page's reconstructed text, across every page. Reuses
    the SAME token list already extracted at upload time (no
    re-ingestion), so bboxes line up with what's already on screen.
    entity_type lets the caller tag the match as one of the real
    PERSON/NRIC/PHONE/ADDRESS types (so it merges into the existing
    filter pills/counts) rather than always being a separate CUSTOM
    bucket; defaults to CUSTOM only if omitted."""
    data = request.get_json(force=True)
    doc_id = data.get("doc_id")
    phrase = (data.get("phrase") or "").strip()
    entity_type = (data.get("entity_type") or "CUSTOM").strip().upper()
    entry = DOCUMENTS.get(doc_id)
    if entry is None:
        return jsonify({"error": "Unknown doc_id"}), 404
    if not phrase:
        return jsonify({"error": "Empty phrase"}), 400
    if entity_type not in ALLOWED_MASK_TYPES:
        return jsonify({"error": f"Invalid entity_type: {entity_type}"}), 400

    document = entry["document"]
    phrase_lower = phrase.lower()
    phrase_words = phrase_lower.split()
    new_entities = []

    for page in document.pages:
        texts_lower = [t.text.lower() for t in page.tokens]
        n = len(phrase_words)
        for i in range(len(texts_lower) - n + 1):
            if texts_lower[i:i + n] != phrase_words:
                continue
            match_tokens = page.tokens[i:i + n]
            bbox = match_tokens[0].bbox
            for t in match_tokens[1:]:
                bbox = bbox.union(t.bbox)
            line_runs = _split_into_line_runs(match_tokens)
            bboxes = []
            for run in line_runs:
                run_bbox = run[0].bbox
                for t in run[1:]:
                    run_bbox = run_bbox.union(t.bbox)
                bboxes.append(run_bbox)
            span_kwargs = dict(
                entity_type=entity_type,
                text=" ".join(t.text for t in match_tokens),
                bbox=bbox,
                bboxes=tuple(bboxes),
                page_num=page.page_num,
                token_indices=tuple(t.token_index for t in match_tokens),
                source="manual",
            )
            if _ENTITY_SPAN_HAS_CONFIDENCE:
                span_kwargs["confidence"] = 1.0
            span = EntitySpan(**span_kwargs)
            span_id = uuid.uuid4().hex
            entry["spans_by_id"][span_id] = span
            new_entities.append(_span_to_dict(span_id, span, line_no=0))

    return jsonify({"entities": new_entities})


@app.route("/api/toggle/<doc_id>/<entity_id>", methods=["POST"])
def toggle_entity(doc_id: str, entity_id: str):
    """Review-before-download: exclude/re-include one detected span
    from the final redaction pass without deleting it from the list."""
    entry = DOCUMENTS.get(doc_id)
    if entry is None or entity_id not in entry["spans_by_id"]:
        return jsonify({"error": "Unknown doc_id/entity_id"}), 404
    data = request.get_json(force=True)
    include = bool(data.get("include", True))
    if include:
        entry["excluded_ids"].discard(entity_id)
    else:
        entry["excluded_ids"].add(entity_id)
    return jsonify({"ok": True})


@app.route("/api/sanitize/<doc_id>", methods=["POST"])
def sanitize(doc_id: str):
    entry = DOCUMENTS.get(doc_id)
    if entry is None:
        return jsonify({"error": "Unknown doc_id"}), 404

    active_spans = [
        span for sid, span in entry["spans_by_id"].items()
        if sid not in entry["excluded_ids"]
    ]
    out_path = entry["tmp_dir"] / "sanitized.pdf"
    count = apply_redactions(str(entry["pdf_path"]), str(out_path), active_spans)
    entry["output_path"] = out_path
    return jsonify({"ok": True, "redacted_count": count})


@app.route("/api/download/<doc_id>")
def download(doc_id: str):
    entry = DOCUMENTS.get(doc_id)
    if entry is None or "output_path" not in entry:
        return "Not sanitized yet", 404
    download_name = f"sanitized_{entry['filename']}"
    return send_file(str(entry["output_path"]), as_attachment=True, download_name=download_name)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
