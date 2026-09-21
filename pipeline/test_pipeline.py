from pathlib import Path

import fitz
import pytest

from pdf_ingestion.extractor import PDFIngestor
from pdf_ingestion.schema import BBox, PageTokens, Token
from pipeline.conflict_resolution import repair_iob2, resolve_page
from pipeline.coordinate_merge import merge_page_spans
from pipeline.redaction import apply_redactions, verify_no_residual_text
from regex_engine.matcher import build_regex_mask_registry, match_page

CHECKPOINT_PATH = "models/checkpoints/best_model.pt"


def _has_checkpoint() -> bool:
    return Path(CHECKPOINT_PATH).exists()


def _mk_token(idx, text, label="O", source="", x=0.0):
    return Token(
        text=text,
        bbox=BBox(x, 0.0, x + 10.0, 10.0),
        page_num=0,
        block_no=0,
        line_no=0,
        word_no=idx,
        token_index=idx,
        char_start=0,
        char_end=len(text),
        label=label,
        source=source,
    )


# ---------------------------------------------------------------------
# repair_iob2: the dangling-I-tag edge case found during design
# ---------------------------------------------------------------------

def test_repair_iob2_fixes_dangling_i_tag_after_regex_override():
    """B-PERSON I-PERSON [regex forces B-NRIC here] I-PERSON — the last
    I-PERSON no longer follows a PERSON token and must become B-PERSON."""
    page = PageTokens(page_num=0, page_width=600, page_height=800)
    page.tokens = [
        _mk_token(0, "Tan", "B-PERSON", "model"),
        _mk_token(1, "Sri", "I-PERSON", "model"),
        _mk_token(2, "901231-06-5678", "B-NRIC", "regex"),
        _mk_token(3, "Ravi", "I-PERSON", "model"),  # dangling
    ]
    repaired_count = repair_iob2(page)
    assert repaired_count == 1
    assert [t.label for t in page.tokens] == [
        "B-PERSON",
        "I-PERSON",
        "B-NRIC",
        "B-PERSON",
    ]


def test_repair_iob2_leaves_valid_sequences_untouched():
    page = PageTokens(page_num=0, page_width=600, page_height=800)
    page.tokens = [
        _mk_token(0, "Ahmad", "B-PERSON"),
        _mk_token(1, "Bin", "I-PERSON"),
        _mk_token(2, "Abdullah", "I-PERSON"),
        _mk_token(3, "lives", "O"),
    ]
    repaired_count = repair_iob2(page)
    assert repaired_count == 0
    assert [t.label for t in page.tokens] == [
        "B-PERSON",
        "I-PERSON",
        "I-PERSON",
        "O",
    ]


# ---------------------------------------------------------------------
# resolve_page: regex source is never overwritten by tier2 prediction
# ---------------------------------------------------------------------

def test_resolve_page_regex_wins_over_model():
    page = PageTokens(page_num=0, page_width=600, page_height=800)
    page.tokens = [
        _mk_token(0, "901231-06-5678", "B-NRIC", "regex"),
        _mk_token(1, "lives", "O"),
    ]
    registry = type("R", (), {"__contains__": lambda self, k: k == (0, 0)})()
    tier2_labels = ["B-PERSON", "O"]  # model wrongly thinks token 0 is a person
    resolve_page(page, registry, tier2_labels)

    assert page.tokens[0].label == "B-NRIC"  # untouched, regex wins
    assert page.tokens[0].source == "regex"
    assert page.tokens[1].label == "O"
    assert page.tokens[1].source == "model"


# ---------------------------------------------------------------------
# coordinate_merge: bbox union math
# ---------------------------------------------------------------------

def test_merge_page_spans_unions_bbox_correctly():
    page = PageTokens(page_num=0, page_width=600, page_height=800)
    page.tokens = [
        Token("Ahmad", BBox(10, 20, 40, 35), 0, 0, 0, 0, 0, 0, 5, "B-PERSON", "model"),
        Token("Bin", BBox(45, 18, 60, 34), 0, 0, 0, 1, 1, 6, 9, "I-PERSON", "model"),
        Token("Abdullah", BBox(65, 20, 100, 36), 0, 0, 0, 2, 2, 10, 18, "I-PERSON", "model"),
    ]
    spans = merge_page_spans(page)
    assert len(spans) == 1
    span = spans[0]
    assert span.entity_type == "PERSON"
    assert span.text == "Ahmad Bin Abdullah"
    assert span.bbox.as_tuple() == (10, 18, 100, 36)  # min(x0,y0), max(x1,y1)
    assert span.token_indices == (0, 1, 2)


def test_merge_page_spans_separates_distinct_entities():
    page = PageTokens(page_num=0, page_width=600, page_height=800)
    page.tokens = [
        Token("Ahmad", BBox(0, 0, 10, 10), 0, 0, 0, 0, 0, 0, 5, "B-PERSON", "model"),
        Token("lives", BBox(15, 0, 25, 10), 0, 0, 0, 1, 1, 6, 11, "O", ""),
        Token("012-3456789", BBox(30, 0, 60, 10), 0, 0, 0, 2, 2, 12, 23, "B-PHONE", "regex"),
    ]
    spans = merge_page_spans(page)
    assert [s.entity_type for s in spans] == ["PERSON", "PHONE"]


def test_merge_page_spans_raises_on_unrepaired_dangling_i_tag():
    """Contract check: coordinate_merge assumes repair_iob2 already ran."""
    page = PageTokens(page_num=0, page_width=600, page_height=800)
    page.tokens = [
        Token("Ravi", BBox(0, 0, 10, 10), 0, 0, 0, 0, 0, 0, 4, "I-PERSON", "model"),
    ]
    with pytest.raises(AssertionError):
        merge_page_spans(page)


# ---------------------------------------------------------------------
# redaction: real byte-level erasure, verified by re-extracting text
# ---------------------------------------------------------------------

def test_apply_redactions_erases_text_bytes(tmp_path):
    from pipeline.coordinate_merge import EntitySpan

    src = tmp_path / "src.pdf"
    out = tmp_path / "out.pdf"

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "NRIC 901231-06-5678 tel 012-3456789")
    doc.save(str(src))
    doc.close()

    ingested = PDFIngestor().extract(str(src))
    page_tokens = ingested.pages[0]
    match_page(page_tokens)  # tags the NRIC/PHONE tokens via regex

    spans = merge_page_spans(page_tokens)
    assert len(spans) == 2

    applied = apply_redactions(str(src), str(out), spans)
    assert applied == 2

    leaked = verify_no_residual_text(str(out), ["901231-06-5678", "012-3456789"])
    assert leaked == []

    # Non-PII text must survive redaction — this isn't erasing the page.
    remaining = fitz.open(str(out))[0].get_text()
    assert "NRIC" in remaining
    assert "tel" in remaining


def test_apply_redactions_rejects_out_of_range_page(tmp_path):
    from pipeline.coordinate_merge import EntitySpan
    from pipeline.redaction import RedactionError
    from pdf_ingestion.schema import BBox

    src = tmp_path / "src.pdf"
    out = tmp_path / "out.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(src))
    doc.close()

    bad_span = EntitySpan(
        entity_type="NRIC",
        text="x",
        bbox=BBox(0, 0, 10, 10),
        page_num=5,  # doesn't exist
        token_indices=(0,),
        source="regex",
    )
    with pytest.raises(RedactionError):
        apply_redactions(str(src), str(out), [bad_span])


# ---------------------------------------------------------------------
# Full end-to-end: only runs if a real trained checkpoint is present
# ---------------------------------------------------------------------

pytestmark_checkpoint = pytest.mark.skipif(
    not _has_checkpoint(), reason="no trained checkpoint in this environment"
)


@pytestmark_checkpoint
def test_full_pipeline_on_in_distribution_sentence(tmp_path):
    """Regression guard for pipeline wiring (ingestion -> Tier1+Tier2 ->
    conflict resolution -> merge -> redact), using a sentence template
    the model has actually seen. This is NOT a generalization test —
    see the out-of-distribution finding documented in models/ (a model
    trained on this corpus predicted O for every token on a novel,
    differently-phrased sentence despite full word-level vocab
    coverage — a template-memorization gap, not a pipeline bug)."""
    from pipeline.inference import TierTwoPredictor
    from pipeline.run import sanitize_pdf

    src = tmp_path / "src.pdf"
    out = tmp_path / "out.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Applicant name , Datuk Nadhirah binti")
    page.insert_text((72, 100), "Mohamed Ali . is notified .")
    doc.save(str(src))
    doc.close()

    predictor = TierTwoPredictor(CHECKPOINT_PATH)
    spans = sanitize_pdf(str(src), str(out), predictor)

    assert any(s.entity_type == "PERSON" for s in spans)
    leaked = verify_no_residual_text(str(out), ["Nadhirah", "Mohamed Ali"])
    assert leaked == []
