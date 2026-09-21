import fitz
import pytest

from pdf_ingestion.extractor import PDFIngestionError, PDFIngestor
from pdf_ingestion.schema import DocumentTokens


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Ahmad Bin Abdullah, NRIC 901231-06-5678")
    page.insert_text((72, 100), "Jalan Taman Bahagia, Poskod 50480 Kuala Lumpur")
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def blank_pdf(tmp_path):
    path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(path))
    doc.close()
    return path


def test_extract_returns_document_tokens(sample_pdf):
    result = PDFIngestor().extract(sample_pdf)
    assert isinstance(result, DocumentTokens)
    assert result.num_pages == 1
    assert len(result.pages[0].tokens) > 0


def test_char_offsets_align_with_page_text(sample_pdf):
    doc = PDFIngestor().extract(sample_pdf)
    page = doc.pages[0]
    for token in page.tokens:
        assert page.page_text[token.char_start:token.char_end] == token.text


def test_token_index_is_sequential(sample_pdf):
    doc = PDFIngestor().extract(sample_pdf)
    indices = [t.token_index for t in doc.pages[0].tokens]
    assert indices == list(range(len(indices)))


def test_bbox_values_are_plausible(sample_pdf):
    doc = PDFIngestor().extract(sample_pdf)
    page = doc.pages[0]
    for token in page.tokens:
        assert 0 <= token.bbox.x0 < token.bbox.x1 <= page.page_width
        assert 0 <= token.bbox.y0 < token.bbox.y1 <= page.page_height


def test_blank_page_produces_no_tokens_not_a_crash(blank_pdf):
    doc = PDFIngestor().extract(blank_pdf)
    assert doc.pages[0].tokens == []
    assert doc.pages[0].page_text == ""


def test_nonexistent_file_raises_ingestion_error(tmp_path):
    with pytest.raises(PDFIngestionError):
        PDFIngestor().extract(tmp_path / "does_not_exist.pdf")


def test_default_labels_are_unset(sample_pdf):
    doc = PDFIngestor().extract(sample_pdf)
    token = doc.pages[0].tokens[0]
    assert token.label == "O"
    assert token.source == ""
    assert token.confidence == 1.0
