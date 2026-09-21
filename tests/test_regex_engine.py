import fitz
import pytest

from pdf_ingestion.extractor import PDFIngestor
from regex_engine.matcher import match_page, build_regex_mask_registry
from regex_engine.patterns import (
    is_plausible_nric,
    NRIC_PATTERN,
    is_plausible_phone,
    MOBILE_PATTERN,
    LANDLINE_PATTERN,
)


def _pdf_with_text(tmp_path, lines: list[str]):
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line)
        y += 28
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------------
# Pattern-level plausibility checks (no PDF needed)
# ---------------------------------------------------------------------

def test_valid_nric_is_plausible():
    m = NRIC_PATTERN.search("901231-06-5678")
    assert m is not None
    assert is_plausible_nric(m)


def test_nric_without_dashes_matches():
    """Regression test for a real finding: some documents write NRIC
    as 12 consecutive digits with no dashes at all
    ("040215031234" instead of "040215-03-1234"). Dashes are now
    optional in NRIC_PATTERN; the date/pb-code plausibility check
    still gates false positives on arbitrary 12-digit numbers."""
    m = NRIC_PATTERN.search("040215031234")
    assert m is not None
    assert is_plausible_nric(m)


def test_nric_without_dashes_rejects_implausible_12_digit_numbers():
    """A random 12-digit number (tracking ref, account number) that
    doesn't decode to a valid date must still be rejected even though
    the dash requirement was relaxed."""
    for bad in ["999999999999", "123456789012", "000000000000"]:
        m = NRIC_PATTERN.search(bad)
        if m is not None:
            assert not is_plausible_nric(m)


def test_nric_spaced_dash_variants_match():
    """Regression test for real documents using space-padded dashes
    ("881204 - 14 - 6033") or space-only separators with no dash at
    all ("000512 14 6889") — an earlier fix only made the dash
    optional, not space-tolerant, missing both of these entirely."""
    for text in ["881204 - 14 - 6033", "000512 14 6889", "750918 - 10 - 5431"]:
        m = NRIC_PATTERN.search(text)
        assert m is not None, f"failed to match: {text}"
        assert is_plausible_nric(m)


def test_invalid_month_nric_is_not_plausible():
    m = NRIC_PATTERN.search("901331-06-5678")  # month 13
    assert m is not None
    assert not is_plausible_nric(m)


def test_feb29_only_valid_in_leap_years():
    m_leap = NRIC_PATTERN.search("000229-06-5678")  # 2000, leap
    assert is_plausible_nric(m_leap)
    m_non_leap = NRIC_PATTERN.search("010229-06-5678")  # 2001, not leap
    # 1901 also not leap -> should fail in both centuries
    assert not is_plausible_nric(m_non_leap)


def test_mobile_number_matches_with_dashes():
    m = MOBILE_PATTERN.search("012-3456789")
    assert m is not None
    assert is_plausible_phone(m)


def test_mobile_011_extended_matches():
    m = MOBILE_PATTERN.search("011-23456789")
    assert m is not None
    assert is_plausible_phone(m)


def test_landline_klang_valley_matches():
    m = LANDLINE_PATTERN.search("03-12345678")
    assert m is not None
    assert is_plausible_phone(m)


def test_mobile_with_irregular_digit_grouping():
    """Regression test for a real resume PDF finding: phone numbers
    formatted with a space splitting the subscriber digits at a
    non-standard boundary (4+3 instead of the 'usual' 3+4) were
    silently missed entirely before the pattern was made separator-
    position-flexible."""
    for text in ["012-1234 432", "012-1234 321", "016-987 6543"]:
        m = MOBILE_PATTERN.search(text)
        assert m is not None, f"failed to match: {text}"


def test_mobile_with_parenthesized_area_code():
    """Regression test for real documents formatting the prefix in
    parentheses, e.g. "(012) 654 9870" — the pattern previously
    required the prefix digits to start the match with no wrapping
    punctuation at all."""
    for text in ["(012) 654 9870", "(016) 441 9988"]:
        m = MOBILE_PATTERN.search(text) or LANDLINE_PATTERN.search(text)
        assert m is not None, f"failed to match: {text}"
        assert is_plausible_phone(m)


def test_mobile_015_prefix_matches():
    """015 is a real, newer Malaysian MVNO mobile prefix that an
    earlier version of MOBILE_PATTERN's second-digit character class
    excluded entirely (only 0,2,3,4,6,7,8,9 were allowed, missing 5)."""
    for text in ["015-960 1234", "+6015-960 1234"]:
        m = MOBILE_PATTERN.search(text)
        assert m is not None, f"failed to match: {text}"
        assert is_plausible_phone(m)


def test_nric_does_not_hijack_plus_prefixed_phone_numbers():
    """Regression test for a real finding: a +60-prefixed phone number
    whose digits (after stripping the +) happen to start with a
    valid-looking date ("+601110984321" -> 60-11-10) was being
    mis-claimed as NRIC before the leading-plus exclusion was added.
    NRIC is never written with a "+" prefix, so this is a safe,
    unambiguous disambiguator."""
    text = "+601110984321"
    assert NRIC_PATTERN.search(text) is None
    assert MOBILE_PATTERN.search(text) is not None


def test_nric_slash_and_dot_separators_match():
    """Real Malaysian documents/forms sometimes use '/' or '.' as the
    NRIC group separator instead of '-' — found via a formatting
    reference catalogue, not yet seen in earlier PDF tests."""
    for text in ["900101/14/5678", "900101.14.5678"]:
        m = NRIC_PATTERN.search(text)
        assert m is not None, f"failed to match: {text}"
        assert is_plausible_nric(m)


def test_nric_parenthetical_place_of_birth_code_matches():
    """Real format variant: the place-of-birth code wrapped in
    parentheses with no separator elsewhere, e.g. "780412(07)5319"."""
    m = NRIC_PATTERN.search("780412(07)5319")
    assert m is not None
    assert is_plausible_nric(m)


# ---------------------------------------------------------------------
# End-to-end: PDF -> tokens -> regex match -> IOB2 labels on Token objects
# ---------------------------------------------------------------------

def test_nric_labels_single_token(tmp_path):
    pdf = _pdf_with_text(tmp_path, ["NRIC 901231-06-5678 tel 012-3456789"])
    doc = PDFIngestor().extract(pdf)
    page = doc.pages[0]
    matches = match_page(page)

    nric_matches = [m for m in matches if m.label == "NRIC"]
    assert len(nric_matches) == 1
    assert len(nric_matches[0].token_indices) == 1  # no internal space -> 1 token

    idx = nric_matches[0].token_indices[0]
    assert page.tokens[idx].label == "B-NRIC"
    assert page.tokens[idx].source == "regex"


def test_phone_number_split_across_tokens_by_spaces(tmp_path):
    """Regression test for the exact risk flagged during design: a
    phone number written with spaces lands as multiple PyMuPDF word
    tokens, and the matcher must still resolve it to a single entity
    spanning all of them with correct B-/I- tagging."""
    pdf = _pdf_with_text(tmp_path, ["Contact: 012 345 6789 for details"])
    doc = PDFIngestor().extract(pdf)
    page = doc.pages[0]

    # Sanity-check our assumption: PyMuPDF does split this into 3 tokens.
    space_split_tokens = [t for t in page.tokens if t.text in ("012", "345", "6789")]
    assert len(space_split_tokens) == 3

    matches = match_page(page)
    phone_matches = [m for m in matches if m.label == "PHONE"]
    assert len(phone_matches) == 1
    assert len(phone_matches[0].token_indices) == 3

    tagged = [page.tokens[i] for i in phone_matches[0].token_indices]
    assert tagged[0].label == "B-PHONE"
    assert tagged[1].label == "I-PHONE"
    assert tagged[2].label == "I-PHONE"
    assert all(t.source == "regex" for t in tagged)


def test_no_false_positive_on_unrelated_digit_run(tmp_path):
    pdf = _pdf_with_text(tmp_path, ["Invoice number 2024011500123456"])
    doc = PDFIngestor().extract(pdf)
    page = doc.pages[0]
    matches = match_page(page)
    assert matches == []


def test_nric_and_phone_do_not_overlap_claim(tmp_path):
    pdf = _pdf_with_text(tmp_path, ["901231-06-5678"])
    doc = PDFIngestor().extract(pdf)
    page = doc.pages[0]
    matches = match_page(page)
    # must not be double-claimed as a phone number too
    assert [m.label for m in matches] == ["NRIC"]


def test_untouched_tokens_default_to_O(tmp_path):
    pdf = _pdf_with_text(tmp_path, ["Ahmad Bin Abdullah lives in Kuala Lumpur"])
    doc = PDFIngestor().extract(pdf)
    page = doc.pages[0]
    match_page(page)
    assert all(t.label == "O" for t in page.tokens)


# ---------------------------------------------------------------------
# Real examples pulled directly from the FYP report (Ch.2/Ch.3), not
# invented data — Section 2.2.2 (p.19) and Table 3.3 (p.57).
# ---------------------------------------------------------------------

def test_intl_prefix_matches_report_spec_format():
    """FYP report Section 3.2.1 (p.53) explicitly specifies
    '+601X-XXXXXXX' as a required format; local-only patterns miss it."""
    m = MOBILE_PATTERN.search("+6012-3456789")
    assert m is not None
    assert is_plausible_phone(m)


def test_intl_prefix_no_plus_sign():
    m = MOBILE_PATTERN.search("6012-3456789")
    assert m is not None
    assert is_plausible_phone(m)


def test_report_worked_example_from_p19(tmp_path):
    """Verbatim scenario from the report's own before/after masking
    example (Section 2.2.2, p.19): 'Ahmad bin Zulkifli ... IC No:
    980513-14-5123 ... mobile number to 012-3456789.'"""
    pdf = _pdf_with_text(
        tmp_path,
        ["Ahmad bin Zulkifli IC No 980513-14-5123 mobile 012-3456789"],
    )
    doc = PDFIngestor().extract(pdf)
    page = doc.pages[0]
    matches = match_page(page)

    labels = sorted((m.label, m.text) for m in matches)
    assert ("NRIC", "980513-14-5123") in [m for m in labels]
    assert any(m[0] == "PHONE" for m in labels)


def test_report_worked_example_from_table_3_3(tmp_path):
    """Table 3.3 (p.57) worked example: 'Encik Khairul bin Azmi memegang
    NRIC 940312-14-5543' — the NRIC token must land in the
    regex_mask_registry regardless of the honorific/name tokens around
    it (those are Tier 2's job, not Tier 1's)."""
    pdf = _pdf_with_text(
        tmp_path,
        ["Encik Khairul bin Azmi memegang NRIC 940312-14-5543"],
    )
    doc = PDFIngestor().extract(pdf)
    registry = build_regex_mask_registry(doc)

    nric_token = next(
        t for t in doc.pages[0].tokens if t.text == "940312-14-5543"
    )
    assert (0, nric_token.token_index) in registry
    match = registry.get_match(0, nric_token.token_index)
    assert match.label == "NRIC"

    # Honorific/name tokens must NOT be in the registry — Tier 1 only
    # owns NRIC/PHONE; PERSON is Tier 2's responsibility per the report.
    encik_token = next(t for t in doc.pages[0].tokens if t.text == "Encik")
    assert (0, encik_token.token_index) not in registry
