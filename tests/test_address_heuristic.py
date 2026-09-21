from data.entity_mutation import tokenize
from regex_engine.address_heuristic import find_address_spans, _is_plausible_postcode


def test_plausible_postcode_range():
    assert _is_plausible_postcode("50480")
    assert _is_plausible_postcode("01000")
    assert not _is_plausible_postcode("99999")  # out of range
    assert not _is_plausible_postcode("123")     # wrong length
    assert not _is_plausible_postcode("abcde")   # not digits


def test_finds_real_address_with_house_number_and_state():
    text = "AHMED ZAINAH 45 Jalan Yap Kwan Seng, 47000 Selangor House phone: 03-1234"
    tokens = tokenize(text)
    spans = find_address_spans(tokens)
    assert len(spans) == 1
    s, e = spans[0]
    assert tokens[s:e] == ["45", "Jalan", "Yap", "Kwan", "Seng,", "47000", "Selangor"]


def test_finds_address_with_no_period_anchor():
    text = "Alamat No. 8, Jalan Meranti 3, 81100 Johor Bahru, Johor"
    tokens = tokenize(text)
    spans = find_address_spans(tokens)
    assert len(spans) >= 1


def test_no_false_positive_on_bare_no_with_unrelated_number():
    """Regression test for a real false positive found during
    development: bare 'no' (without a period) is a common English word
    and must NOT be treated as an address anchor, even when an
    unrelated 5-digit number happens to appear nearby."""
    text = "There is no reason to believe the population reached 45231 residents last year."
    tokens = tokenize(text)
    spans = find_address_spans(tokens)
    assert spans == []


def test_no_false_positive_on_plain_prose_with_postcode_like_number():
    text = "The stadium was completed in 1998 and can hold up to 45000 spectators comfortably."
    tokens = tokenize(text)
    spans = find_address_spans(tokens)
    assert spans == []


def test_no_match_without_nearby_postcode():
    text = "He lives on Jalan Bukit Bintang somewhere in the city centre area."
    tokens = tokenize(text)
    spans = find_address_spans(tokens)
    assert spans == []


def test_two_separate_addresses_in_one_text_both_found():
    text = "First office at Jalan Ampang, 50450 Kuala Lumpur and second at Jalan Tun Razak, 50400 Kuala Lumpur."
    tokens = tokenize(text)
    spans = find_address_spans(tokens)
    assert len(spans) == 2
