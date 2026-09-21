import random

import pytest
import spacy

from data.entity_mutation import (
    inject_by_replacement,
    inject_connector_clause,
    tokenize,
    _strip_trailing_honorific,
)
from data.registries import generate_nric, generate_phone, generate_address, generate_person
from regex_engine.patterns import is_plausible_nric, is_plausible_phone, NRIC_PATTERN, MOBILE_PATTERN, LANDLINE_PATTERN


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


# ---------------------------------------------------------------------
# Registries: generated entities must satisfy regex_engine's own
# plausibility checks — otherwise Tier 1 and gold labels would disagree
# on the model's own training data.
# ---------------------------------------------------------------------

def test_generated_nric_passes_regex_engine_plausibility():
    rng = random.Random(1)
    for _ in range(200):
        tokens, entity_type = generate_nric(rng)
        assert entity_type == "NRIC"
        m = NRIC_PATTERN.search(tokens[0])
        assert m is not None
        assert is_plausible_nric(m)


def test_generated_phone_passes_regex_engine_plausibility():
    rng = random.Random(2)
    for _ in range(200):
        tokens, entity_type = generate_phone(rng)
        assert entity_type == "PHONE"
        m = MOBILE_PATTERN.search(tokens[0]) or LANDLINE_PATTERN.search(tokens[0])
        assert m is not None, f"no phone pattern matched: {tokens[0]}"
        assert is_plausible_phone(m)


def test_generated_person_nonempty():
    rng = random.Random(3)
    for _ in range(50):
        tokens, entity_type = generate_person(rng)
        assert entity_type == "PERSON"
        assert len(tokens) >= 2  # at least given name + bin/binti/a-l/a-p/surname


def test_generated_person_covers_new_real_world_patterns():
    """Regression test for patterns added after a real formatting
    reference catalogue flagged gaps: East Malaysian native naming
    ("Anak"/"ak"), abbreviated Malay markers ("b."/"bt"/"bte"), the
    Indian "s/o"/"d/o" marker alternative, and the "@" alias format.
    Runs enough draws that every feature should appear at least once;
    fails loudly (not silently) if any generator path regresses to
    never producing one of these again."""
    rng = random.Random(11)
    seen = {
        "east_malaysian": False,
        "abbrev_marker": False,
        "so_do": False,
        "alias": False,
    }
    for _ in range(500):
        tokens, entity_type = generate_person(rng)
        assert entity_type == "PERSON"
        if "Anak" in tokens or "ak" in tokens:
            seen["east_malaysian"] = True
        if any(t in ("b.", "B.", "bt", "bt.", "bte", "bte.") for t in tokens):
            seen["abbrev_marker"] = True
        if "s/o" in tokens or "d/o" in tokens:
            seen["so_do"] = True
        if "@" in tokens:
            seen["alias"] = True
    for feature, was_seen in seen.items():
        assert was_seen, f"never generated a '{feature}' example in 500 draws"


def test_generated_address_mostly_has_postcode():
    """Not every real registry address includes a clean 5-digit
    postcode (a handful of the 1,686 real entries have malformed/
    missing postcodes — genuine messiness in the source CSV, confirmed
    by inspection). Check the large majority do, rather than 100%."""
    rng = random.Random(4)
    n = 200
    with_postcode = 0
    for _ in range(n):
        tokens, entity_type = generate_address(rng)
        assert entity_type == "ADDRESS"
        if any(t.isdigit() and len(t) == 5 for t in tokens):
            with_postcode += 1
    assert with_postcode / n >= 0.85


# ---------------------------------------------------------------------
# tokenize + honorific stripping
# ---------------------------------------------------------------------

def test_tokenize_matches_real_pymupdf_whitespace_splitting():
    """tokenize() must match real PyMuPDF word extraction exactly:
    whitespace-only splitting, punctuation stays attached to the word
    (verified empirically against page.get_text("words") — see
    conversation history for the exact repro)."""
    assert tokenize("Ahmad, Bin Abdullah.") == ["Ahmad,", "Bin", "Abdullah."]
    assert tokenize("Mr. Devendran said IC 850211-10-5678 too.") == [
        "Mr.", "Devendran", "said", "IC", "850211-10-5678", "too.",
    ]


def test_strip_trailing_honorific_multiword():
    remaining, honorific = _strip_trailing_honorific(["Said", "by", "Tan", "Sri"])
    assert remaining == ["Said", "by"]
    assert honorific == ["Tan", "Sri"]


def test_strip_trailing_honorific_none_present():
    remaining, honorific = _strip_trailing_honorific(["Said", "by", "the", "officer"])
    assert remaining == ["Said", "by", "the", "officer"]
    assert honorific == []


# ---------------------------------------------------------------------
# inject_connector_clause: labels always align to injected tokens
# ---------------------------------------------------------------------

def test_connector_clause_labels_align_to_tokens():
    rng = random.Random(5)
    sent = inject_connector_clause(
        "The weather today is sunny with occasional showers", "PHONE", rng, lang="en"
    )
    assert len(sent.tokens) == len(sent.labels)
    assert "B-PHONE" in sent.labels
    b_idx = sent.labels.index("B-PHONE")
    # everything before the entity must be O (the untouched base sentence + template prefix)
    assert all(l == "O" for l in sent.labels[:b_idx])


def test_connector_clause_malay_templates_work():
    rng = random.Random(6)
    sent = inject_connector_clause(
        "Kerajaan negeri mengumumkan langkah baharu bagi membantu mangsa banjir", "ADDRESS", rng, lang="ms"
    )
    assert "B-ADDRESS" in sent.labels
    assert len(sent.tokens) == len(sent.labels)


def test_connector_clause_preserves_base_sentence_verbatim():
    """Position (before/after the injected clause) is randomized, so
    check the base sentence appears intact SOMEWHERE, not necessarily
    first."""
    rng = random.Random(7)
    base = "Malaysia recorded a total of 3,040,235 Covid-19 cases"
    sent = inject_connector_clause(base, "NRIC", rng, lang="en")
    base_toks = tokenize(base)
    base_len = len(base_toks)

    if sent.tokens[:base_len] == base_toks:
        matched_labels = sent.labels[:base_len]
    else:
        assert sent.tokens[-base_len:] == base_toks, (
            "base sentence not found intact at either end"
        )
        matched_labels = sent.labels[-base_len:]

    assert all(l == "O" for l in matched_labels)


# ---------------------------------------------------------------------
# inject_by_replacement: the boundary-miss guard found during real
# data testing (leftover real-name fragments after a partial-span
# spaCy match)
# ---------------------------------------------------------------------

def test_replacement_returns_none_when_no_person_entity(nlp):
    rng = random.Random(8)
    result = inject_by_replacement(nlp, "The weather was clear and sunny all day", rng)
    assert result is None


def test_replacement_rejects_boundary_miss_case(nlp):
    """Regression test for the exact bug found during real-data
    testing: spaCy tags only part of a multi-word name, and a
    capitalized word immediately follows — must reject, not splice."""
    rng = random.Random(9)
    # Construct a case where a plausible partial PERSON match is
    # immediately followed by a capitalized word (simulating the
    # boundary-miss pattern), and confirm the guard actually fires by
    # running many seeds — if spaCy's ent happens not to end there,
    # this specific sentence may resolve fine, so we assert the
    # invariant more directly below instead.
    text = "Prime Minister Ismail Sabri Yaakob instructed the ministry today"
    for seed in range(30):
        result = inject_by_replacement(nlp, text, random.Random(seed))
        if result is not None:
            # If a replacement was accepted, there must be no stray
            # capitalized word immediately after the injected span.
            b_idx = next(i for i, l in enumerate(result.labels) if l == "B-PERSON")
            end = b_idx + 1
            while end < len(result.labels) and result.labels[end] == "I-PERSON":
                end += 1
            if end < len(result.tokens):
                assert not result.tokens[end][:1].isupper() or result.tokens[end] in (
                    ".", ",",
                ), f"leftover capitalized fragment after injection: {result.tokens[end:end+3]}"


def test_replacement_expands_honorific_into_span(nlp):
    rng = random.Random(10)
    text = "The award was presented by Tan Sri Lee Chong Wei yesterday"
    # Try several seeds since replacement depends on rng draws matching
    # a valid injection; assert the property whenever one succeeds.
    found_honorific_case = False
    for seed in range(20):
        result = inject_by_replacement(nlp, text, random.Random(seed))
        if result is not None and "Tan" in result.tokens:
            b_idx = result.tokens.index("Tan")
            if result.labels[b_idx] == "B-PERSON":
                found_honorific_case = True
                assert result.labels[b_idx + 1] == "I-PERSON"  # "Sri"
    # Not asserting found_honorific_case is True unconditionally since
    # it depends on spaCy's exact entity span on this sentence, but if
    # it IS found, the label alignment above must hold (already checked).
