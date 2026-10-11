import random

import pytest

from data.entity_mutation import (
    find_person_spans,
    inject_by_replacement,
    inject_connector_clause,
    tokenize,
    _strip_trailing_honorific,
)
from data.registries import generate_nric, generate_phone, generate_address, generate_person
from regex_engine.patterns import is_plausible_nric, is_plausible_phone, NRIC_PATTERN, MOBILE_PATTERN, LANDLINE_PATTERN


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
# find_person_spans: rule-based detector (no trained model)
# ---------------------------------------------------------------------

def _spans_text(text: str) -> list[str]:
    tokens = tokenize(text)
    return [" ".join(tokens[a:b]) for a, b in find_person_spans(tokens)]


def test_title_led_span_excludes_the_title():
    assert _spans_text("Datuk Seri Anwar Ibrahim said the budget would be tabled .") == ["Anwar Ibrahim"]
    assert _spans_text("The award was presented by Tan Sri Lee Chong Wei yesterday") == ["Lee Chong Wei"]


def test_bare_tan_is_a_surname_not_a_title():
    # "Tan" only counts as a title in "Tan Sri"; no anchor, no registry name -> no span.
    assert _spans_text("Tan Ah Kow met Hong Leong Bank officials") == []


def test_patronym_led_spans():
    assert _spans_text("Ahmad bin Ali was arrested and Kumar a/l Raju was freed") == [
        "Ahmad bin Ali", "Kumar a/l Raju",
    ]


def test_registry_anchored_span_needs_two_name_words():
    assert _spans_text("Nurul Huda Hassan won the title") == ["Nurul Huda Hassan"]
    assert _spans_text("Hakim said the market was calm") == []


def test_lone_initial_is_not_a_name():
    assert _spans_text("Datuk M. said the plan would go ahead") == []


def test_all_caps_dateline_is_not_a_name_word():
    # "Yaakob KOTA KINABALU:" - the dateline of the next article must not be absorbed.
    assert _spans_text("Datuk Seri Shahrul Ikram Yaakob KOTA KINABALU: said so") == ["Shahrul Ikram Yaakob"]


def test_place_and_venue_words_block_registry_spans():
    assert _spans_text("Fans filled Ibrahim Stadium on Friday") == []
    assert _spans_text("They climbed Puteri Gunung Ledang last week") == []


def test_company_and_market_text_gives_no_person_span():
    assert _spans_text("Sime Darby Plantation lost 35 sen to RM3.45 and Top Glove trimmed") == []
    assert _spans_text("The weather was clear and sunny all day") == []


def test_trailing_punctuation_is_kept_inside_the_span_token():
    # "Abdullah," is one whitespace token; the span includes it, and
    # inject_by_replacement re-emits the comma as its own token.
    assert _spans_text("Tan Sri Dr Noor Hisham Abdullah, said cases fell") == ["Noor Hisham Abdullah,"]


# ---------------------------------------------------------------------
# inject_by_replacement
# ---------------------------------------------------------------------

def test_replacement_returns_none_when_no_person_entity():
    rng = random.Random(8)
    assert inject_by_replacement("The weather was clear and sunny all day", rng) is None


def test_replacement_rejects_boundary_miss_case():
    """Detector stops at an organisation word ('Holdings'): a capitalised
    word right after the span means a real fragment would be left next to
    the synthetic name, so the sentence must be refused, not spliced."""
    for seed in range(20):
        assert inject_by_replacement(
            "Datuk Seri Foo Holdings Berhad announced a merger", random.Random(seed)
        ) is None


def test_replacement_expands_honorific_into_span():
    text = "The award was presented by Tan Sri Lee Chong Wei yesterday"
    for seed in range(10):
        result = inject_by_replacement(text, random.Random(seed))
        assert result is not None
        b_idx = result.tokens.index("Tan")
        assert result.labels[b_idx] == "B-PERSON"
        assert result.labels[b_idx + 1] == "I-PERSON"  # "Sri"
        assert result.tokens[0:4] == ["The", "award", "was", "presented"]
        assert result.labels[:4] == ["O", "O", "O", "O"]
        assert result.tokens[-1] == "yesterday" and result.labels[-1] == "O"
        # real name fully replaced. Check the whole sequence, not single words:
        # the synthetic generator may legitimately emit "Wei" or "Chong" itself.
        joined = " ".join(result.tokens)
        assert "Lee Chong Wei" not in joined


def test_replacement_keeps_trailing_comma_as_separate_o_token():
    result = inject_by_replacement("Tan Sri Dr Noor Hisham Abdullah, said cases fell", random.Random(3))
    assert result is not None
    assert "," in result.tokens
    comma_idx = result.tokens.index(",")
    assert result.labels[comma_idx] == "O"
    assert result.labels[comma_idx - 1].endswith("PERSON")
    assert "Noor Hisham Abdullah" not in " ".join(result.tokens)


def test_replacement_pulls_chained_honorifics_into_span():
    result = inject_by_replacement("Datuk Seri Dr Zambry Abdul spoke today", random.Random(4))
    # no real-name leftovers, and every leading honorific is inside the span
    assert result is not None
    assert result.labels[0] == "B-PERSON" and result.tokens[:3] == ["Datuk", "Seri", "Dr"]
    assert "Zambry" not in result.tokens


# ---------------------------------------------------------------------
# State-name variants in generated addresses
# ---------------------------------------------------------------------

from data.registries import STATE_VARIANTS, generate_address, vary_state_in_address, _tokenize_address


def _variants(text: str, n: int = 80) -> set[str]:
    return {
        " ".join(vary_state_in_address(_tokenize_address(text), random.Random(i), 1.0))
        for i in range(n)
    }


def test_state_gets_full_and_short_variants():
    v = _variants("No. 5, Jalan Mawar, 40000 Shah Alam, Selangor")
    assert any(x.endswith("Selangor Darul Ehsan") for x in v)
    assert any(x.endswith(", Selangor") for x in v)
    v = _variants("12 Jalan Tun, 70000 Seremban, Negeri Sembilan")
    assert any(x.endswith(", N . Sembilan") for x in v) and any(x.endswith(", N9") for x in v)
    assert any("Darul Khusus" in x for x in v)
    v = _variants("5 Jalan Raja, 10250 Georgetown, Pulau Pinang")
    assert any(x.endswith(", Penang") for x in v)


def test_state_variants_leave_cities_and_streets_alone():
    assert _variants("Kampung Y, Kuala Selangor") == {"Kampung Y , Kuala Selangor"}
    assert _variants("5 Jalan Z, 20000 Kuala Terengganu") == {"5 Jalan Z , 20000 Kuala Terengganu"}
    for x in _variants("Lot 2, 80000 Johor Bahru, Johor"):
        assert "Johor Bahru" in x  # the city is never rewritten


def test_state_variants_keep_all_caps_style():
    for x in _variants("45 JALAN X, 70000 SEREMBAN, NEGERI SEMBILAN"):
        assert x == x.upper()


def test_state_variants_probability_zero_changes_nothing():
    toks = _tokenize_address("No. 5, Jalan Mawar, 40000 Shah Alam, Selangor")
    assert vary_state_in_address(toks, random.Random(1), 0.0) == toks


def test_every_variant_is_a_known_state_spelling_and_generator_still_returns_address():
    assert all(v for vs in STATE_VARIANTS.values() for v in vs)
    rng = random.Random(0)
    for _ in range(300):
        tokens, label = generate_address(rng)
        assert label == "ADDRESS" and tokens
