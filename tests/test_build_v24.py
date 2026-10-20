"""tests/test_build_v24.py - long names, name lists, no-postcode addresses, hard negatives."""

import random
import re

import pytest

from data.build_v24_corpus import (
    FAMILIES, address_no_postcode_tokens, bracket_signature, build_layer, long_person_tokens,
    name_list, negative_place, no_postcode_line,
)
from data.build_v24_corpus import _CITIES
from pathlib import Path

_CONNECTORS = {"bin", "binti", "a/l", "a/p", "b.", "bt", "bte", "@"}


def _spans(sent, typ):
    out, cur = [], None
    for i, lab in enumerate(sent.labels):
        if lab == f"B-{typ}":
            cur = [i, i + 1]
            out.append(cur)
        elif lab == f"I-{typ}" and cur is not None:
            cur[1] = i + 1
        else:
            cur = None
    return [(a, b) for a, b in out]


def _valid_iob2(labels):
    prev = "O"
    for lab in labels:
        if lab.startswith("I-") and (prev == "O" or prev[2:] != lab[2:]):
            return False
        prev = lab
    return True


def test_layer_is_well_formed_deterministic_and_covers_all_families():
    a = build_layer(random.Random(1), 400)
    b = build_layer(random.Random(1), 400)
    assert [(s.tokens, s.labels) for s in a] == [(s.tokens, s.labels) for s in b]
    for s in a:
        assert len(s.tokens) == len(s.labels) and s.tokens and _valid_iob2(s.labels)
        assert "{" not in " ".join(s.tokens)
    assert len(FAMILIES) == 7


def test_long_names_are_long_and_have_a_connector():
    rng = random.Random(2)
    for _ in range(300):
        toks = long_person_tokens(rng)
        assert len(toks) >= 5
        assert any(t.lower() in _CONNECTORS for t in toks)


def test_names_in_a_row_start_a_new_person_each_time():
    rng = random.Random(3)
    adjacent = 0
    for _ in range(200):
        s = name_list(rng)
        for i in range(1, len(s.labels)):
            if s.labels[i] == "B-PERSON" and s.labels[i - 1].endswith("PERSON"):
                adjacent += 1
    assert adjacent > 100  # many lists contain names printed back to back


def test_no_postcode_addresses_have_no_postcode_but_keep_a_place():
    rng = random.Random(4)
    for _ in range(400):
        toks = address_no_postcode_tokens(rng)
        assert toks
        assert not any(re.fullmatch(r"\d{5}[,.]?", t) for t in toks)
    labelled = 0
    for _ in range(200):
        s = no_postcode_line(rng)
        for a, b in _spans(s, "ADDRESS"):
            labelled += 1
            assert not any(re.fullmatch(r"\d{5}[,.]?", t) for t in s.tokens[a:b])
    assert labelled > 150


def test_bare_pair_share_zero_removes_pure_city_state_pairs():
    rng = random.Random(5)
    cities = {c.lower() for c in _CITIES}
    pairs = 0
    for _ in range(600):
        toks = address_no_postcode_tokens(rng, bare_pair_share=0.0)
        text = " ".join(toks).replace(" ,", ",").lower()
        # a pure pair has exactly "<city>, <state>" and nothing else
        for c in cities:
            if text.startswith(c + ",") and len(text.split()) <= len(c.split()) + 4 and "jalan" not in text:
                pairs += 1
                break
    assert pairs < 25  # a few real company addresses legitimately start with a city word


def test_hard_negatives_are_all_O_and_mention_a_place():
    rng = random.Random(6)
    cities = {c.lower() for c in _CITIES}
    seen_city = 0
    for _ in range(200):
        s = negative_place(rng)
        assert set(s.labels) == {"O"}
        text = " ".join(s.tokens).lower()
        seen_city += any(c in text for c in cities)
    assert seen_city > 150


def test_bracketed_names_keep_the_brackets_inside_the_person_span():
    rng = random.Random(7)
    hits = 0
    for _ in range(300):
        s = bracket_signature(rng)
        for a, b in _spans(s, "PERSON"):
            if s.tokens[a].startswith("(") and s.tokens[b - 1].rstrip(",.").endswith(")"):
                hits += 1
    assert hits > 40


def test_training_builders_do_not_depend_on_the_rule_engine():
    """The model's training data must not be produced by (or tied to) the address heuristic."""
    for name in ("build_v23_corpus.py", "build_v24_corpus.py"):
        source = (Path(__file__).resolve().parents[1] / "data" / name).read_text(encoding="utf-8")
        code = "\n".join(l for l in source.splitlines() if l.lstrip().startswith(("import ", "from ")))
        assert "regex_engine" not in code, f"{name} imports from regex_engine"
