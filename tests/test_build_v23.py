"""tests/test_build_v23.py - the letter/document-style corpus layer."""

import random

import pytest

from data.build_v23_corpus import (
    attach_punctuation_sentence, attach_punctuation_tokens, build_layer, letter_block,
    given_name_part, render, _strip_honorific,
)
from data.build_v18_corpus import Sentence


def _valid_iob2(labels):
    prev = "O"
    for lab in labels:
        if lab.startswith("I-") and (prev == "O" or prev[2:] != lab[2:]):
            return False
        prev = lab
    return True


def test_layer_is_well_formed_and_deterministic():
    a = build_layer(random.Random(1), 300)
    b = build_layer(random.Random(1), 300)
    assert [(s.tokens, s.labels) for s in a] == [(s.tokens, s.labels) for s in b]
    for s in a:
        assert len(s.tokens) == len(s.labels) and s.tokens
        assert _valid_iob2(s.labels)
        assert "{" not in " ".join(s.tokens), f"unfilled placeholder: {s.tokens}"


def test_all_four_entity_types_and_enough_context_variety():
    layer = build_layer(random.Random(2), 600)
    kinds = {lab[2:] for s in layer for lab in s.labels if lab.startswith("B-")}
    assert kinds == {"PERSON", "ADDRESS", "NRIC", "PHONE"}
    skeletons = {" ".join(t for t, l in zip(s.tokens, s.labels) if l == "O") for s in layer}
    assert len(skeletons) > 300  # many different contexts, not a few repeated shapes


def test_nric_line_matches_the_pdf_token_shape():
    s = render("{P} (MyKad: {NRIC}), {ADDR}", random.Random(3))
    i = s.labels.index("B-NRIC")
    assert s.tokens[i - 1] == "(MyKad:"          # label word is its own O token
    assert s.tokens[i].endswith("),")            # closing bracket + comma stay attached, like PyMuPDF


def test_honorifics_are_outside_the_person_span():
    for seed in range(40):
        s = render("Dear {HON} {PF},", random.Random(seed))
        first_person = s.labels.index("B-PERSON")
        assert all(l == "O" for l in s.labels[:first_person])
        assert s.tokens[1:first_person]  # the honorific tokens exist and are O
        assert s.tokens[0] == "Dear"


def test_signature_repeats_the_sender_name():
    found = 0
    for seed in range(60):
        s = letter_block(random.Random(seed))
        names = []
        cur = []
        for t, l in zip(s.tokens, s.labels):
            if l == "B-PERSON":
                cur = [t]; names.append(cur)
            elif l == "I-PERSON" and cur:
                cur.append(t)
            else:
                cur = []
        flat = [" ".join(n).strip(",.").lower() for n in names]
        if flat and flat[0] == flat[-1] and len(flat) >= 2:
            found += 1
    assert found > 10  # header name and signature are the same person in a good share of letters


def test_attach_punctuation_tokens_and_sentence_labels():
    assert attach_punctuation_tokens(["PAYONG", ",", "KOTA", "BHARU", ",", "(", "x"]) == ["PAYONG,", "KOTA", "BHARU,", "(x"]
    s = Sentence(tokens=["Kuala", "Lumpur", ",", "Selangor", "."],
                 labels=["B-ADDRESS", "I-ADDRESS", "I-ADDRESS", "I-ADDRESS", "O"])
    out = attach_punctuation_sentence(s)
    assert out.tokens == ["Kuala", "Lumpur,", "Selangor."]
    assert out.labels == ["B-ADDRESS", "I-ADDRESS", "I-ADDRESS"]  # 'Selangor.' keeps the entity label


def test_strip_honorific_keeps_surname_tan():
    assert _strip_honorific(["Tan", "Sri", "Lim", "Guan"]) == ["Lim", "Guan"]
    assert _strip_honorific(["Tan", "Mei", "Ling"]) == ["Tan", "Mei", "Ling"]
    assert _strip_honorific(["Encik", "Ali", "bin", "Abu"]) == ["Ali", "bin", "Abu"]


def test_given_name_part_stops_before_the_connector():
    rng = random.Random(0)
    for _ in range(20):
        part = given_name_part(rng, ["Wan", "Ahmad", "Akmal", "Bin", "Wan", "Said"])
        assert "Bin" not in part and part == ["Wan", "Ahmad", "Akmal"][: len(part)]
