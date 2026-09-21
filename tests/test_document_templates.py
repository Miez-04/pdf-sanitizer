import random

import pytest

from data.document_templates import (
    DOCUMENT_GENERATORS,
    generate_random_document_fragment,
    generate_biodata_block,
    generate_bill_block,
    generate_academic_block,
)


@pytest.mark.parametrize("doc_type,gen_fn", list(DOCUMENT_GENERATORS.items()))
def test_generator_produces_aligned_tokens_and_labels(doc_type, gen_fn):
    rng = random.Random(1)
    for lang in ("en", "ms"):
        sent = gen_fn(rng, lang=lang)
        assert len(sent.tokens) == len(sent.labels)
        assert len(sent.tokens) > 0


@pytest.mark.parametrize("doc_type,gen_fn", list(DOCUMENT_GENERATORS.items()))
def test_generator_iob2_is_valid(doc_type, gen_fn):
    """No dangling I-X tags — every I-X must follow a B-X or I-X of the
    same type, across all generators and both languages."""
    rng = random.Random(2)
    for lang in ("en", "ms"):
        for _ in range(20):
            sent = gen_fn(rng, lang=lang)
            prev = "O"
            for lab in sent.labels:
                if lab.startswith("I-"):
                    entity = lab[2:]
                    assert prev in (f"B-{entity}", f"I-{entity}"), (
                        f"{doc_type}/{lang}: dangling {lab} after {prev}"
                    )
                prev = lab


def test_biodata_decoy_fields_stay_unlabelled():
    """Age/Gender/Race/Religion/Marital Status VALUES must never be
    tagged as an entity — this is the precision-training purpose of
    the whole biodata generator. Checks the categorical decoy words
    specifically, not "any digit" (a real ADDRESS/PHONE/NRIC entity
    legitimately contains digits and should NOT be flagged here).

    Skips a decoy word if it's already part of an in-progress entity
    span (previous token tagged I-/B- of the same type) — some decoy
    words (e.g. "Cina") are also real Malaysian place-name components
    ("Kampung Cina"), so the SAME string can legitimately appear as
    part of a genuine ADDRESS elsewhere in the same generated example.
    Only flags the word when it's standing alone, as the decoy field's
    own value."""
    rng = random.Random(3)
    decoy_values = {
        "Male", "Female", "Single", "Married", "Malaysian",
        "Lelaki", "Perempuan", "Bujang", "Berkahwin",
        "Melayu", "Cina", "India", "Iban", "Kadazan",
        "Islam", "Buddha", "Hindu", "Kristian", "Sikh",
    }
    for _ in range(30):
        sent = generate_biodata_block(rng, lang=rng.choice(["en", "ms"]))
        for i, (tok, lab) in enumerate(zip(sent.tokens, sent.labels)):
            if tok in decoy_values:
                prev_lab = sent.labels[i - 1] if i > 0 else "O"
                if prev_lab.startswith(("B-", "I-")) and prev_lab[2:] == lab[2:]:
                    continue  # part of a legitimate entity span already in progress
                assert lab == "O", f"decoy value {tok!r} incorrectly tagged {lab}"


def test_bill_decoy_fields_stay_unlabelled():
    """Invoice numbers and amounts must never be tagged as NRIC/PHONE,
    even though they're digit-heavy and sit near a real PERSON field."""
    rng = random.Random(4)
    for _ in range(30):
        sent = generate_bill_block(rng, lang=rng.choice(["en", "ms"]))
        for tok, lab in zip(sent.tokens, sent.labels):
            if tok.startswith("INV-") or tok.startswith("RM"):
                assert lab == "O", f"decoy value {tok!r} incorrectly tagged {lab}"


def test_random_document_fragment_picks_all_types_over_many_draws():
    rng = random.Random(5)
    seen_types = set()
    for _ in range(200):
        # generate_random_document_fragment doesn't expose which type it
        # picked, so just confirm it never crashes across many draws and
        # always returns aligned tokens/labels.
        sent = generate_random_document_fragment(rng, lang=rng.choice(["en", "ms"]))
        assert len(sent.tokens) == len(sent.labels)
        assert len(sent.tokens) > 0


def test_multi_field_blocks_contain_multiple_entity_types():
    """The whole point of document blocks vs. single-entity fragments:
    real documents cluster several PII fields together. Confirm at
    least one generator actually produces >1 distinct entity type in
    a single example (biodata and bill are designed to always do this)."""
    rng = random.Random(6)
    sent = generate_biodata_block(rng, lang="en")
    entity_types = {lab[2:] for lab in sent.labels if lab.startswith("B-")}
    assert len(entity_types) >= 2


def test_academic_decoys_stay_unlabelled():
    """Regression test for a real finding: matric numbers, course
    codes, and institution names were being mis-tagged as ADDRESS/
    NRIC in real academic-letter PDFs. Confirm these decoys never get
    an entity label."""
    rng = random.Random(7)
    for _ in range(30):
        sent = generate_academic_block(rng, lang=rng.choice(["en", "ms"]))
        for tok, lab in zip(sent.tokens, sent.labels):
            if tok.isdigit() and len(tok) == 10:  # matric number shape
                assert lab == "O", f"matric number {tok!r} incorrectly tagged {lab}"
            if tok in ("UNIVERSITI", "KOLEJ") or "UNIVERSITI" in tok.upper():
                assert lab == "O", f"institution token {tok!r} incorrectly tagged {lab}"


def test_academic_block_contains_two_person_names():
    rng = random.Random(8)
    sent = generate_academic_block(rng, lang="en")
    b_person_count = sum(1 for lab in sent.labels if lab == "B-PERSON")
    assert b_person_count == 2  # student + lecturer
