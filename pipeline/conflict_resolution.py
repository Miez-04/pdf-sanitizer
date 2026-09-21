"""
pipeline.conflict_resolution

Implements the FYP pseudocode's gate directly:

    FOR each token index ti:
        IF ti IN regex_mask_registry:
            label[ti] = regex_mask_registry.get(ti).label   # Tier 1 wins
        ELSE:
            label[ti] = tier2_prediction[ti]                # Tier 1 silent -> Tier 2

Edge case the pseudocode doesn't cover, found while testing this on
real multi-entity sentences: Tier 1 can force a token to B-NRIC/B-PHONE
in the *middle* of what Tier 2 predicted as a continuous I-PERSON /
I-ADDRESS span. That leaves a dangling "I-X" token whose preceding
token is no longer B-X or I-X of the same type — invalid IOB2, and it
would merge into the wrong entity in pipeline.coordinate_merge (or get
silently dropped depending on how strict the merge is). repair_iob2()
below is a required second pass, not optional cleanup: run it every
time, not just when something looks wrong.
"""

from __future__ import annotations

from pdf_ingestion.schema import DocumentTokens, PageTokens
from regex_engine.address_heuristic import find_address_spans
from regex_engine.matcher import RegexMaskRegistry
from regex_engine.patterns import (
    NRIC_PATTERN,
    PHONE_PATTERNS,
    is_plausible_nric,
    is_plausible_phone,
)

MODEL_SOURCE = "model"
REGEX_SOURCE = "regex"
HEURISTIC_SOURCE = "address_heuristic"


def resolve_page(
    page: PageTokens,
    registry: RegexMaskRegistry,
    tier2_labels: list[str],
) -> None:
    """Mutates page.tokens' label/source in place. Tokens already
    labelled by regex_engine.matcher (source == "regex") are left
    untouched — that IS the "Tier 1 wins" rule, already applied at
    match time. Every other token gets its Tier 2 prediction."""
    assert len(tier2_labels) == len(page.tokens), (
        f"tier2_labels length {len(tier2_labels)} != token count "
        f"{len(page.tokens)} for page {page.page_num}"
    )

    for token, predicted_label in zip(page.tokens, tier2_labels):
        if (page.page_num, token.token_index) in registry:
            continue  # Tier 1 already claimed this token; do not overwrite
        token.label = predicted_label
        token.source = MODEL_SOURCE

    _reject_implausible_model_nric_phone(page)
    repair_iob2(page)
    _fill_address_gaps_with_heuristic(page)


def _fill_address_gaps_with_heuristic(page: PageTokens) -> int:
    """
    Address heuristic (regex_engine.address_heuristic) runs LAST, as a
    pure gap-filler — NOT with NRIC/PHONE's "always wins" priority.

    Found necessary after an eval regression: giving the address
    heuristic the same absolute priority as NRIC/PHONE regex caused it
    to OVERRIDE already-correct model predictions whenever its own
    span boundary (anchor-keyword-based, approximate) differed by even
    one token from the model's — seqeval's exact-span matching then
    scored that as both a full miss AND a full false positive, despite
    most tokens agreeing. NRIC/PHONE regex is boundary-exact by
    construction (fixed digit-count patterns), so "always wins" is
    safe there; this heuristic is not, so it only fills spans that are
    CURRENTLY ENTIRELY "O" — i.e. genuine gaps the model missed
    completely (the real resume-PDF failure mode this was built for),
    never touching a token the model already made any prediction for.
    """
    filled = 0
    token_texts = [t.text for t in page.tokens]
    for start_idx, end_idx in find_address_spans(token_texts):
        span = page.tokens[start_idx:end_idx]
        # Allow completing/correcting a PARTIAL address-only guess from
        # the model (found necessary: the model sometimes tags just 2
        # of a real 7-token address as ADDRESS with the wrong boundary,
        # which blocked the heuristic from supplying the full, correct
        # span under a strict "only touch tokens still O" rule) — but
        # still never override a DIFFERENT entity type the model
        # predicted, which is what the original override bug was about.
        if any(t.label != "O" and not t.label.endswith("ADDRESS") for t in span):
            continue
        for i, token in enumerate(span):
            token.label = "B-ADDRESS" if i == 0 else "I-ADDRESS"
            token.source = HEURISTIC_SOURCE
            filled += 1
    return filled


def _reject_implausible_model_nric_phone(page: PageTokens) -> int:
    """Cross-checks the model's OWN NRIC/PHONE predictions (tokens
    Tier 1 didn't independently claim) against Tier 1's own format
    rules, downgrading implausible ones to "O".

    Rationale: Tier 1's regex is comprehensive for NRIC/PHONE
    (including flexible-separator phone matching), so a model-only
    NRIC/PHONE guess that fails Tier 1's own plausibility check is far
    more likely a false positive on an unrelated digit-heavy token
    (student IDs, course codes, reference numbers — confirmed via real
    test PDFs where e.g. a 10-digit matric number got mis-tagged) than
    a genuine number Tier 1 somehow missed. The model brings real,
    complementary value for PERSON/ADDRESS, where no such deterministic
    check exists — this filter is specific to NRIC/PHONE, where one
    does. Downgrades the WHOLE contiguous model-predicted span at once
    (not token-by-token) so a multi-token span isn't left partially
    mangled. Returns the number of tokens downgraded."""
    downgraded = 0
    i = 0
    n = len(page.tokens)
    while i < n:
        token = page.tokens[i]
        entity = token.label[2:] if token.label.startswith("B-") else None
        if token.source == MODEL_SOURCE and entity in ("NRIC", "PHONE"):
            span = [token]
            j = i + 1
            while (
                j < n
                and page.tokens[j].source == MODEL_SOURCE
                and page.tokens[j].label == f"I-{entity}"
            ):
                span.append(page.tokens[j])
                j += 1

            span_text = " ".join(t.text for t in span)
            if entity == "NRIC":
                m = NRIC_PATTERN.search(span_text)
                plausible = bool(m and is_plausible_nric(m))
            else:
                plausible = any(
                    (m := pattern.search(span_text)) and is_plausible_phone(m)
                    for pattern in PHONE_PATTERNS
                )

            if not plausible:
                for t in span:
                    t.label = "O"
                downgraded += len(span)
            i = j
        else:
            i += 1
    return downgraded


def repair_iob2(page: PageTokens) -> int:
    """Fixes dangling I-X tokens created when a regex hard-override
    lands inside a Tier-2-predicted span, or when Tier 2 itself emits
    an invalid sequence (BiLSTM-CRF with a full transition matrix can
    still do this on a genuinely uncertain token). A dangling I-X
    becomes B-X: it's still evidence of that entity type at that
    position, just no longer "continuing" the previous token's entity.
    Returns the number of tokens repaired (useful for eval/ logging)."""
    repaired = 0
    prev_label = "O"
    for token in page.tokens:
        label = token.label
        if label.startswith("I-"):
            entity = label[2:]
            valid_predecessor = prev_label in (f"B-{entity}", f"I-{entity}")
            if not valid_predecessor:
                token.label = f"B-{entity}"
                repaired += 1
        prev_label = token.label
    return repaired


def resolve_document(
    document: DocumentTokens,
    registry: RegexMaskRegistry,
    tier2_labels_by_page: list[list[str]],
) -> None:
    for page, tier2_labels in zip(document.pages, tier2_labels_by_page):
        resolve_page(page, registry, tier2_labels)
