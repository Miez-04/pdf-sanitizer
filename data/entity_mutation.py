"""
data.entity_mutation

FYP Section 3.2.2 Step 2 + 3: injects a synthetic entity from
data.registries into a real, unannotated baseline sentence, and
produces IOB2 tags for the injected span. Because we control exactly
which tokens were inserted, the tags are exact by construction — no
NER model or manual annotation needed, unlike weak/silver labeling.

Two injection strategies, chosen based on what the source language and
sentence content actually support (an honest scoping decision, not
uniform across languages — see build_corpus.py's docstring):

1. Replacement (English only): spaCy finds a real PERSON mention
   already in the sentence; we swap it for a synthetic name, keeping
   100% of the surrounding real grammar/phrasing. Highest quality —
   the "carrier" text was never touched, so there's no plausibility
   risk. Honorifics immediately preceding the detected span (Datuk,
   Tan Sri, Dr, etc.) are pulled into the replaced span too, matching
   the existing corpus's PERSON-includes-honorific convention.

2. Connector-clause injection (English + Malay, all entity types):
   a short natural clause containing the synthetic entity is appended
   to a real base sentence. Used for NRIC/PHONE/ADDRESS always (these
   essentially never occur naturally in news prose) and for PERSON
   when no real name was found to replace. Diversity here comes from
   the real base sentence varying across hundreds/thousands of
   distinct real sentences — the connector clause itself IS a small
   template set, same limitation as the original dataset, just now
   layered onto much more varied carrier context.
"""

from __future__ import annotations

import random
import re

from data.registries import ENTITY_GENERATORS
from models.dataset import Sentence

_TOKEN_RE = re.compile(r"\w+(?:['/]\w+)*|[^\w\s]")

HONORIFIC_TOKENS_LOWER = {
    "encik", "tuan", "datuk", "dato'", "dato", "tan", "sri", "dr", "haji",
    "ir", "cik", "puan", "datin", "prof", "hajjah", "tun",
}
# multi-word honorifics stored as tuples for suffix matching against
# the tail of tokens_before
MULTI_WORD_HONORIFICS = [
    ["tan", "sri"], ["puan", "sri"], ["dato'"], ["dato"], ["datuk"],
    ["tuan"], ["encik"], ["cik"], ["puan"], ["datin"], ["dr"], ["prof"],
    ["haji"], ["hajjah"], ["tun"],
]


def tokenize(text: str) -> list[str]:
    """
    Whitespace-only split, matching PyMuPDF's real word extraction
    EXACTLY (verified empirically: page.get_text("words") splits
    purely on whitespace — "Mr.", "Friday.", "850211-10-5678" all stay
    as single tokens with punctuation attached, never split out).

    This replaces an earlier version that used the _TOKEN_RE word/
    punctuation regex splitter above — that version split "Mr." into
    "Mr"+"." and "850211-10-5678" into 5 pieces, silently training the
    whole corpus at a DIFFERENT token granularity than real inference
    ever produces. Found via an adversarial eval collapse (99.5%
    holdout F1 -> 31% adversarial F1) that traced back to this exact
    mismatch, not to the model failing to generalize. _TOKEN_RE is
    kept above only so the diff is legible, not for active use.
    """
    return text.split()


def _strip_trailing_honorific(tokens_before: list[str]) -> tuple[list[str], list[str]]:
    """If tokens_before ends with a known honorific (possibly
    multi-word), split it off. Returns (remaining_tokens, honorific_tokens)."""
    lower = [t.lower() for t in tokens_before]
    for honorific in sorted(MULTI_WORD_HONORIFICS, key=len, reverse=True):
        n = len(honorific)
        if n <= len(lower) and lower[-n:] == honorific:
            return tokens_before[:-n], tokens_before[-n:]
    return tokens_before, []


def inject_by_replacement(nlp, text: str, rng: random.Random) -> Sentence | None:
    """English-only: replace a real spaCy-detected PERSON span with a
    synthetic name. Returns None if no PERSON entity was found, OR if
    the entity boundary looks unsafe to replace (see below) — callers
    should fall back to inject_connector_clause() on None."""
    doc = nlp(text)
    person_ents = [e for e in doc.ents if e.label_ == "PERSON"]
    if not person_ents:
        return None

    ent = rng.choice(person_ents)
    tokens_before = tokenize(text[: ent.start_char])
    tokens_after = tokenize(text[ent.end_char :])

    # Boundary-miss guard: spaCy sometimes tags only PART of a 3+ word
    # Malaysian name (e.g. catches "Hisham Abdullah" out of "Tan Sri
    # Dr Noor Hisham Abdullah"). Replacing just that span leaves the
    # real leftover name fragment stitched next to the synthetic one
    # ("Vijay a/p Sivam Ismail Sabri Yaakob" — half real, half fake,
    # ungrammatical, and worse, leaks an untagged real name). A
    # capitalized word immediately after the span is the tell; when
    # seen, refuse this sentence rather than risk it.
    # No trailing $ anchor: tokens can now carry attached punctuation
    # ("Yaakob," / "Yaakob.") since tokenize() is whitespace-only —
    # only the leading letters need to look like a capitalized word.
    if tokens_after and re.match(r"^[A-Z][a-z]+", tokens_after[0]):
        return None

    tokens_before, honorific_tokens = _strip_trailing_honorific(tokens_before)

    injected_tokens, entity_type = ENTITY_GENERATORS["PERSON"](rng)
    full_span = honorific_tokens + injected_tokens

    labels = (
        ["O"] * len(tokens_before)
        + [f"B-{entity_type}"]
        + [f"I-{entity_type}"] * (len(full_span) - 1)
        + ["O"] * len(tokens_after)
    )
    tokens = tokens_before + full_span + tokens_after

    if len(tokens) < 3:
        return None
    return Sentence(tokens=tokens, labels=labels)


# ---------------------------------------------------------------------
# Connector-clause templates. {ENTITY} is replaced by the generated
# span's tokens during injection.
# ---------------------------------------------------------------------

CONNECTOR_TEMPLATES = {
    "en": {
        "PERSON": [
            "The matter was handled by {ENTITY} .",
            "{ENTITY} confirmed the report .",
            "According to {ENTITY} , the situation is under control .",
            "{ENTITY} was present at the meeting .",
            "The complaint was filed by {ENTITY} .",
            "Signed , {ENTITY} .",
            "This form was witnessed by {ENTITY} .",
            "The case is being reviewed by {ENTITY} .",
            "{ENTITY} , the appointed representative , will attend the hearing .",
            "Please direct all enquiries to {ENTITY} .",
        ],
        "PHONE": [
            "For more information , contact {ENTITY} .",
            "The hotline number is {ENTITY} .",
            "He can be reached at {ENTITY} .",
            "Contact No : {ENTITY} .",
            "Tel : {ENTITY} .",
            "{ENTITY} is the number to call in case of emergency .",
            "Please call {ENTITY} to confirm your appointment .",
            "Customer service can be reached at {ENTITY} during office hours .",
        ],
        "NRIC": [
            "The applicant's IC number is {ENTITY} .",
            "Identification No : {ENTITY} .",
            "IC No : {ENTITY} .",
            "{ENTITY} is the registered identification number on file .",
            "Please quote reference IC {ENTITY} in all correspondence .",
        ],
        "ADDRESS": [
            "The event was held at {ENTITY} .",
            "The office is located at {ENTITY} .",
            "Address : {ENTITY} .",
            "Correspondence should be sent to {ENTITY} .",
            "The premises are situated at {ENTITY} .",
            "{ENTITY} is where the branch office is based .",
            "Please deliver the parcel to {ENTITY} .",
            "The company is registered at {ENTITY} .",
            "Visitors should report to {ENTITY} upon arrival .",
            "{ENTITY} , the registered business address , was inspected last week .",
        ],
    },
    "ms": {
        "PERSON": [
            "Perkara itu dikendalikan oleh {ENTITY} .",
            "{ENTITY} mengesahkan laporan tersebut .",
            "Menurut {ENTITY} , keadaan kini terkawal .",
            "{ENTITY} hadir dalam mesyuarat tersebut .",
            "Aduan tersebut difailkan oleh {ENTITY} .",
            "Ditandatangani , {ENTITY} .",
            "Borang ini disaksikan oleh {ENTITY} .",
            "Kes ini sedang disemak oleh {ENTITY} .",
            "Sebarang pertanyaan boleh diajukan kepada {ENTITY} .",
        ],
        "PHONE": [
            "Untuk maklumat lanjut , hubungi {ENTITY} .",
            "Nombor talian hotline ialah {ENTITY} .",
            "Beliau boleh dihubungi di {ENTITY} .",
            "No. Tel : {ENTITY} .",
            "Sila hubungi {ENTITY} untuk mengesahkan janji temu anda .",
            "{ENTITY} ialah talian kecemasan yang boleh dihubungi .",
        ],
        "NRIC": [
            "Nombor Kad Pengenalan pemohon ialah {ENTITY} .",
            "No. KP : {ENTITY} .",
            "No. Kad Pengenalan : {ENTITY} .",
            "Sila catatkan nombor rujukan {ENTITY} dalam surat menyurat .",
        ],
        "ADDRESS": [
            "Majlis tersebut diadakan di {ENTITY} .",
            "Pejabat itu terletak di {ENTITY} .",
            "Alamat : {ENTITY} .",
            "Surat menyurat hendaklah dihantar ke {ENTITY} .",
            "Premis tersebut beralamat di {ENTITY} .",
            "Syarikat itu berdaftar di {ENTITY} .",
            "Sila hantar bungkusan tersebut ke {ENTITY} .",
            "Pelawat dikehendaki melapor diri di {ENTITY} apabila tiba .",
            "{ENTITY} ialah alamat perniagaan berdaftar yang diperiksa minggu lepas .",
        ],
    },
}


FORM_FIELD_TEMPLATES = {
    "en": {
        "PERSON": ["Name : {ENTITY}", "Full Name : {ENTITY}", "Applicant : {ENTITY}"],
        "ADDRESS": ["Address : {ENTITY}", "Home Address : {ENTITY}", "Correspondence Address : {ENTITY}"],
        "PHONE": ["Tel : {ENTITY}", "H/P : {ENTITY}", "Mobile Phone : {ENTITY}", "House Phone : {ENTITY}", "Contact No : {ENTITY}"],
        "NRIC": ["IC Number : {ENTITY}", "I/C Number : {ENTITY}", "NRIC : {ENTITY}", "No. K/P : {ENTITY}"],
    },
    "ms": {
        "PERSON": ["Nama : {ENTITY}", "Nama Penuh : {ENTITY}", "Pemohon : {ENTITY}"],
        "ADDRESS": ["Alamat : {ENTITY}", "Alamat Rumah : {ENTITY}", "Alamat Surat Menyurat : {ENTITY}"],
        "PHONE": ["Tel : {ENTITY}", "No. Telefon : {ENTITY}", "H/P : {ENTITY}"],
        "NRIC": ["No. K/P : {ENTITY}", "No. Kad Pengenalan : {ENTITY}"],
    },
}


def inject_form_fragment(
    entity_type: str, rng: random.Random, lang: str = "en"
) -> Sentence:
    """
    Resumes, forms, and letters present PII as ISOLATED FRAGMENTS, not
    as clauses embedded in flowing prose — "AHMED ZAINAH" standing
    alone at the top of a resume with zero surrounding sentence
    structure, or "Tel : 012-3456789" as a bare label-value pair. No
    amount of injecting entities into news-sentence context (what
    inject_connector_clause does) teaches a model to recognize THIS
    shape, because it never appears in that data at all. This
    generates the fragment directly, with no base sentence — found
    necessary after a real resume PDF test missed every standalone
    name and label-value field despite the pipeline working correctly
    on prose-embedded entities.

    Two sub-cases for PERSON specifically, since real resumes show
    both: a bare ALL-CAPS name with literally no label (the exact
    "AHMED ZAINAH" / "R THEVA" pattern the model missed), and a
    labelled "Name : ..." field like other entity types get.
    """
    entity_tokens, resolved_type = ENTITY_GENERATORS[entity_type](rng)

    if entity_type == "PERSON" and rng.random() < 0.4:
        tokens = [t.upper() for t in entity_tokens]
        labels = ["B-PERSON"] + ["I-PERSON"] * (len(tokens) - 1)
        return Sentence(tokens=tokens, labels=labels)

    if entity_type == "ADDRESS" and rng.random() < 0.3:
        # Bare address line with NO label at all — the exact pattern
        # missed in a real resume test: "45 Jalan Yap Kwan Seng, 47000
        # Selangor" sitting directly under a name with zero lead-in.
        labels = ["B-ADDRESS"] + ["I-ADDRESS"] * (len(entity_tokens) - 1)
        return Sentence(tokens=entity_tokens, labels=labels)

    template = rng.choice(FORM_FIELD_TEMPLATES[lang][entity_type])
    before_str, after_str = template.split("{ENTITY}")
    before_tokens = tokenize(before_str)
    after_tokens = tokenize(after_str)

    tokens = before_tokens + entity_tokens + after_tokens
    labels = (
        ["O"] * len(before_tokens)
        + [f"B-{resolved_type}"]
        + [f"I-{resolved_type}"] * (len(entity_tokens) - 1)
        + ["O"] * len(after_tokens)
    )
    return Sentence(tokens=tokens, labels=labels)


def inject_connector_clause(
    base_text: str, entity_type: str, rng: random.Random, lang: str = "en"
) -> Sentence:
    """Appends OR prepends a real base sentence (all-O context) with a
    short templated clause carrying the synthetic entity — position is
    randomized (~50/50) so entities don't always land sentence-final,
    which an adversarial eval showed the model was silently learning
    as a positional shortcut rather than the entity's actual shape.
    base_text itself is always used verbatim/untouched; only the
    clause is tagged."""
    base_tokens = tokenize(base_text)
    base_labels = ["O"] * len(base_tokens)

    template = rng.choice(CONNECTOR_TEMPLATES[lang][entity_type])
    entity_tokens, resolved_type = ENTITY_GENERATORS[entity_type](rng)

    before_str, after_str = template.split("{ENTITY}")
    before_tokens = tokenize(before_str)
    after_tokens = tokenize(after_str)

    clause_tokens = before_tokens + entity_tokens + after_tokens
    clause_labels = (
        ["O"] * len(before_tokens)
        + [f"B-{resolved_type}"]
        + [f"I-{resolved_type}"] * (len(entity_tokens) - 1)
        + ["O"] * len(after_tokens)
    )

    if rng.random() < 0.5:
        return Sentence(
            tokens=base_tokens + clause_tokens,
            labels=base_labels + clause_labels,
        )
    else:
        return Sentence(
            tokens=clause_tokens + base_tokens,
            labels=clause_labels + base_labels,
        )
