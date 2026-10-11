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

1. Replacement (English only): a rule-based detector (find_person_spans:
   title-led, patronym-led and registry-anchored patterns, no trained
   model of any kind) finds a PERSON mention already in the sentence; we
   swap it for a synthetic name, keeping
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

from data.registries import (
    ENGLISH_GIVEN_NAMES,
    ENTITY_GENERATORS,
    EAST_MALAYSIAN_FIRST,
    INDIAN_FIRST,
    MALAY_FEMALE_FIRST,
    MALAY_MALE_FIRST,
    MALAY_PATRONYM,
)
from models.dataset import Sentence

_TOKEN_RE = re.compile(r"\w+(?:['/]\w+)*|[^\w\s]")

HONORIFIC_TOKENS_LOWER = {
    "encik", "tuan", "datuk", "dato'", "dato", "tan", "sri", "dr", "dr.", "haji",
    "ir", "cik", "puan", "datin", "prof", "hajjah", "tun",
}
# multi-word honorifics stored as tuples for suffix matching against
# the tail of tokens_before
MULTI_WORD_HONORIFICS = [
    ["datuk", "seri"], ["datuk", "sri"], ["dato'", "seri"], ["dato'", "sri"],
    ["datin", "seri"],
    ["tan", "sri"], ["puan", "sri"], ["dato'"], ["dato"], ["datuk"],
    ["tuan"], ["encik"], ["cik"], ["puan"], ["datin"], ["dr."], ["dr"], ["prof"],
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


# ---------------------------------------------------------------------
# Rule-based PERSON span detector (replaces the earlier spaCy NER call).
# No trained model is involved: everything below is string patterns plus
# this project's own name registries (data/registries.py).
#
# Design rule: PRECISION over recall. A missed name only means that
# sentence falls back to connector-clause injection (a different, still
# valid training sentence); a wrong span would put a wrong gold label
# into the training data.
# ---------------------------------------------------------------------

_TRAILING_PUNCT = ",.;:!?)\"'\u201d\u2019"
_NAME_CONNECTORS = {"bin", "binti", "bte", "bt", "a/l", "a/p", "d/o", "s/o", "anak", "ak", "@"}
_PATRONYM_CONNECTORS = {"bin", "binti", "bte", "a/l", "a/p", "anak"}

# Title sequences that introduce a name (detection anchors only). Which of
# them are pulled INTO the PERSON span is decided separately by
# MULTI_WORD_HONORIFICS (existing project convention: Malay honorifics are
# part of the span; English Mr/Mrs/Ms stay outside it).
_TITLE_SEQUENCES = sorted(
    [
        ["tan", "sri"], ["puan", "sri"], ["datuk", "seri"], ["datuk", "sri"],
        ["dato'", "seri"], ["dato'", "sri"], ["datin", "seri"],
        ["datuk"], ["dato'"], ["dato"], ["datin"], ["encik"], ["tuan"],
        ["cik"], ["puan"], ["dr."], ["dr"], ["prof"], ["haji"], ["hajjah"], ["tun"],
        ["mr"], ["mrs"], ["ms"], ["mdm"], ["madam"], ["miss"],
    ],
    key=len,
    reverse=True,  # longest first, so "tan sri" wins over a bare "tan"
)

# Words that mark an organisation / place / product, not a person. A span
# containing one of these is dropped (title-led spans are cut before it).
_NON_PERSON_WORDS = {
    "sdn", "bhd", "berhad", "bank", "holdings", "group", "corporation", "corp",
    "ltd", "inc", "university", "universiti", "hospital", "hotel", "road",
    "street", "jalan", "taman", "kampung", "airlines", "industries", "energy",
    "resources", "capital", "properties", "trading", "technology",
    "technologies", "international", "plantation", "index", "market",
    "ministry", "department", "committee", "council", "party", "club",
    # places / venues / landmarks (found by reviewing detector output on
    # the English news corpus: "Ibrahim Stadium", "Puteri Gunung Ledang")
    "stadium", "gunung", "bukit", "pulau", "tasik", "sungai", "park",
    "airport", "station", "centre", "center", "school", "college", "mosque",
    "masjid", "temple", "complex", "tower", "bridge", "island", "lake",
    "river", "mountain", "hill", "beach", "bay", "fort",
}

_KNOWN_GIVEN_LOWER = {
    name.lower()
    for pool in (
        MALAY_MALE_FIRST, MALAY_FEMALE_FIRST, MALAY_PATRONYM,
        ENGLISH_GIVEN_NAMES, INDIAN_FIRST, EAST_MALAYSIAN_FIRST,
    )
    for entry in pool
    for name in entry.split()
}

_NAME_WORD_RE = re.compile(r"^[A-Z][A-Za-z'\u2019\-]*$")
_INITIAL_RE = re.compile(r"^[A-Z]\.$")


def _core(token: str) -> str:
    """'Yaakob,' -> 'Yaakob'. Input: str. Output: str without trailing punctuation."""
    return token.rstrip(_TRAILING_PUNCT)


def _is_name_word(token: str) -> bool:
    """True for a capitalised alphabetic word that can be part of a name
    ('Ismail', 'Yaakob,', \"O'Brien\", initial 'A.'). False for possessives
    ('Malaysia's'), lowercase words, and organisation/place marker words."""
    if _INITIAL_RE.match(token):
        return True
    core = _core(token)
    if not core or core.lower().endswith(("'s", "\u2019s")):
        return False
    if len(core) > 1 and core.isupper():
        return False  # news datelines / acronyms ("KOTA", "KLCI"), not name words
    if core.lower() in _NON_PERSON_WORDS:
        return False
    return bool(_NAME_WORD_RE.match(core))


def _is_connector(token: str) -> bool:
    return _core(token).lower() in _NAME_CONNECTORS or token == "@"


def _title_len_at(tokens: list[str], i: int) -> int:
    """Number of tokens of the title sequence starting at index i, or 0.
    'Tan Sri' -> 2, bare 'Tan' (a surname) -> 0."""
    lowered = [_core(t).lower() for t in tokens[i : i + 2]]
    for seq in _TITLE_SEQUENCES:
        if lowered[: len(seq)] == seq:
            return len(seq)
    return 0


def _ends_clause(token: str) -> bool:
    """True if the token ends with punctuation that stops a name
    (comma, full stop...), except an initial like 'A.'."""
    return token != _core(token) and not _INITIAL_RE.match(token)


def _collect_name(tokens: list[str], start: int, max_words: int = 6) -> int:
    """Starting at index `start`, return the exclusive end index of a run
    of name words / connectors (at most max_words). Returns `start` when
    no name begins there. Never ends on a connector."""
    n = len(tokens)
    if start >= n or not _is_name_word(tokens[start]):
        return start
    k = start
    while k < n and k - start < max_words:
        tok = tokens[k]
        if not (_is_name_word(tok) or _is_connector(tok)):
            break
        k += 1
        if _ends_clause(tok):
            break
    while k > start and _is_connector(tokens[k - 1]):
        k -= 1
    # A run made only of initials ("M.") is not a name.
    if all(_INITIAL_RE.match(t) for t in tokens[start:k]):
        return start
    return k


def find_person_spans(tokens: list[str]) -> list[tuple[int, int]]:
    """Input: list of whitespace tokens. Output: sorted list of
    (start, end) index pairs (end exclusive) of PERSON name spans, titles
    NOT included. Three passes, each only on tokens no earlier pass used:
      1. title-led:      'Datuk Seri Anwar Ibrahim', 'Dr Noor Hisham'
      2. patronym-led:   'Ahmad bin Ali', 'Kumar a/l Raju'
      3. registry-anchored: first word is a known given name from
         data/registries.py and at least one more name word follows."""
    n = len(tokens)
    claimed = [False] * n
    spans: list[tuple[int, int]] = []

    def claim(a: int, b: int) -> None:
        spans.append((a, b))
        for k in range(a, b):
            claimed[k] = True

    # Pass 1: title-led
    i = 0
    while i < n:
        t_len = _title_len_at(tokens, i)
        if not t_len:
            i += 1
            continue
        j = i + t_len
        while j < n and _title_len_at(tokens, j):  # chained: 'Datuk Seri Dr ...'
            j += _title_len_at(tokens, j)
        end = _collect_name(tokens, j)
        if end > j:
            claim(j, end)
            i = end
        else:
            i = max(j, i + 1)

    # Pass 2: patronym-led (given name(s) + bin/binti/a-l/a-p + father's name)
    for i in range(1, n - 1):
        if claimed[i] or _core(tokens[i]).lower() not in _PATRONYM_CONNECTORS:
            continue
        left = i
        while (
            left > 0
            and i - left < 3
            and not claimed[left - 1]
            and _is_name_word(tokens[left - 1])
            and not _ends_clause(tokens[left - 1])
        ):
            left -= 1
        right = _collect_name(tokens, i + 1)
        if left < i and right > i + 1 and not any(claimed[left:right]):
            claim(left, right)

    # Pass 3: registry-anchored
    i = 0
    while i < n:
        if (
            claimed[i]
            or not _is_name_word(tokens[i])
            or _core(tokens[i]).lower() not in _KNOWN_GIVEN_LOWER
            or (i > 0 and _core(tokens[i - 1]).lower() in _NON_PERSON_WORDS)
        ):
            i += 1
            continue
        end = _collect_name(tokens, i, max_words=4)
        if end - i >= 2 and not any(claimed[i:end]):
            claim(i, end)
            i = end
        else:
            i += 1

    return sorted(spans)


def inject_by_replacement(text: str, rng: random.Random) -> Sentence | None:
    """English-only: replace a rule-detected PERSON span (find_person_spans)
    with a synthetic name. Returns None if no PERSON span was found, OR if
    the boundary looks unsafe to replace (see below) - callers should fall
    back to inject_connector_clause() on None."""
    tokens = tokenize(text)
    spans = find_person_spans(tokens)
    if not spans:
        return None

    start, end = rng.choice(spans)
    # Punctuation attached to the last name token ("Yaakob,") must survive
    # as its own token after the injected name.
    last = tokens[end - 1]
    leftover = last[len(_core(last)) :]
    tokens_before = tokens[:start]
    tokens_after = ([leftover] if leftover else []) + tokens[end:]

    # Boundary guard: a capitalised word immediately after the span (no
    # punctuation between) means the detector stopped inside a longer name
    # or at an organisation word. Replacing would leave a real name
    # fragment stitched to the synthetic one (half real, half fake, and an
    # untagged real name), so refuse the sentence instead.
    if not leftover and tokens_after and re.match(r"^[A-Z][a-z]+", tokens_after[0]):
        return None

    # Pull ALL directly preceding honorific groups into the span
    # ("Datuk Seri Dr ..."), matching the PERSON-includes-honorific rule.
    honorific_tokens: list[str] = []
    while True:
        tokens_before, found = _strip_trailing_honorific(tokens_before)
        if not found:
            break
        honorific_tokens = found + honorific_tokens

    injected_tokens, entity_type = ENTITY_GENERATORS["PERSON"](rng)
    full_span = honorific_tokens + injected_tokens

    labels = (
        ["O"] * len(tokens_before)
        + [f"B-{entity_type}"]
        + [f"I-{entity_type}"] * (len(full_span) - 1)
        + ["O"] * len(tokens_after)
    )
    new_tokens = tokens_before + full_span + tokens_after

    if len(new_tokens) < 3:
        return None
    return Sentence(tokens=new_tokens, labels=labels)


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
