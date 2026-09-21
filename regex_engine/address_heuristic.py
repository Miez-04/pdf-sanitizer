"""
regex_engine.address_heuristic

Tier 1.5: a deterministic address detector, added after extensive data
diversity work (document-type templates, real address pools, bare-
format training) still left ADDRESS as the weakest-performing entity
across real test PDFs — confirmed repeatedly across resumes, letters,
and forms. NRIC and PHONE have reliable deterministic formats regex
can exploit; ADDRESS has no single fixed format, but Malaysian
addresses DO have one near-universal anchor: a 5-digit postcode,
almost always adjacent to a street/area-type keyword (Jalan, Taman,
Lot, No., etc.).

This finds postcode + nearby-keyword combinations and expands outward
to capture the surrounding address block — the same "Tier 1 wins"
priority as NRIC/PHONE regex, but specific to ADDRESS, and
deliberately conservative: it only claims a span when BOTH a
plausible postcode AND an anchor keyword are present nearby, to avoid
false-firing on an unrelated 5-digit number or an unrelated mention of
a street name with no postcode attached.
"""

from __future__ import annotations

import regex

from regex_engine.patterns import MOBILE_PATTERN, LANDLINE_PATTERN

# Malaysian postcodes run 01000-98999 in practice (verified range
# against the real postalcode.json registry used elsewhere in this
# project — see data/registries.py's load_postcode_registry).
_POSTCODE_RE = regex.compile(r"(?<!\d)(\d{5})(?!\d)")

ADDRESS_ANCHOR_KEYWORDS = {
    "jalan", "jln", "lorong", "lrg", "persiaran", "lebuh", "lebuhraya",
    "lengkok", "taman", "tmn", "kampung", "kg", "bandar", "seksyen",
    "sek", "no.", "unit", "tingkat", "tkt", "blok", "pt", "wisma",
    "menara", "kompleks", "bangunan", "beralamat",
    # Added after real PDF testing: apartment/condo building types are
    # extremely common address anchors in Malaysian resumes/forms and
    # were entirely missing ("1-2-2- Pangsapuri Idam" went undetected
    # because "pangsapuri" wasn't recognized as an anchor at all).
    "pangsapuri", "apartment", "kondominium", "condominium", "flat",
    "pt.", "kompleks", "villa", "residensi", "residence", "residen",
    # Added from a Malaysian formatting-reference catalogue: "lor."/
    # "lor" (abbreviated lorong), "pers."/"persrn" (abbreviated
    # persiaran), "kampong" (alternate spelling of kampung), and
    # "peti surat" (PO Box) — none previously covered.
    "lor.", "lor", "pers.", "persrn", "kampong", "peti", "kavling",
    # Deliberately EXCLUDED despite appearing in real addresses: "no"
    # (without a period), "off", "lot", "block", "batu", "alamat",
    # "address" — all common enough as ordinary words that pairing
    # them with an unrelated nearby 5-digit number produces false
    # positives on plain prose (confirmed: "There is no reason ...
    # reached 45231 residents" was incorrectly claimed as an address
    # anchored on bare "no"). "No." WITH the period is kept since real
    # addresses consistently write it that way and it's far less
    # ambiguous as a token.
}

# Common state/FT names — extending the span forward past the postcode
# to include these, since real addresses end with "<postcode> <city>,
# <state>", not just the postcode alone.
_STATE_TOKENS = {
    "selangor", "johor", "kedah", "kelantan", "melaka", "malacca",
    "negeri", "sembilan", "pahang", "perak", "perlis", "pulau", "pinang",
    "penang", "sabah", "sarawak", "terengganu", "kuala", "lumpur",
    "putrajaya", "labuan", "wp", "malaysia",
}

_WINDOW = 12  # how many tokens back to search for an anchor keyword
_EXTRA_ANCHOR_WINDOW = 3  # extra tokens to look for a SECOND, earlier anchor
                          # (catches "Pangsapuri Idam Jalan X, postcode" where
                          # "pangsapuri" is the more complete anchor but
                          # "jalan" is nearer to the postcode). Kept SMALL
                          # deliberately — a real bug was found where a
                          # window of 5 crossed over an unrelated phone-
                          # number field sitting between the building name
                          # and the street name, swallowing it into the
                          # address span. 3 still catches the real "Idam" +
                          # "Pangsapuri" case (2 tokens back) with a margin,
                          # without reaching distant unrelated content.

# Matches unit/lot codes common in Malaysian addresses: "G-2", "1-2-2-",
# "A-3-5", "12A" — anything alphanumeric with at least one digit,
# optionally hyphen-segmented. Requires a digit (via lookahead) so it
# doesn't accidentally match plain words like "Selangor".
_UNIT_CODE_RE = regex.compile(r"^(?=.*\d)[A-Za-z0-9]+(-[A-Za-z0-9]*)*$")


def _looks_like_phone_number(text: str) -> bool:
    """A unit/lot code candidate that ALSO independently matches a
    phone pattern (e.g. "010-8765432") must not be treated as a house
    number — found via a real 2-column resume layout where a phone
    value sitting immediately before "Jalan ..." got swallowed into
    the address span otherwise."""
    return bool(MOBILE_PATTERN.search(text) or LANDLINE_PATTERN.search(text))


def _is_plausible_postcode(text: str) -> bool:
    if not (text.isdigit() and len(text) == 5):
        return False
    code = int(text)
    return 1000 <= code <= 98999


def find_address_spans(tokens: list[str]) -> list[tuple[int, int]]:
    """
    tokens: whitespace-tokenized text (same granularity as
    pdf_ingestion — see data.entity_mutation.tokenize's docstring for
    why whitespace-only matters here).

    Returns a list of (start_index, end_index) token-index spans
    (end_index exclusive), each anchored on one postcode + its nearest
    preceding address keyword, extended:
      - backward through a unit/lot code immediately before the anchor
        ("G-2, Jalan ...", "1-2-2- Pangsapuri Idam Jalan ...")
      - backward further to an EARLIER anchor keyword within a small
        extra window, so a building-type word further back than the
        nearest street keyword still gets captured
      - forward through any trailing state/city tokens after the
        postcode
    """
    spans: list[tuple[int, int]] = []
    claimed: set[int] = set()

    for i, tok in enumerate(tokens):
        cleaned = tok.strip(",.;:()")
        if not _is_plausible_postcode(cleaned):
            continue
        if i in claimed:
            continue

        # Search backward for the nearest anchor keyword.
        start = None
        for j in range(i - 1, max(-1, i - 1 - _WINDOW), -1):
            word = tokens[j].strip(",.;:()").lower()
            if word in ADDRESS_ANCHOR_KEYWORDS:
                start = j
                break
        if start is None:
            continue  # no nearby keyword — too risky to claim, skip

        # Look a bit further back for an EARLIER anchor keyword — e.g.
        # "pangsapuri" sitting before "jalan" — since a building-type
        # word further from the postcode is often the more complete
        # start of the address, not the nearer street keyword alone.
        # Plain descriptive words in between (e.g. "Idam" in
        # "Pangsapuri Idam Jalan ...") don't break the scan — only the
        # fixed window size bounds how far back this looks.
        for k in range(start - 1, max(-1, start - 1 - _EXTRA_ANCHOR_WINDOW), -1):
            word = tokens[k].strip(",.;:()").lower()
            if word in ADDRESS_ANCHOR_KEYWORDS:
                start = k
                break

        # A bare house/lot/unit number often sits immediately BEFORE
        # the (possibly now-extended) start — "45 Jalan ...", "G-2,
        # Jalan ...", "1-2-2- Pangsapuri ...". Include it if present.
        if start > 0:
            prev = tokens[start - 1].strip(",.;:()")
            if _UNIT_CODE_RE.match(prev) and not _looks_like_phone_number(prev):
                start -= 1

        # Extend forward past the postcode through a trailing city/
        # state tail (e.g. "50480 Kuala Lumpur", "68000 Ampang,
        # Selangor" — "Ampang" is a city, not in _STATE_TOKENS, so
        # extension can't require every token to be a known state; it
        # also accepts capitalized words for a bounded number of extra
        # tokens, since real addresses end "<postcode> <City>, <State>"
        # and the city name varies too widely to enumerate).
        end = i + 1
        extra_capitalized_used = 0
        found_state = False
        _MAX_EXTRA_CAPITALIZED = 4
        while end < len(tokens):
            raw = tokens[end]
            word = raw.strip(",.;:()").lower()
            if word in _STATE_TOKENS:
                end += 1
                found_state = True
            elif (
                not found_state
                and extra_capitalized_used < _MAX_EXTRA_CAPITALIZED
                and raw[:1].isupper()
                and not raw.strip(",.;:()").isdigit()
            ):
                # Loose capitalized-word extension (for an unlisted
                # city name) ONLY before a real state token is found —
                # states are terminal in "<postcode> <City>, <State>",
                # so once one is matched, stop accepting arbitrary
                # capitalized words (otherwise this swallows whatever
                # capitalized word happens to follow the address, e.g.
                # the next field's label).
                end += 1
                extra_capitalized_used += 1
            else:
                break

        if any(k in claimed for k in range(start, end)):
            continue  # overlaps a previously-claimed span

        spans.append((start, end))
        claimed.update(range(start, end))

    return spans
