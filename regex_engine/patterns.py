"""
regex_engine.patterns

Deterministic Tier-1 rules. Uses the third-party `regex` module (not
stdlib `re`) so patterns can use possessive quantifiers / atomic groups
later if backtracking blowup becomes an issue on dense pages.

Every compiled pattern here is deliberately *broad* (catches format
variants); precision is enforced afterwards by the `is_plausible_*`
checks, not by making the regex itself more restrictive. This keeps
false negatives (missed PII) low, which matters more than false
positives for a redaction tool — an over-flagged token just costs a
HITL review click, a missed one is a leak.
"""

from __future__ import annotations

import calendar
import regex  # PyPI: regex, not stdlib re

# ---------------------------------------------------------------------
# MyKad NRIC: YYMMDD-PB-####  (6-2-4 digits, dash-separated)
# ---------------------------------------------------------------------

NRIC_PATTERN = regex.compile(
    r"""
    (?<![\d\-+])                # not preceded by a digit/dash/plus (avoid mid-number
                                 # matches, and phone numbers like "+60111..." whose
                                 # digits can coincidentally form a valid-looking date —
                                 # NRIC is never written with a leading "+")
    (?P<yy>\d{2})
    (?P<mm>\d{2})
    (?P<dd>\d{2})
    (?:
        \( (?P<pb1>\d{2}) \)     # parenthetical place-of-birth code: "780412(07)5319" —
        (?P<sn1>\d{4})            # a real format variant with no separator elsewhere
        |
        [-\s./]*                  # separator optional and flexible — dash, space(s),
        (?P<pb2>\d{2})             # dot, or slash (all seen in real documents), or
        [-\s./]*                  # nothing at all: "940512105119", "881204 - 14 -
        (?P<sn2>\d{4})             # 6033", "000512 14 6889", "900101/14/5678",
    )                              # "900101.14.5678".
    (?![\d-])                   # not followed by a digit/dash
    """,
    regex.VERBOSE,
)

# Official JPN place-of-birth codes are a large, imperfectly-documented
# table (Malaysian states 01-16, plus historical/foreign-birth codes up
# to 99). Hardcoding an unverified full mapping here would be worse
# than no check — a wrong "invalid" verdict silently drops a real NRIC.
# We therefore only reject codes JPN has never issued (00, and gaps
# publicly confirmed unused); everything else in 01-99 passes through
# to the dash-format + calendar-date check, which carries most of the
# precision burden. Replace with the authoritative JPN table if/when
# you have it — see README / CRITICAL_IMPROVEMENTS.
INVALID_PB_CODES = {"00", "17", "18", "19", "20"}


def is_plausible_nric(match: regex.Match) -> bool:
    """Cheap calendar-date + known-bad-code check. Not a full checksum
    (MyKad has no public checksum digit) — this is a plausibility
    filter to cut obvious false positives like invoice/tracking numbers
    that happen to be 6-2-4 digit-dash-digit."""
    yy = int(match.group("yy"))
    mm = int(match.group("mm"))
    dd = int(match.group("dd"))
    pb = match.group("pb1") or match.group("pb2")

    if pb in INVALID_PB_CODES:
        return False

    if not (1 <= mm <= 12):
        return False

    # Century is ambiguous from the IC alone; a person could be born
    # in either 19YY or 20YY. Accept if the day is valid in *either*
    # (this matters specifically for Feb 29 leap-year edge cases).
    for century in (1900, 2000):
        year = century + yy
        try:
            max_day = calendar.monthrange(year, mm)[1]
        except calendar.IllegalMonthError:
            continue
        if 1 <= dd <= max_day:
            return True
    return False


# ---------------------------------------------------------------------
# Malaysian phone numbers (mobile + landline), separator-tolerant
# ---------------------------------------------------------------------

_SEP = r"[-\s]?"

# Flexible subscriber-digit run: exactly N digits total, but the
# separator (dash or space) between any two of them is OPTIONAL and
# can fall anywhere — not just at a fixed 3-4 or 4-4 boundary. Found
# necessary against real resume data: "012-1234 432" splits the last
# 7 digits as 4+3, not the "standard" 3+4 grouping a rigid pattern
# assumes, and people format phone numbers inconsistently in the wild.
# Real validation happens in is_plausible_phone's total-digit-count
# check below, not by being clever about grouping here.
def _flex_digits(n: int) -> str:
    return r"\d(?:[-\s]?\d){" + str(n - 1) + "}"


_FLEX7 = _flex_digits(7)
_FLEX8 = _flex_digits(8)

# Trunk prefix: either the local "0", or the international "+60"/"60"
# country code (FYP spec 3.2.1 explicitly requires "+601X-XXXXXXX" as
# a supported format alongside the local "03-XXXXXXXX" form).
_LOCAL_PREFIX = r"0"
_INTL_PREFIX = r"\+?60[-\s]?"
_PREFIX = rf"(?:{_LOCAL_PREFIX}|{_INTL_PREFIX})"

# Mobile: 01[0,2,3,4,5,6,7,8,9]-XXXXXXX (7 digits) or 011-XXXXXXXX (8 digits),
# under either trunk prefix above. The whole "0XX"/"01X" prefix is
# sometimes wrapped in parentheses, e.g. "(012) 654 9870" — a format
# found in real documents this pattern didn't handle at all before.
# "5" added to the allowed second digit: 015 is a newer real MVNO
# mobile prefix, previously excluded.
MOBILE_PATTERN = regex.compile(
    rf"""
    (?<!\d)
    \(?
    {_PREFIX}
    (?:
        1(?P<mprefix1>[023456789])                 # 01X, standard mobile
        \)?
        {_SEP} {_FLEX7}
        |
        11                                          # 011, extended mobile
        \)?
        {_SEP} {_FLEX8}
    )
    (?!\d)
    """,
    regex.VERBOSE,
)

# Landline: 03-XXXXXXXX (Klang Valley, 8 digits) or 0[4-9]-XXXXXXX (7 digits),
# under either trunk prefix above. Same optional-parentheses handling
# as mobile, for area codes written like "(03) 7955 4321".
LANDLINE_PATTERN = regex.compile(
    rf"""
    (?<!\d)
    \(?
    {_PREFIX}
    (?:
        3 \)? {_SEP} {_FLEX8}                        # 03, Klang Valley
        |
        [4-9] \)? {_SEP} {_FLEX7}                     # other states
    )
    (?!\d)
    """,
    regex.VERBOSE,
)

PHONE_PATTERNS = (MOBILE_PATTERN, LANDLINE_PATTERN)


def _normalize_to_local_digits(raw_match_text: str) -> str:
    """Strip separators and collapse the +60/60 country code down to
    the equivalent local 0-prefixed digit string, so length checks
    don't need two separate ranges for local vs. international."""
    digits = regex.sub(r"\D", "", raw_match_text)
    if digits.startswith("60") and not digits.startswith("600"):
        digits = "0" + digits[2:]
    return digits


def is_plausible_phone(match: regex.Match) -> bool:
    """Digit-count sanity check after separator stripping and country-
    code normalization — guards against the mobile/landline patterns
    matching part of a longer unrelated digit run when separators are
    absent."""
    digits = _normalize_to_local_digits(match.group(0))
    # 9-10 digits: 0XX-XXXXXXX / 03-XXXXXXXX. 11 digits: 011-XXXXXXXX
    # (the one prefix with an extra digit in the subscriber number).
    return 9 <= len(digits) <= 11