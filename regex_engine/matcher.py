"""
regex_engine.matcher

Runs Tier-1 patterns against each PageTokens.page_text and maps the
resulting character spans back onto the Token list produced by
pdf_ingestion. This is the module responsible for the token-splitting
edge case flagged earlier: a phone number written with spaces
("012 345 6789") lands as 3 separate word-tokens from PyMuPDF, so a
single regex match can and does cover multiple Token objects. NRIC
rarely splits (PyMuPDF only breaks on whitespace, and NRIC has no
internal spaces) but the mapping code makes no assumption either way —
it always resolves a match span to *however many* tokens it overlaps.

Tokens are mutated in place (label/source/confidence) rather than
returning a parallel structure, since pipeline's conflict-resolution
gate and the HITL UI both need to read labels directly off the
Token objects produced by ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass

import regex

from pdf_ingestion.schema import DocumentTokens, PageTokens, Token
from regex_engine.patterns import (
    NRIC_PATTERN,
    PHONE_PATTERNS,
    is_plausible_nric,
    is_plausible_phone,
)

REGEX_SOURCE = "regex"


@dataclass(slots=True)
class RegexMatch:
    label: str              # "NRIC" | "PHONE"
    text: str
    char_start: int
    char_end: int
    page_num: int
    token_indices: tuple[int, ...]  # indices into PageTokens.tokens, in order


class RegexMaskRegistry:
    """
    Direct implementation of the `regex_mask_registry` named in the FYP
    pseudocode (Section 3.3.2, Step 1-3): the set of token indices Tier 1
    has claimed, keyed by (page_num, token_index) so pipeline's conflict
    resolution gate can do the exact membership check the pseudocode
    describes — `IF ti IN regex_mask_registry: hard-mask override` —
    without re-deriving it from the Token objects.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[int, int], RegexMatch] = {}

    def register(self, match: RegexMatch) -> None:
        for token_index in match.token_indices:
            self._entries[(match.page_num, token_index)] = match

    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self._entries

    def get_match(self, page_num: int, token_index: int) -> RegexMatch | None:
        return self._entries.get((page_num, token_index))

    def __len__(self) -> int:
        return len(self._entries)


def _map_span_to_tokens(
    page: PageTokens, char_start: int, char_end: int
) -> tuple[int, ...]:
    """Return token_index values for every token whose char range
    overlaps [char_start, char_end). O(n) scan per match; pages are
    small enough (hundreds of tokens) that this doesn't need an
    interval tree, but if profiling later says otherwise, bisect on
    a sorted char_start list is the next step."""
    return tuple(
        t.token_index
        for t in page.tokens
        if t.char_start < char_end and t.char_end > char_start
    )


def _apply_iob2(page: PageTokens, token_indices: tuple[int, ...], entity: str) -> None:
    for pos, idx in enumerate(token_indices):
        token = page.tokens[idx]
        token.label = f"B-{entity}" if pos == 0 else f"I-{entity}"
        token.source = REGEX_SOURCE
        token.confidence = 1.0


def match_page(page: PageTokens) -> list[RegexMatch]:
    """Run all Tier-1 patterns against one page, mutate its tokens'
    labels in place, and return the match records (used by eval/ and
    the HITL UI to show *why* a box was flagged)."""
    matches: list[RegexMatch] = []
    claimed: list[tuple[int, int]] = []  # char spans already matched, to skip overlap

    def _overlaps_claimed(start: int, end: int) -> bool:
        return any(start < c_end and end > c_start for c_start, c_end in claimed)

    # NRIC first — most specific pattern, and MyKad strings could
    # otherwise be partially re-matched by the looser phone patterns.
    for m in NRIC_PATTERN.finditer(page.page_text):
        if not is_plausible_nric(m):
            continue
        start, end = m.start(), m.end()
        if _overlaps_claimed(start, end):
            continue
        token_indices = _map_span_to_tokens(page, start, end)
        if not token_indices:
            continue
        _apply_iob2(page, token_indices, "NRIC")
        matches.append(
            RegexMatch("NRIC", m.group(0), start, end, page.page_num, token_indices)
        )
        claimed.append((start, end))

    for pattern in PHONE_PATTERNS:
        for m in pattern.finditer(page.page_text):
            if not is_plausible_phone(m):
                continue
            start, end = m.start(), m.end()
            if _overlaps_claimed(start, end):
                continue
            token_indices = _map_span_to_tokens(page, start, end)
            if not token_indices:
                continue
            _apply_iob2(page, token_indices, "PHONE")
            matches.append(
                RegexMatch(
                    "PHONE", m.group(0), start, end, page.page_num, token_indices
                )
            )
            claimed.append((start, end))

    return matches


def match_document(document: DocumentTokens) -> list[RegexMatch]:
    all_matches: list[RegexMatch] = []
    for page in document.pages:
        all_matches.extend(match_page(page))
    return all_matches


def build_regex_mask_registry(document: DocumentTokens) -> RegexMaskRegistry:
    """Entry point matching the FYP pseudocode's INITIALIZE/Tier-1 steps:
    runs Tier 1 across the whole document and returns the populated
    registry pipeline's conflict-resolution gate reads from."""
    registry = RegexMaskRegistry()
    for match in match_document(document):
        registry.register(match)
    return registry
