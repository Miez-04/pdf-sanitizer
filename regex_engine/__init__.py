from regex_engine.patterns import (
    NRIC_PATTERN,
    MOBILE_PATTERN,
    LANDLINE_PATTERN,
    PHONE_PATTERNS,
    is_plausible_nric,
    is_plausible_phone,
)
from regex_engine.matcher import (
    RegexMatch,
    RegexMaskRegistry,
    match_page,
    match_document,
    build_regex_mask_registry,
)

__all__ = [
    "NRIC_PATTERN",
    "MOBILE_PATTERN",
    "LANDLINE_PATTERN",
    "PHONE_PATTERNS",
    "is_plausible_nric",
    "is_plausible_phone",
    "RegexMatch",
    "RegexMaskRegistry",
    "match_page",
    "match_document",
    "build_regex_mask_registry",
]
