"""
data/real_samples.py

Converts data/raw/samples.pdf — 40 real Malaysian documents (10 each
of resumes, letters, reports, invoices) provided as real-world
reference data — into gold IOB2 training examples.

Strategy: NRIC/PHONE/ADDRESS are already reliably auto-detectable by
this project's own regex_engine (dash-optional NRIC, flexible-
separator phone, postcode-anchored address heuristic — all verified
against real PDFs earlier in this project). Re-deriving them by hand
here would just duplicate that work and risk transcription errors.
PERSON is the one entity type nothing in regex_engine can find, so
that's curated by hand below — the highest-value, hardest-to-automate
part of this dataset.

This is real, naturally-occurring document text — not injected into
template carriers — so PII spans here reflect the true range of
formats these documents actually use, rather than this project's own
(sometimes incomplete) model of what that range looks like.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from data.entity_mutation import tokenize
from models.dataset import Sentence
from regex_engine.address_heuristic import find_address_spans
from regex_engine.matcher import match_page
from pdf_ingestion.schema import BBox, PageTokens, Token

# Known false positives from automated regex/heuristic detection on
# THIS specific real text — a handful of digit-heavy decoy values
# (account numbers, reference codes) coincidentally match the phone/
# address patterns' shape. Manually excluded here so the gold labels
# used for training don't teach the model a wrong pattern. Each entry
# is the exact (page_num, token_text) to force back to "O".
FALSE_POSITIVE_OVERRIDES: set[tuple[int, str]] = {
    (32, "09812-3401-998"),  # account number, not a phone number
}
# Exact substrings as they appear in the extracted text — including
# honorifics/parenthetical Chinese names where present, since those
# are part of the real PERSON mention as written.
PERSON_SPANS: dict[int, list[str]] = {
    0: ["JASON TAN WEI JIE"],
    1: ["SITI NURHALIZA BINTI MUHAMMAD SHAHRIZAL"],
    2: ["PRIYA A/P CHANDRAN"],
    3: ["AHMAD SYAKIR ISMAIL"],
    4: ["LIM KOK LEONG (林国良)"],
    5: ["TAN SOCK LING (CHERYL)", "Dr. Faridah Binti Ahmad"],
    6: ["MUHAMMAD SYAHMI BIN ABDUL RAHMAN", "Encik Hashim Bin Ismail"],
    7: ["KUMARAN A/L GANESAN"],
    8: ["STEPHANIE LINDA WONG (黄丽达)"],
    9: ["AHMAD SHAZWAN"],
    10: ["Ahmad Zaki Bin Hashim", "AHMAD ZAKI BIN HASHIM"],
    11: ["TAN SOCK HUA (JENNIFER)", "TAN SOCK HUA"],
    12: ["SUBRAMANIAM A/L RAMASAMY", "KAVITHA A/P SUBRAMANIAM"],
    13: ["David Wong Chun Wai"],
    14: ["Nurul Aisyah Binti Zulkifli", "NURUL AISYAH BINTI ZULKIFLI"],
    15: ["Marilyn Anak Jinggut", "MARILYN ANAK JINGGUT"],
    16: ["Mohd Hafiz Bin Awang Damit", "MOHD HAFIZ BIN AWANG DAMIT"],
    17: ["Lim Chee Keong", "LIM CHEE KEONG", "Siti Fatimah Binti Abdullah", "SITI FATIMAH BINTI ABDULLAH"],
    18: ["RAJNIKANTHAN A/L VELU"],
    19: ["Farah Syazwani Mohd Nor", "FARAH SYAZWANI MOHD NOR"],
    20: ["Ir. Tan Boon Hock"],
    21: ["AHMAD FAISAL BIN RAZALI", "Mohd Azlan Bin Mansor"],
    22: ["NURUL HUDA BINTI OTHMAN"],
    23: ["SURESH A/L KALIAPPAN"],
    24: ["GRACE ANAK JALONG"],
    25: ["LIM CHEE SENG", "PUAN ZAITON BINTI HASHIM", "Lim Chee Seng", "Puan Zaiton Binti Hashim"],
    26: ["RAJESWARI A/P MUNUSAMY", "Tan Chin Seng"],
    27: ["MOHD RIDZUAN AWANG", "Dr. Nurul Hanim Binti Ahmad"],
    28: ["WONG MEI LING (KAREN WONG)", "Karen Wong Mei Ling"],
    29: ["DR. HAFIZAH BINTI MOHD YUSOF"],
    30: ["NURUL HAZIQAH BINTI AZMAN"],
    31: ["NURUL HAZIQAH BINTI AZMAN"],
    32: ["KUMAR A/L VENGADASALAM"],
    33: ["LEONG KA WAI (ALVIN)"],
    34: ["SARAVANAN A/L BALAKRISHNAN"],
    35: ["PATRICIA ANAK NYLANG"],
    36: ["MOHD SHAHRIZAL AWANG"],
    37: ["CHONG MEI FANG"],
    38: ["MUHAMMAD AMIRUL BIN HASHIM"],
    39: ["SITI FATIMAH BINTI ZULKIFLI"],
}


def _page_text_to_fake_page(text: str) -> PageTokens:
    """Builds a PageTokens from real page text using the SAME
    whitespace tokenizer as the rest of the pipeline, so regex_engine
    and the address heuristic — both written against that granularity
    — behave identically to how they would on real PDF-extracted text."""
    tokens = tokenize(text)
    page = PageTokens(page_num=0, page_width=1000.0, page_height=2000.0)
    x_cursor = 0.0
    char_cursor = 0
    text_parts = []
    for i, tok in enumerate(tokens):
        width = max(len(tok), 1) * 6.0
        page.tokens.append(
            Token(
                text=tok,
                bbox=BBox(x_cursor, 0.0, x_cursor + width, 12.0),
                page_num=0,
                block_no=0,
                line_no=0,
                word_no=i,
                token_index=i,
                char_start=char_cursor,
                char_end=char_cursor + len(tok),
            )
        )
        text_parts.append(tok)
        x_cursor += width + 6.0
        char_cursor += len(tok) + 1
    page.page_text = " ".join(text_parts)
    return page


def _mark_person_spans(page: PageTokens, person_texts: list[str]) -> None:
    """Finds EVERY occurrence of each curated PERSON substring in the
    token stream and tags all of them — substrings are matched against
    the SAME tokenize() output to guarantee alignment, not against raw
    characters. Matching strips leading/trailing punctuation from each
    token before comparing (but tags the ORIGINAL token, punctuation
    included) — needed for signature blocks like "(AHMAD ZAKI BIN
    HASHIM)" where whitespace-tokenization glues the parentheses onto
    the first/last name token.

    Does NOT stop after the first match: these real documents very
    commonly repeat a name 2-3 times (header, body mention, signature
    block) — an earlier version of this function broke after the
    first hit and silently left every repeat untagged."""
    strip_chars = "(),.;:\"'"
    tokens_stripped = [t.text.strip(strip_chars) for t in page.tokens]
    for person_text in person_texts:
        person_tokens = [t.strip(strip_chars) for t in tokenize(person_text)]
        n = len(person_tokens)
        for i in range(len(tokens_stripped) - n + 1):
            if tokens_stripped[i : i + n] == person_tokens:
                if all(page.tokens[i + j].label == "O" for j in range(n)):
                    for j in range(n):
                        page.tokens[i + j].label = "B-PERSON" if j == 0 else "I-PERSON"
                        page.tokens[i + j].source = "gold"


def build_real_sample_sentences(
    pdf_pages_json: str | Path, person_spans: dict[int, list[str]] | None = None
) -> list[Sentence]:
    """Returns one Sentence per page, each with real document text and
    gold PERSON/ADDRESS/NRIC/PHONE labels. person_spans defaults to
    the module-level PERSON_SPANS (SAMPLES.pdf's 40 pages) but accepts
    an override so this same function can process a second/third real
    document batch without duplicating the whole module."""
    if person_spans is None:
        person_spans = PERSON_SPANS

    with open(pdf_pages_json, encoding="utf-8") as f:
        pages_text = json.load(f)

    sentences = []
    for page_num, text in enumerate(pages_text):
        page = _page_text_to_fake_page(text)

        # NRIC/PHONE via the project's own regex — already verified
        # reliable against real PDFs (dash-optional NRIC, flexible
        # phone separators).
        match_page(page)

        # ADDRESS via the postcode-anchored heuristic, filling only
        # tokens NRIC/PHONE regex didn't already claim (same
        # gap-filler contract as pipeline.conflict_resolution).
        token_texts = [t.text for t in page.tokens]
        for start_idx, end_idx in find_address_spans(token_texts):
            span = page.tokens[start_idx:end_idx]
            if any(t.label != "O" for t in span):
                continue
            for i, tok in enumerate(span):
                tok.label = "B-ADDRESS" if i == 0 else "I-ADDRESS"
                tok.source = "gold"

        # PERSON via hand-curated spans (see PERSON_SPANS above).
        _mark_person_spans(page, person_spans.get(page_num, []))

        # Manual false-positive corrections (see FALSE_POSITIVE_OVERRIDES).
        for tok in page.tokens:
            if (page_num, tok.text) in FALSE_POSITIVE_OVERRIDES:
                tok.label = "O"
                tok.source = ""

        sentences.append(
            Sentence(
                tokens=[t.text for t in page.tokens],
                labels=[t.label for t in page.tokens],
            )
        )
    return sentences


def write_iob2(sentences: list[Sentence], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for tok, lab in zip(sent.tokens, sent.labels):
                f.write(f"{tok} {lab}\n")
            f.write("\n")


if __name__ == "__main__":
    import sys

    pages_json = sys.argv[1] if len(sys.argv) > 1 else "/tmp/samples_pages.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/real_samples.txt"

    sentences = build_real_sample_sentences(pages_json)
    write_iob2(sentences, out_path)

    from collections import Counter

    counts = Counter()
    for s in sentences:
        for lab in s.labels:
            if lab.startswith("B-"):
                counts[lab[2:]] += 1
    print(f"pages processed: {len(sentences)}")
    print(f"entity counts: {dict(counts)}")
    print(f"saved to {out_path}")
