"""
data/build_v21_corpus.py

Targets a NEW regression confirmed from real-PDF testing (Oct 2026): a
market-comparison financial table — percentages, RM currency amounts,
company names, a "TOTAL" row — got mistagged ADDRESS. Checked
regex_engine/address_heuristic.py directly: it REQUIRES a nearby
anchor keyword (jalan, taman, no., etc.) before claiming anything, and
none of these table cells have one — so this is confirmed to be the
MODEL (Tier 2), not the Tier 1.5 heuristic.

Root cause: the ADDRESS training pool (ADDRESS_VALUE_POOL in
build_v19_corpus.py) is inherently comma-and-digit-heavy ("No. 12,
Jalan SS 2/45, 47300 Petaling Jaya, Selangor"), and there has never
been a negative example showing OTHER comma/digit-heavy content that
ISN'T an address. The model appears to have learned "digits with
commas" as a shortcut signal instead of the real anchor a Malaysian
address actually has: a postcode + state. Fixed the same way as
course-title-as-ADDRESS and section-header-as-PERSON before it —
explicit negative examples of the confusable shape.

Combines with v20. Torch-free, same reasoning as v18/19/20.

Usage:
    python -m data.build_v21_corpus
"""

from __future__ import annotations

import argparse
import random

from data.build_v18_corpus import (
    Sentence,
    parse_iob2_file,
    write_iob2,
    skeleton,
    split_by_skeleton,
    normalize_address_lead_words,
    entity_token_counts,
)


# =======================================================================
# Financial/numeric-table negatives — the exact confusable shape:
# percentages, RM currency amounts (comma-grouped), company names,
# table header phrases, and a "TOTAL" row — all genuinely O.
# =======================================================================

COMPANY_NAMES = [
    "Shimano Inc.", "Daiwa Sports Sdn. Bhd.", "Penn Fishing", "Reelworks Co.",
    "Northport Logistics Services", "Maju Holdings Berhad",
    "Golden Screen Cinemas Sdn. Bhd.", "Sunway Construction Group",
    "Bintang Capital Partners", "Perusahaan Otomobil Nasional Berhad",
]

PERCENTAGES = [
    "40.00%", "35.00%", "33.00%", "25.00%", "24.00%", "0.00%", "6.00%",
    "100%", "37.00%", "18.50%", "7.25%", "99.99%",
]

# Comma-grouped RM figures — the exact shape ("19,250,000") that
# apparently reads as address-like to the model. Includes the
# multi-token variants that arise from how a table extracts: "13,"
# and "200,000" can land as separate tokens depending on cell
# wrapping, so both single-token and split forms are covered.
RM_AMOUNTS = [
    "22,000,000", "19,250,000", "20,350,000", "18,150,000", "13,750,000",
    "18,200,000", "0", "3,300,000", "55,000,000", "374,300", "1,200,000",
    "6,500", "4,500", "50,000",
]

TABLE_HEADER_PHRASES = [
    "Before entering the market (%)", "Before entering the market (RM)",
    "After entering the market (%)", "After entering the market (RM)",
    "Market Share (%)", "Revenue (RM)", "Net Profit (RM)",
    "Growth Rate (%)", "TOTAL",
]

FINANCIAL_TABLE_ROW_TEMPLATES = [
    "{company} {pct} {rm} {pct} {rm}",
    "{company} {rm} {rm} {pct}",
    "{header} {rm}",
    "{header} {pct}",
    "TOTAL {pct} {rm} {pct} {rm}",
    "{pct} {rm}",
]


def build_financial_table_negatives(rng: random.Random, count: int) -> list[Sentence]:
    out = []
    for _ in range(count):
        template = rng.choice(FINANCIAL_TABLE_ROW_TEMPLATES)
        text = template.format(
            company=rng.choice(COMPANY_NAMES),
            pct=rng.choice(PERCENTAGES),
            rm=rng.choice(RM_AMOUNTS),
            header=rng.choice(TABLE_HEADER_PHRASES),
        )
        tokens = text.split(" ")
        out.append(Sentence(tokens=tokens, labels=["O"] * len(tokens)))

    # Bare single-cell negatives too — a table often extracts as ONE
    # token per cell with no surrounding row context at all.
    for value_pool in (COMPANY_NAMES, PERCENTAGES, RM_AMOUNTS, TABLE_HEADER_PHRASES):
        for value in value_pool:
            tokens = value.split(" ")
            out.append(Sentence(tokens=tokens, labels=["O"] * len(tokens)))

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v20", default="data/processed/semi_synthetic_v20.txt")
    parser.add_argument("--financial-negative-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--out", default="data/processed/semi_synthetic_v21.txt")
    parser.add_argument("--train-pool-out", default="data/processed/train_pool_v21.txt")
    parser.add_argument("--test-holdout-out", default="data/processed/test_holdout_v21.txt")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    v20 = parse_iob2_file(args.v20)
    print(f"v20 base: {len(v20)} sentences")

    financial_negatives = build_financial_table_negatives(rng, args.financial_negative_count)
    print(f"financial-table negatives: {len(financial_negatives)}")

    combined_raw = v20 + financial_negatives
    combined = []
    lead_word_fixes = 0
    for s in combined_raw:
        fixed, changed = normalize_address_lead_words(s)
        combined.append(fixed)
        lead_word_fixes += changed
    print(f"address lead-word boundary fixes applied: {lead_word_fixes}")

    rng.shuffle(combined)

    before = entity_token_counts(v20)
    after = entity_token_counts(combined)
    print("\nEntity token counts, v20 -> v21:")
    for tag in sorted(set(before) | set(after)):
        print(f"  {tag:8s} {before.get(tag, 0):6d} -> {after.get(tag, 0):6d}")

    write_iob2(combined, args.out)
    print(f"\nWrote {len(combined)} sentences -> {args.out}")

    train_pool, test_holdout = split_by_skeleton(combined, args.test_ratio, args.split_seed)
    train_skel = {skeleton(s) for s in train_pool}
    test_skel = {skeleton(s) for s in test_holdout}
    assert not (train_skel & test_skel), "skeleton leaked between splits — bug"

    write_iob2(train_pool, args.train_pool_out)
    write_iob2(test_holdout, args.test_holdout_out)
    print(f"train pool:   {len(train_pool)} sentences -> {args.train_pool_out}")
    print(f"test holdout: {len(test_holdout)} sentences -> {args.test_holdout_out}")
    print("skeleton overlap between splits: 0 (verified)")


if __name__ == "__main__":
    main()
