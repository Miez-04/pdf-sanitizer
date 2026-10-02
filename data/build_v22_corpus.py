"""
data/build_v22_corpus.py

Builds v22 from v20, NOT v21. Measured evidence (adversarial eval,
same fixed test set across all versions):

              PERSON F1   ADDRESS F1   Combined F1
    v19:      0.5217      0.6000       0.6383
    v20:      0.6087      0.6000       0.6809   <- best so far
    v21:      0.4667      0.4000       0.5556   <- regression, worse than v19

v21 added 345 financial-table negative sentences, all built from ~6
narrow row-shape templates. That fixed the ONE reported false positive
(a market-comparison table) but measurably hurt PERSON and ADDRESS
everywhere else — likely a case of a large, narrow-shaped negative
batch shifting the model's decision boundary more broadly than
intended. Confirmed via corpus inspection: v20 and v21 both label
every company-name token 100% consistently as O — the regression is
NOT a training-label bug, it's negative transfer from over-narrow,
over-large negative injection.

This version is deliberately conservative:

  1. ORGANISATION DISAMBIGUATION, kept small and varied (not the 345
     financial-table-only sentences from v21). Company names shown in
     SEVERAL different contexts (document header, standalone mention,
     inline sentence) — not just one financial-table row shape — so
     the model learns "this word shape is not PERSON/ADDRESS" more
     generally, rather than "this word shape in THIS ONE table layout
     is not PERSON/ADDRESS".

  2. BIN/BINTI CONTINUATION CONTRASTIVE PAIRS, per explicit user
     guidance: when Bin/Binti appears, the model should keep tagging
     I-PERSON through whatever follows it; when a name has no such
     connector (e.g. "Iskandar Ishak"), it's already complete at 2
     tokens and needs no further continuation. Generated as genuine
     minimal pairs — same name pool, with and without a Bin/Binti-
     linked second surname — so the presence/absence of the connector
     is the clearest signal in the data for when to keep extending.

Combines with v20. Torch-free, same reasoning as v18/19/20/21.
IMPORTANT: after training, compare the new adversarial score against
v20's 0.6809 baseline — NOT v21's regressed 0.5556 — to confirm this
iteration doesn't repeat the same mistake.

Usage:
    python -m data.build_v22_corpus
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
from data.build_v19_corpus import MALAY_NAMES, INDIAN_NAMES


# =======================================================================
# 1. Organisation disambiguation — small, varied, NOT one narrow shape.
# =======================================================================

ORG_NAMES = [
    "Shimano Inc.", "Daiwa Sports Sdn. Bhd.", "Penn Fishing", "Reelworks Co.",
    "Northport Logistics Services", "Maju Holdings Berhad",
    "Sunway Construction Group", "Perusahaan Otomobil Nasional Berhad",
    "Petronas Chemicals Group Berhad", "Tenaga Nasional Berhad",
    "Malayan Banking Berhad", "Genting Malaysia Berhad",
]

ORG_CONTEXT_TEMPLATES = [
    "{org}",                                    # bare standalone mention
    "Prepared for {org}",                       # document header
    "{org} is a leading player in this industry .",   # inline sentence
    "The report was submitted to {org} .",       # inline, different position
    "{org} recorded steady growth this year .",  # inline, subject position
    "Company : {org}",                           # labeled form field
]


def build_org_disambiguation_sentences(rng: random.Random, count: int) -> list[Sentence]:
    out = []
    for _ in range(count):
        org = rng.choice(ORG_NAMES)
        template = rng.choice(ORG_CONTEXT_TEMPLATES)
        text = template.format(org=org)
        tokens = text.split(" ")
        out.append(Sentence(tokens=tokens, labels=["O"] * len(tokens)))
    return out


# =======================================================================
# 2. Bin/Binti continuation contrastive pairs
# =======================================================================

# First-name-only pool (no connector) paired with a second surname —
# used to build BOTH the "with Bin/Binti, keep extending" and the
# "without it, already complete" forms of the same underlying name.
GIVEN_PLUS_SURNAME_PAIRS = [
    ("Iskandar", "Ishak"), ("Ahmad", "Faisal"), ("Hafizah", "Yusof"),
    ("Zulkifli", "Mangsor"), ("Rosnah", "Kamal"), ("Faridah", "Hassan"),
    ("Halimah", "Yusof"), ("Nurul", "Huda"), ("Aminah", "Zainal"),
    ("Rashid", "Karim"),
]
BIN_CONNECTORS_M = ["Bin", "bin", "b."]
BIN_CONNECTORS_F = ["Binti", "binti", "bt", "bte"]

CONTINUATION_LABEL_TEMPLATES = [
    "Name: {P}", "Nama: {P}", "Dr. {P}", "Encik {P}", "Puan {P}", "{P}",
]


def build_bin_binti_contrastive_pairs(rng: random.Random) -> list[Sentence]:
    out = []
    for given, surname in GIVEN_PLUS_SURNAME_PAIRS:
        template = rng.choice(CONTINUATION_LABEL_TEMPLATES)

        # Form A: no connector — name is complete at 2 tokens, nothing
        # after it should be tagged, even if more text follows.
        short_name = f"{given} {surname}"
        text_a = template.replace("{P}", short_name) + " is available for consultation ."
        out.append(_tag_person_prefix(text_a, short_name))

        # Form B: WITH a Bin/Binti connector plus a third name — the
        # ENTIRE chain, connector included, must stay I-PERSON; nothing
        # gets cut off at the connector itself.
        connector = rng.choice(BIN_CONNECTORS_M + BIN_CONNECTORS_F)
        extra_surname = rng.choice(MALAY_NAMES).split(" ")[-1]  # borrow a trailing surname token
        long_name = f"{given} {connector} {extra_surname}"
        text_b = template.replace("{P}", long_name) + " is available for consultation ."
        out.append(_tag_person_prefix(text_b, long_name))

    return out


def _tag_person_prefix(text: str, name: str) -> Sentence:
    tokens = text.split(" ")
    name_tokens = name.split(" ")
    labels = ["O"] * len(tokens)
    n = len(name_tokens)
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n] == name_tokens:
            labels[i] = "B-PERSON"
            for k in range(1, n):
                labels[i + k] = "I-PERSON"
            break
    return Sentence(tokens=tokens, labels=labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v20", default="data/processed/semi_synthetic_v20.txt")
    parser.add_argument("--org-negative-count", type=int, default=80,
                         help="Deliberately much smaller than v21's 345 — "
                              "see module docstring for why.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--out", default="data/processed/semi_synthetic_v22.txt")
    parser.add_argument("--train-pool-out", default="data/processed/train_pool_v22.txt")
    parser.add_argument("--test-holdout-out", default="data/processed/test_holdout_v22.txt")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    v20 = parse_iob2_file(args.v20)
    print(f"v20 base: {len(v20)} sentences")

    org_negatives = build_org_disambiguation_sentences(rng, args.org_negative_count)
    print(f"organisation-disambiguation sentences (6 varied contexts, not 1): {len(org_negatives)}")

    bin_binti_pairs = build_bin_binti_contrastive_pairs(rng)
    print(f"Bin/Binti contrastive pairs: {len(bin_binti_pairs)} "
          f"({len(GIVEN_PLUS_SURNAME_PAIRS)} base names x 2 forms)")

    combined_raw = v20 + org_negatives + bin_binti_pairs
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
    print("\nEntity token counts, v20 -> v22:")
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
    print("\nIMPORTANT: compare the new adversarial score against v20's 0.6809 "
          "baseline, not v21's regressed 0.5556.")


if __name__ == "__main__":
    main()
