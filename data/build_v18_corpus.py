"""
data/build_v18_corpus.py

Builds semi_synthetic_v18.txt by COMBINING what's already been
achieved across the project's history, rather than generating more
volume from scratch:

  1. semi_synthetic_v17.txt         — the most complete synthetic
                                       corpus (carries forward every
                                       fix from v3 through v17), with
                                       exact-duplicate sentences
                                       removed (v17 had 495/6714 —
                                       7.4% — exact dupes; these waste
                                       training steps without adding
                                       signal).
  2. real_samples.txt +
     real_samples_2.txt             — the two hand-labeled REAL
                                       documents (a resume, a hospital
                                       admission form). Real layouts,
                                       real vocabulary — the highest-
                                       value examples in the project,
                                       but only 55 sentences total
                                       against v17's ~6700, so they're
                                       oversampled (see
                                       REAL_SAMPLE_OVERSAMPLE below)
                                       or they'd be statistically
                                       invisible to the loss function.
  3. Structural hard negatives       — all-O section headers/acronyms
                                       ("COMPUTER SKILLS", "CASE Tool,
                                       Linux,", "LANGUAGE"...), added
                                       because real resume-PDF testing
                                       (Sept 2026, model 13/17
                                       comparison) showed exactly
                                       these strings getting
                                       mistagged B-PERSON. Zero
                                       positive entities — the point
                                       is teaching the model these
                                       specific surface patterns are
                                       NOT PERSON.

Does NOT re-run the spaCy-based generation in build_corpus.py against
the raw news/merged corpora — v17 already represents that accumulated
work. Re-running it from scratch would be redundant effort, not an
improvement; if you want MORE fresh synthetic volume on top of this
(not just what v17 already has), that's a separate, additive step —
run build_corpus.py again with a new --out path and concatenate.

Deliberately torch-free: models/dataset.py imports torch at module
level (needed for its PIIDataset/PyTorch Dataset class), which this
script has no need for — it only needs the plain parse/Sentence logic,
duplicated here so this can run in any environment that can read text
files, no ML framework required.

Usage:
    python -m data.build_v18_corpus \
        --v17 data/processed/semi_synthetic_v17.txt \
        --real data/processed/real_samples.txt data/processed/real_samples_2.txt \
        --out data/processed/semi_synthetic_v18.txt \
        --train-pool-out data/processed/train_pool_v18.txt \
        --test-holdout-out data/processed/test_holdout_v18.txt
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------
# Torch-free reimplementation of models.dataset.{Sentence, parse_iob2_file}
# and eval.split_holdout.{skeleton, split_by_skeleton}. Kept byte-for-byte
# equivalent in logic to those — see the docstring above for why this
# duplicates rather than imports them.
# ---------------------------------------------------------------------

@dataclass(slots=True)
class Sentence:
    tokens: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tokens)


def _split_compound_line(token_field: str, label: str) -> list[tuple[str, str]]:
    words = token_field.split(" ")
    if len(words) == 1:
        return [(words[0], label)]
    pairs = [(words[0], label)]
    continuation = "I-" + label[2:] if label.startswith("B-") else label
    pairs.extend((w, continuation) for w in words[1:])
    return pairs


def parse_iob2_file(path: str | Path) -> list[Sentence]:
    sentences: list[Sentence] = []
    current = Sentence()
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line.strip():
                if len(current) > 0:
                    sentences.append(current)
                    current = Sentence()
                continue
            *token_parts, label = line.split(" ")
            token_field = " ".join(token_parts)
            if not token_field:
                continue
            for word, tag in _split_compound_line(token_field, label):
                current.tokens.append(word)
                current.labels.append(tag)
    if len(current) > 0:
        sentences.append(current)
    return sentences


def write_iob2(sentences: list[Sentence], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for tok, lab in zip(sent.tokens, sent.labels):
                f.write(f"{tok} {lab}\n")
            f.write("\n")


def skeleton(sent: Sentence) -> str:
    """Identical to eval.split_holdout.skeleton — must match exactly,
    since the whole point is that train/test splits produced here are
    comparable to every prior version's holdout methodology."""
    out = []
    for tok, lab in zip(sent.tokens, sent.labels):
        if lab == "O":
            out.append(tok)
        elif lab.startswith("B-"):
            out.append(f"<{lab[2:]}>")
    return " ".join(out)


def split_by_skeleton(
    sentences: list[Sentence], test_ratio: float, seed: int
) -> tuple[list[Sentence], list[Sentence]]:
    groups: dict[str, list[Sentence]] = defaultdict(list)
    for s in sentences:
        groups[skeleton(s)].append(s)

    keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    cut = int(len(keys) * (1 - test_ratio))
    train_keys, test_keys = keys[:cut], keys[cut:]

    train_pool = [s for k in train_keys for s in groups[k]]
    test_holdout = [s for k in test_keys for s in groups[k]]
    return train_pool, test_holdout


# ---------------------------------------------------------------------
# Structural hard negatives — see module docstring point 3.
# Mirrors what was proposed for data/entity_mutation.py earlier; kept
# here instead so this script stays self-contained and doesn't require
# editing that file to use it.
# ---------------------------------------------------------------------

# Confirmed via direct corpus analysis (Sept 2026) against a real
# redacted-PDF test: these words are inconsistently included/excluded
# from the ADDRESS span across the existing corpus.
#   'No.'  (with period) -> tagged O          297/297 times (100%)
#   'No'   (no period)   -> tagged B-ADDRESS   69/69  times (100%)
#   'Lot'                -> inconsistent: 61 B-ADDRESS / 44 O / 1 I-ADDRESS
#   'Unit'                -> inconsistent: 74 B-ADDRESS / 22 O / 1 I-ADDRESS
#   'Sublot'              -> tagged O           11/11  times (100%)
#   'Block'/'Blok'        -> mostly B-ADDRESS, but 22 O for 'Block'
# The 'No.'/'No' split is a punctuation artifact (whatever generated
# each variant did so with opposite, but each internally consistent,
# conventions) teaching the model contradictory signal purely on
# trailing punctuation. 'Lot'/'Unit'/'Block' are genuinely inconsistent
# example-to-example, blurring the CRF's boundary confidence. Both
# surfaced directly in real-PDF testing: the model consistently leaves
# these lead words unredacted while correctly redacting everything
# after them — e.g. "No. 55, Jalan Pasir Pinji 3, ..." redacts to just
# "No." left exposed, rest blacked out.
#
# Resolution: always fold the lead word INTO the ADDRESS span. For
# redaction, leaving "No." or "Sublot" exposed next to a black box is
# both a minor residual-identifying-fragment leak and looks visibly
# broken — there's no real argument for excluding it, unlike some
# entity-boundary questions where either convention is defensible.
ADDRESS_LEAD_WORDS = {"no.", "no", "lot", "sublot", "unit", "block", "blok"}


def normalize_address_lead_words(sent: Sentence) -> tuple[Sentence, int]:
    """Returns (possibly-modified Sentence, number of tokens changed).
    Never mutates the input in place — dataclass fields are copied."""
    tokens, labels = list(sent.tokens), list(sent.labels)
    changed = 0
    for i in range(len(tokens) - 1):
        word = tokens[i].lower().rstrip(".")
        if labels[i] != "O" or word not in ADDRESS_LEAD_WORDS:
            continue
        if labels[i + 1] == "B-ADDRESS":
            labels[i] = "B-ADDRESS"
            labels[i + 1] = "I-ADDRESS"
            changed += 1
        elif labels[i + 1] == "I-ADDRESS":
            labels[i] = "I-ADDRESS"
            changed += 1
    return Sentence(tokens=tokens, labels=labels), changed


SECTION_HEADER_NEGATIVES = [
    "COMPUTER SKILLS", "LANGUAGE SKILLS", "PERSONAL PARTICULARS",
    "WORKING EXPERIENCE", "LEADERSHIP SKILLS", "ACADEMIC QUALIFICATION",
    "CO-CURRICULAR ACTIVITIES", "PERSONAL DETAILS", "CAREER OBJECTIVE",
    "PROFESSIONAL EXPERIENCE", "SKILLS SUMMARY", "REFERENCES", "EDUCATION",
    "EMPLOYMENT HISTORY", "TECHNICAL SKILLS", "AREAS OF INTEREST",
    "EXTRA-CURRICULAR ACTIVITIES", "TRAINING ATTENDED", "HOBBIES",
]
ACRONYM_NEGATIVE_PHRASES = [
    "Rational CASE Tool, Linux",
    "CGPA : 3.55",
    "SPM Grade : 5A",
    "IELTS Band 7.5",
    "TOEFL Score : 600",
    "MUET Band 4",
]


def build_structural_negatives(rng: random.Random, count: int) -> list[Sentence]:
    out = []
    for _ in range(count):
        text = rng.choice(
            SECTION_HEADER_NEGATIVES if rng.random() < 0.7 else ACRONYM_NEGATIVE_PHRASES
        )
        tokens = text.split(" ")
        out.append(Sentence(tokens=tokens, labels=["O"] * len(tokens)))
    return out


def entity_token_counts(sentences: list[Sentence]) -> Counter:
    c = Counter()
    for s in sentences:
        for lab in s.labels:
            if lab != "O":
                c[lab.split("-")[-1]] += 1
    return c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v17", default="data/processed/semi_synthetic_v17.txt")
    parser.add_argument(
        "--real", nargs="+",
        default=["data/processed/real_samples.txt", "data/processed/real_samples_2.txt"],
    )
    parser.add_argument(
        "--real-sample-oversample", type=int, default=6,
        help="How many times to repeat each real hand-labeled sentence "
             "so it isn't statistically invisible against v17's volume.",
    )
    parser.add_argument("--hard-negative-count", type=int, default=350)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--out", default="data/processed/semi_synthetic_v18.txt")
    parser.add_argument("--train-pool-out", default="data/processed/train_pool_v18.txt")
    parser.add_argument("--test-holdout-out", default="data/processed/test_holdout_v18.txt")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ---- 1. v17, deduplicated ----
    v17 = parse_iob2_file(args.v17)
    seen = set()
    v17_deduped = []
    for s in v17:
        key = tuple(zip(s.tokens, s.labels))
        if key not in seen:
            seen.add(key)
            v17_deduped.append(s)
    print(f"v17: {len(v17)} sentences -> {len(v17_deduped)} after removing "
          f"{len(v17) - len(v17_deduped)} exact duplicates")

    # ---- 2. real samples, oversampled ----
    real_sentences = []
    for path in args.real:
        parsed = parse_iob2_file(path)
        real_sentences.extend(parsed)
        print(f"{path}: {len(parsed)} sentences")
    real_oversampled = real_sentences * args.real_sample_oversample
    print(f"real samples oversampled {args.real_sample_oversample}x -> "
          f"{len(real_oversampled)} sentences")

    # ---- 3. structural hard negatives ----
    negatives = build_structural_negatives(rng, args.hard_negative_count)
    print(f"structural hard negatives: {len(negatives)} sentences")

    # ---- combine + normalize address lead-word boundaries ----
    combined_raw = v17_deduped + real_oversampled + negatives
    combined = []
    lead_word_fixes = 0
    for s in combined_raw:
        fixed, changed = normalize_address_lead_words(s)
        combined.append(fixed)
        lead_word_fixes += changed
    print(f"address lead-word boundary fixes applied: {lead_word_fixes} tokens "
          f"across the combined corpus (see ADDRESS_LEAD_WORDS)")

    rng.shuffle(combined)

    before = entity_token_counts(v17)
    after = entity_token_counts(combined)
    print("\nEntity token counts, v17 (raw) -> v18 (combined):")
    for tag in sorted(set(before) | set(after)):
        print(f"  {tag:8s} {before.get(tag, 0):6d} -> {after.get(tag, 0):6d}")

    write_iob2(combined, args.out)
    print(f"\nWrote {len(combined)} sentences -> {args.out}")

    # ---- skeleton-disjoint split, same method as eval/split_holdout.py ----
    train_pool, test_holdout = split_by_skeleton(combined, args.test_ratio, args.split_seed)
    train_skel = {skeleton(s) for s in train_pool}
    test_skel = {skeleton(s) for s in test_holdout}
    overlap = train_skel & test_skel
    assert not overlap, f"{len(overlap)} skeletons leaked into both splits — bug"

    write_iob2(train_pool, args.train_pool_out)
    write_iob2(test_holdout, args.test_holdout_out)
    print(f"train pool:   {len(train_pool)} sentences, {len(train_skel)} unique skeletons -> {args.train_pool_out}")
    print(f"test holdout: {len(test_holdout)} sentences, {len(test_skel)} unique skeletons -> {args.test_holdout_out}")
    print("skeleton overlap between splits: 0 (verified)")


if __name__ == "__main__":
    main()
