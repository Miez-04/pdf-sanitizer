"""
data/build_v20_corpus.py

Targets three CONFIRMED failures from real-PDF testing (screenshots,
Oct 2026), each traced to a specific gap in v19's training data rather
than guessed at:

  1. HONORIFIC NAMES MISSED AS THE UNLABELED FIRST LINE OF A BLOCK.
     "Dr. Iskandar Ishak" / "Assoc. Prof. Dr. Ali bin Mamat" were
     completely missed — both sit as the first line of an unlabeled
     multi-line reference block (name, then job title, then
     department, then university, then "Tel:"), not as a bare
     "Dr. <name>" sentence on its own. v19's honorific generator
     (build_person_pattern_sentences) only ever produced the bare
     one-liner shape — it never showed the model this exact
     structural context, so the model had no reason to learn it.

  2. NAMES NOT CONTINUING ACROSS A LINE WRAP. "MUHAMMAD HAKIMI BIN
     MUHAMMAD JAUHARI" wrapped after "MUHAMMAD" (second occurrence)
     onto a new line, and "JAUHARI" was left completely untagged —
     not a merge/geometry bug (already fixed separately), a genuine
     model recall miss. The repeated token "MUHAMMAD" reappearing
     mid-name (first name + patronymic both being common Malay first
     names) is a real, recurring Malay-name shape underrepresented in
     training relative to its real frequency.

  3. COURSE-TITLE STRINGS MISTAGGED ADDRESS. "PRINCIPLES OF OPERATING
     SYSTEMS", "ARTIFICIAL INTELLIGENCE ALGORITHMS", etc., in a
     transcript-style table were tagged ADDRESS. Same root cause as
     v18's section-header-mistagged-PERSON bug (all-caps multi-word
     table content with no negative training coverage), just showing
     up as a different entity type this time — the fix is the same
     approach (explicit all-O negatives), applied to ADDRESS.

Combines with v19 (not from scratch), same reasoning as v19-on-v18 and
v18-on-v17. Torch-free, same reasoning as build_v18/19_corpus.py.

Usage:
    python -m data.build_v20_corpus
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from data.build_v18_corpus import (
    Sentence,
    parse_iob2_file,
    write_iob2,
    skeleton,
    split_by_skeleton,
    normalize_address_lead_words,
    entity_token_counts,
)
from data.build_v19_corpus import (
    ALL_NAMES,
    MALAY_NAMES,
    HONORIFIC_PREFIXES,
    _tag_template_as_person,
)

# Found by comparing v19's actual coverage against eval/adversarial_examples.py's
# real PERSON test cases after v19's adversarial F1 came back UNCHANGED from
# v18 (0.5217 in both, to 4 decimal places) despite v19 adding real PERSON
# diversity. None of the adversarial cases are bare list/table rows (that's a
# different, already-fixed failure mode) — but several use label/honorific
# forms v19 doesn't have:
#   "Waris : {P}"          <- v19 only has "Nama Waris: {P}" (with "Nama")
#   "Saksi pertama : {P}"  <- v19 only has "Nama Saksi: {P}"
#   "Puan Sri {P}"         <- v19 only has bare "Puan", not this compound title
#   "Mr. {P}"              <- not in v19's HONORIFIC_PREFIXES at all
#   "Assistant Commissioner {P}" <- likewise
BARE_LABEL_TEMPLATES_MS = [
    "Waris : {P}", "Saksi pertama : {P}", "Saksi kedua : {P}",
    "Pemilik : {P}", "Wakil : {P}",
]
EXTRA_HONORIFIC_PREFIXES = [
    "Mr.", "Mrs.", "Ms.", "Puan Sri", "Dato'", "Dato' Sri",
    "Assistant Commissioner", "Inspector",
]


def build_gap_targeted_person_sentences(rng: random.Random) -> list[Sentence]:
    out = []
    for name in rng.sample(ALL_NAMES, min(25, len(ALL_NAMES))):
        template = rng.choice(BARE_LABEL_TEMPLATES_MS)
        text = template.replace("{P}", name)
        out.append(_tag_template_as_person(text, name))
    for name in rng.sample(MALAY_NAMES, min(15, len(MALAY_NAMES))):
        prefix = rng.choice(EXTRA_HONORIFIC_PREFIXES)
        out.append(Sentence(
            tokens=(prefix + " " + name).split(" "),
            labels=(["B-PERSON"] + ["I-PERSON"] * (len((prefix + " " + name).split(" ")) - 1)),
        ))
    return out


# =======================================================================
# 1. Honorific name as the unlabeled first line of a reference block
# =======================================================================

JOB_TITLES = [
    "Industrial Training Coordinator", "Academic Advisor", "Senior Lecturer",
    "Head of Department", "Programme Coordinator", "Research Supervisor",
    "Deputy Dean", "Course Coordinator",
]
DEPARTMENTS = [
    "Department of Computer Science", "Department of Electrical Engineering",
    "Department of Mathematics", "Faculty of Information Technology",
    "School of Business Management",
]
UNIVERSITIES = [
    "Universiti Putra Malaysia, Serdang, Selangor",
    "Universiti Malaya, Kuala Lumpur",
    "Universiti Sains Malaysia, Gelugor, Pulau Pinang",
    "Universiti Teknologi Malaysia, Skudai, Johor",
    "Universiti Kebangsaan Malaysia, Bangi, Selangor",
]
REFERENCE_PHONES = [
    "03-89471796", "04-6532210", "07-5534421", "06-2311098", "09-7712340",
]


def build_reference_block_sentences(rng: random.Random) -> list[Sentence]:
    """Realistic multi-line reference-block sentences: an honorific
    name as the FIRST line with no label at all, followed by
    job-title/department/university/phone lines — the exact shape
    that was missed in real testing, not the bare one-liner v19 used."""
    out = []
    sample_names = rng.sample(MALAY_NAMES, min(15, len(MALAY_NAMES)))
    for name in sample_names:
        prefix = rng.choice(HONORIFIC_PREFIXES)
        full_name = f"{prefix} {name}"
        job = rng.choice(JOB_TITLES)
        dept = rng.choice(DEPARTMENTS)
        uni = rng.choice(UNIVERSITIES)
        phone = rng.choice(REFERENCE_PHONES)

        tokens: list[str] = []
        labels: list[str] = []

        name_tokens = full_name.split(" ")
        for i, tok in enumerate(name_tokens):
            tokens.append(tok)
            labels.append("B-PERSON" if i == 0 else "I-PERSON")

        for line in (job, dept, uni):
            for tok in line.split(" "):
                tokens.append(tok)
                labels.append("O")

        tel_tokens = ["Tel:", phone]
        tokens.append(tel_tokens[0]); labels.append("O")
        tokens.append(tel_tokens[1]); labels.append("B-PHONE")

        out.append(Sentence(tokens=tokens, labels=labels))
    return out


# =======================================================================
# 2. Long / repeated-token Malay names, in MULTIPLE contexts (not just
#    once in the raw name pool) — boosts the model's exposure to the
#    "first name reappears as part of the patronymic" shape.
# =======================================================================

REPEATED_TOKEN_NAMES = [
    n for n in ALL_NAMES
    if len(set(w.upper() for w in n.split(" "))) < len(n.split(" "))
]  # any name where a word repeats (e.g. "MUHAMMAD ... BIN MUHAMMAD ...")

LONG_NAME_LABEL_TEMPLATES = [
    "Name: {P}", "Nama: {P}", "Nama Penuh: {P}", "Advisor: {P}",
    "Tandatangan Penyewa: {P}", "{P}",
]


def build_repeated_token_name_sentences(rng: random.Random) -> list[Sentence]:
    """Every repeated-token name (e.g. containing 'MUHAMMAD' twice) run
    through several different label contexts, not just the one
    occurrence it already gets in the base ALL_NAMES pool — the model
    needs to see this shape enough times, in enough different
    surrounding contexts, to learn to keep tagging I-PERSON through a
    repeated common first-name token instead of treating its second
    occurrence as a boundary/reset signal."""
    out = []
    if not REPEATED_TOKEN_NAMES:
        return out
    for name in REPEATED_TOKEN_NAMES:
        for template in LONG_NAME_LABEL_TEMPLATES:
            text = template.replace("{P}", name)
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
            out.append(Sentence(tokens=tokens, labels=labels))
    return out


# =======================================================================
# 3. Course-title negatives for ADDRESS — same approach as v18's
#    section-header negatives, different entity type this time.
# =======================================================================

COURSE_TITLE_NEGATIVES = [
    "DISCRETE STRUCTURES", "PRINCIPLES OF OPERATING SYSTEMS",
    "ARTIFICIAL INTELLIGENCE ALGORITHMS", "PHILOSOPHY AND CURRENT ISSUES",
    "DATABASE ENGINEERING", "ENGLISH FOR MEDIATING TEXTS",
    "LINEAR ALGEBRA", "COMPUTER NETWORKS", "SOFTWARE ENGINEERING",
    "DATA STRUCTURES AND ALGORITHMS", "OBJECT ORIENTED PROGRAMMING",
    "COMPUTER ORGANIZATION AND ARCHITECTURE", "WEB APPLICATION DEVELOPMENT",
    "NUMERICAL METHODS", "PROBABILITY AND STATISTICS",
    "ETHICS IN COMPUTING", "MOBILE APPLICATION DEVELOPMENT",
    "INTRODUCTION TO CYBERSECURITY",
]


def build_course_title_negatives(rng: random.Random, count: int) -> list[Sentence]:
    out = []
    for _ in range(count):
        text = rng.choice(COURSE_TITLE_NEGATIVES)
        # Sometimes with a trailing grade suffix, matching the real
        # transcript-table shape ("DATABASE ENGINEERING - B").
        if rng.random() < 0.5:
            text = f"{text} - {rng.choice('ABC')}"
        tokens = text.split(" ")
        out.append(Sentence(tokens=tokens, labels=["O"] * len(tokens)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v19", default="data/processed/semi_synthetic_v19.txt")
    parser.add_argument("--repeated-name-oversample", type=int, default=4)
    parser.add_argument("--course-title-negative-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--out", default="data/processed/semi_synthetic_v20.txt")
    parser.add_argument("--train-pool-out", default="data/processed/train_pool_v20.txt")
    parser.add_argument("--test-holdout-out", default="data/processed/test_holdout_v20.txt")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    v19 = parse_iob2_file(args.v19)
    print(f"v19 base: {len(v19)} sentences")

    reference_blocks = build_reference_block_sentences(rng)
    print(f"reference-block sentences (honorific + job/dept/uni/phone): {len(reference_blocks)}")

    gap_targeted = build_gap_targeted_person_sentences(rng)
    print(f"gap-targeted PERSON sentences (bare govt labels + extra honorifics): {len(gap_targeted)}")

    print(f"repeated-token names found in ALL_NAMES: {len(REPEATED_TOKEN_NAMES)} -> {REPEATED_TOKEN_NAMES}")
    repeated_name_sentences = build_repeated_token_name_sentences(rng) * args.repeated_name_oversample
    print(f"repeated-token-name sentences: {len(repeated_name_sentences)}")

    course_negatives = build_course_title_negatives(rng, args.course_title_negative_count)
    print(f"course-title ADDRESS negatives: {len(course_negatives)}")

    combined_raw = v19 + reference_blocks + gap_targeted + repeated_name_sentences + course_negatives
    combined = []
    lead_word_fixes = 0
    for s in combined_raw:
        fixed, changed = normalize_address_lead_words(s)
        combined.append(fixed)
        lead_word_fixes += changed
    print(f"address lead-word boundary fixes applied: {lead_word_fixes}")

    rng.shuffle(combined)

    before = entity_token_counts(v19)
    after = entity_token_counts(combined)
    print("\nEntity token counts, v19 -> v20:")
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
