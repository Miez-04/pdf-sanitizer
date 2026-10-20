"""
data/build_v25_corpus.py

Adds TABLE / HEADING data on top of the v24 corpus. Reason (measured on a real
academic transcript, token-level against a small gold file):

                      real PII covered   wrong tokens   wrong spans   micro F1
    v25_ns model            17 / 17            12             0          0.74
    v26_ns model            14 / 17            47             5          0.36

v26 had learned long names, names in a row and addresses without a postcode
from prose, letters and lists. A transcript is none of those: a printed header
row ("NAME GENDER PROGRAM FACULTY ADDRESS NRIC/PASSPORT STUDENT ID ...") is
followed by a row of values, then course rows full of codes, grades and
all-caps words. The model called "ACADEMIC ... NAME GENDER", "DIPLOMA FINAL",
"LU A- MANAGEMENT" and grade-table rows names and addresses, and cut the real
name "MUHAMMAD HAKIMI BIN ..." short.

This layer teaches, with label O or exact spans:
  1. header rows and ALL-CAPS headings are not names or addresses;
  2. in a flattened value row the gender word, student number, citizenship,
     programme and CGPA are O, and only the name / NRIC / address are entities,
     with an address often SPLIT in two pieces (street ... other cells ...
     postcode city state);
  3. long course / grade / semester tables (codes, grades, "UNITS OBTAINED: (18)")
     are O;
  4. invoice-style tables: a bill-to name and address are entities, items,
     amounts and totals are O.

Independence: imports nothing from regex_engine. Honorific, punctuation and
generator conventions are the same as data/build_v23_corpus.py. No sentence,
name or number is copied from a test document. Torch-free.

    python -m data.build_v25_corpus --v24 data/processed/semi_synthetic_v24_ns3k.txt ^
        --out data/processed/semi_synthetic_v25_ns3k.txt ^
        --train-pool-out data/processed/train_pool_v25_ns3k.txt ^
        --test-holdout-out data/processed/test_holdout_v25_ns3k.txt
"""

from __future__ import annotations

import argparse
import random
import re

from data.build_v18_corpus import (
    Sentence,
    entity_token_counts,
    parse_iob2_file,
    skeleton,
    split_by_skeleton,
    write_iob2,
)
from data.build_v23_corpus import _cat, _entity, _o, address_tokens, nric_tokens, phone_tokens
from data.build_v24_corpus import mixed_person_tokens

TITLES = ["ACADEMIC RECORD", "STATEMENT OF RESULTS", "OFFICIAL TRANSCRIPT", "STUDENT RESULT SLIP",
          "SLIP KEPUTUSAN PEPERIKSAAN", "KAD LAPORAN AKADEMIK", "SEMESTER RESULT", "ACADEMIC TRANSCRIPT"]
LABELS = ["NAME", "GENDER", "PROGRAM", "FACULTY", "ADDRESS", "NRIC/PASSPORT", "STUDENT ID", "CITIZENSHIP",
          "INTAKE SESSION", "LEVEL", "RESULTS", "FINAL SESSION", "CGPA", "STATUS", "DATE OF BIRTH", "RACE",
          "RELIGION", "NO. MATRIK", "NAMA", "JANTINA", "PROGRAM PENGAJIAN", "FAKULTI", "ALAMAT", "KEWARGANEGARAAN"]
PROGRAMS = ["DIPLOMA IN BUSINESS STUDIES", "BACHELOR OF ENGINEERING", "DIPLOMA SAINS KOMPUTER", "SIJIL TEKNOLOGI MAKLUMAT",
            "BACHELOR OF ACCOUNTANCY", "DIPLOMA KEJURUTERAAN ELEKTRIK", "MASTER OF EDUCATION", "DIPLOMA PERAKAUNAN"]
FACULTIES = ["FAKULTI KEJURUTERAAN ELEKTRIK", "FACULTY OF BUSINESS AND MANAGEMENT", "KOLEJ KEMAHIRAN TINGGI",
             "SCHOOL OF HEALTH SCIENCES", "FAKULTI SAINS SOSIAL", "FACULTY OF LAW"]
GENDERS = ["MALE", "FEMALE", "LELAKI", "PEREMPUAN"]
CITIZEN = ["WARGANEGARA MALAYSIA", "MALAYSIAN", "BUKAN WARGANEGARA", "MALAYSIA"]
STATUS = ["COMPLETED", "GRADUATED", "ONGOING", "TAMAT", "ACTIVE"]
MONTHS = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER",
          "NOVEMBER", "DECEMBER"]
COURSES = ["ENGINEERING MATHEMATICS", "FINANCIAL ACCOUNTING", "COMMUNICATION SKILLS", "DATA STRUCTURES",
           "COMPUTER NETWORKS", "GENERAL PHYSICS", "MARKETING PRINCIPLES", "ISLAMIC STUDIES", "BAHASA MALAYSIA",
           "BUSINESS STATISTICS", "ENGLISH FOR ACADEMIC PURPOSES", "ORGANISATIONAL BEHAVIOUR", "ELECTRIC CIRCUITS",
           "WEB PROGRAMMING", "ECONOMICS FOR MANAGERS", "TECHNICAL DRAWING", "LINEAR ALGEBRA", "ENTREPRENEURSHIP",
           "MOBILE APPLICATION DEVELOPMENT", "CIVIC AND ETHICS", "DATABASE SYSTEMS", "PROJECT MANAGEMENT"]
GRADES = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "D", "F"]
HEADINGS = ["BAHAGIAN A", "MAKLUMAT PERIBADI PELAJAR", "ACADEMIC RECORD", "SECTION B", "STUDENT INFORMATION",
            "FINAL RESULT", "DIPLOMA PROGRAMME", "COURSE CODE", "CREDIT UNIT", "GRADE POINT", "LAMPIRAN B", "LAMPIRAN D",
            "JADUAL 2", "SENARAI SEMAK", "TERMA DAN SYARAT", "PENGAKUAN PELAJAR", "STATUS KEPUTUSAN", "SEMESTER SUMMARY",
            "NAME GENDER PROGRAM", "FACULTY ADDRESS", "LEVEL RESULTS", "CGPA STATUS", "PART 1 SESSION 2", "UNITS OBTAINED",
            "STATEMENT OF ACCOUNT", "PERINCIAN BAYARAN", "OFFICIAL USE ONLY", "CATATAN PENTING"]
ITEMS = ["WIRELESS ROUTER", "OFFICE CHAIR", "A4 PAPER 5 REAMS", "LAPTOP BAG", "PRINTER TONER", "DESK LAMP",
         "NETWORK CABLE 10M", "USB FLASH DRIVE", "WHITEBOARD MARKER SET"]
_POSTCODE = re.compile(r"^\d{5}[,.]?$")


def _up(tokens: list[str]) -> list[str]:
    return [t.upper() for t in tokens]


def _text(s: str) -> Sentence:
    t, l = _o(s)
    return Sentence(tokens=t, labels=l)


def _ent(kind: str, tokens: list[str]) -> Sentence:
    t, l = _entity(kind, tokens)
    return Sentence(tokens=t, labels=l)


def _money(rng: random.Random) -> str:
    return f"RM{rng.randint(5, 9000):,}.{rng.randint(0, 99):02d}"


def label_row(rng: random.Random) -> Sentence:
    n = rng.randint(5, 12)
    picked = sorted(rng.sample(range(len(LABELS)), n))
    return _text(" ".join(LABELS[i] for i in picked))


def _address_pieces(rng: random.Random) -> list[list[str]]:
    """Address tokens in capitals, split in two at the postcode when there is one
    (a table prints the street in one cell and 'postcode city state' in another)."""
    toks = _up(address_tokens(rng))
    for i, t in enumerate(toks):
        if _POSTCODE.match(t) and 2 <= i < len(toks) - 1 and rng.random() < 0.7:
            return [toks[:i], toks[i:]]
    return [toks]


def value_row(rng: random.Random) -> Sentence:
    """A flattened row of table values. Only the name, NRIC and address are entities."""
    parts: list[Sentence] = []
    if rng.random() < 0.85:
        parts.append(_text(rng.choice(GENDERS)))
    parts.append(_ent("PERSON", _up(mixed_person_tokens(rng, 0.45))))
    if rng.random() < 0.7:
        parts.append(_ent("NRIC", nric_tokens(rng)))
    pieces = _address_pieces(rng) if rng.random() < 0.9 else []
    if pieces:
        parts.append(_ent("ADDRESS", pieces[0]))
    parts.append(_text(str(rng.randint(2018000000, 2025999999))))            # student number: not an entity here
    parts.append(_text(rng.choice(CITIZEN)))
    parts.append(_text(f"{rng.choice(MONTHS)} {rng.randint(2018, 2025)}"))
    parts.append(_text(rng.choice(STATUS)))
    parts.append(_text(rng.choice(PROGRAMS)))
    if len(pieces) > 1:
        parts.append(_ent("ADDRESS", pieces[1]))
    parts.append(_text(f"{rng.uniform(2.0, 4.0):.2f}"))
    return _cat(parts)


def semester(rng: random.Random, number: int) -> Sentence:
    year = rng.randint(2019, 2024)
    toks = f"PART {number} {year}{rng.randint(1, 4)} - SESSION {rng.randint(1, 2)} {year}/{year + 1}".split()
    rows: list[str] = []
    for _ in range(rng.randint(4, 7)):
        code = "".join(rng.choice("ABCDEFGHIJKLMNPRSTUVW") for _ in range(3)) + str(rng.randint(100, 399))
        rows.append(f"{rng.choice(COURSES)} {code} {rng.choice([3.0, 3.33, 3.67, 4.0, 2.0]):.2f} "
                    f"{rng.choice([1, 2, 3, 4]):.2f} LU {rng.choice(GRADES)}")
    gpa = f"{rng.uniform(2.0, 4.0):.2f}"
    tail = f"GPA : {gpa} CGPA : UNITS OBTAINED: ({rng.randint(12, 90)}) {rng.randint(12, 24)} AD {gpa}"
    return _text(" ".join(toks) + " " + " ".join(rows) + " " + tail)


def transcript_full(rng: random.Random) -> Sentence:
    parts = [_text(rng.choice(TITLES)), label_row(rng), value_row(rng)]
    parts += [semester(rng, i + 1) for i in range(rng.randint(1, 2))]
    if rng.random() < 0.6:
        parts.append(_text(f"DATE : {rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2020, 2026)} "
                           "THIS IS COMPUTER GENERATED. NO SIGN NEEDED"))
    return _cat(parts)


def label_value_short(rng: random.Random) -> Sentence:
    parts = [label_row(rng), value_row(rng)]
    if rng.random() < 0.5:
        parts.insert(0, _text(rng.choice(TITLES)))
    return _cat(parts)


def course_table_only(rng: random.Random) -> Sentence:
    parts = [_text(rng.choice(TITLES))] if rng.random() < 0.5 else []
    parts += [semester(rng, i + 1) for i in range(rng.randint(1, 3))]
    return _cat(parts)


def invoice_table(rng: random.Random) -> Sentence:
    parts = [_text(f"INVOICE NO : {rng.randint(1000, 99999)} DATE : {rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2022, 2026)} BILL TO :"),
             _ent("PERSON", mixed_person_tokens(rng, 0.3)), _ent("ADDRESS", address_tokens(rng))]
    if rng.random() < 0.6:
        parts += [_text("TEL :"), _ent("PHONE", phone_tokens(rng))]
    parts.append(_text("DESCRIPTION QTY UNIT PRICE AMOUNT"))
    total = 0.0
    for _ in range(rng.randint(2, 5)):
        q, p = rng.randint(1, 9), rng.randint(5, 900)
        total += q * p
        parts.append(_text(f"{rng.choice(ITEMS)} {q} RM{p}.00 RM{q * p:,}.00"))
    parts.append(_text(f"SUBTOTAL RM{total:,.2f} SST 8% RM{total * 0.08:,.2f} TOTAL RM{total * 1.08:,.2f}"))
    return _cat(parts)


def headings_only(rng: random.Random) -> Sentence:
    n = rng.randint(3, 8)
    words = " ".join(rng.choice(HEADINGS) for _ in range(n))
    if rng.random() < 0.4:
        words += f" {rng.randint(1, 60)} PAGE {rng.randint(1, 3)} OF {rng.randint(3, 9)}"
    return _text(words)


FAMILIES = [(transcript_full, 0.22), (label_value_short, 0.25), (course_table_only, 0.15),
            (invoice_table, 0.18), (headings_only, 0.20)]


def build_layer(rng: random.Random, count: int) -> list[Sentence]:
    funcs = [f for f, _ in FAMILIES]
    weights = [w for _, w in FAMILIES]
    return [rng.choices(funcs, weights)[0](rng) for _ in range(count)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v24", default="data/processed/semi_synthetic_v24_ns3k.txt")
    ap.add_argument("--layer-count", type=int, default=1200,
                    help="new sequences (some are long: a transcript is about 200 tokens)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--split-seed", type=int, default=123)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--out", default="data/processed/semi_synthetic_v25_ns3k.txt")
    ap.add_argument("--train-pool-out", default="data/processed/train_pool_v25_ns3k.txt")
    ap.add_argument("--test-holdout-out", default="data/processed/test_holdout_v25_ns3k.txt")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = parse_iob2_file(args.v24)
    layer = build_layer(rng, args.layer_count)
    for s in layer:
        assert len(s.tokens) == len(s.labels)
    combined = base + layer
    rng.shuffle(combined)
    lens = sorted(len(s.tokens) for s in layer)
    print(f"v24 base: {len(base)} | new layer: {len(layer)} (len median {lens[len(lens) // 2]}, max {lens[-1]}) | total: {len(combined)}")
    before, after = entity_token_counts(base), entity_token_counts(combined)
    for tag in sorted(set(before) | set(after)):
        print(f"  {tag:8s} {before.get(tag, 0):6d} -> {after.get(tag, 0):6d}")
    write_iob2(combined, args.out)
    train_pool, test_holdout = split_by_skeleton(combined, args.test_ratio, args.split_seed)
    assert not ({skeleton(s) for s in train_pool} & {skeleton(s) for s in test_holdout}), "skeleton leak"
    write_iob2(train_pool, args.train_pool_out)
    write_iob2(test_holdout, args.test_holdout_out)
    print(f"wrote {args.out}\ntrain pool {len(train_pool)} -> {args.train_pool_out}\ntest holdout {len(test_holdout)} -> {args.test_holdout_out}")


if __name__ == "__main__":
    main()
