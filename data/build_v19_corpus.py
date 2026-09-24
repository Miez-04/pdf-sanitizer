"""
data/build_v19_corpus.py

Adds two CONFIRMED training-data gaps on top of v18, both traced to
real evidence the user supplied directly rather than guessed at:

  1. BARE NAME-LIST CONTEXT. Real-PDF testing (a class list PDF, "BIL
     | NAMA" table) showed the model missing most names when they
     appear as a bare list/table entry with no surrounding sentence
     or label. Checked entity_mutation.py's CONNECTOR_TEMPLATES: every
     existing PERSON training example wraps the name in SOME labeled
     or narrative context ("Nama: X", "X is applying for...", etc.) —
     a name sitting completely alone as a list/table row is a shape
     the model has never been trained on. Fixed here by generating
     numbered-list and table-row sentences directly from the user's
     own real name list (Example_name.txt — 91 real Malay/Chinese/
     Indian names spanning three naming systems).

  2. LABEL/PATTERN DIVERSITY. The user supplied a formatting
     catalogue (POSSIBLE_FORMATTING.pdf) documenting label and
     structural variants NOT all represented in the existing
     CONNECTOR_TEMPLATES — bilingual labels ("Nama Pesakit / Patient
     Name:"), implicit no-label headers, inline narrative mentions,
     honorifics (Ir., Dr., Puan, Encik), name-format variants (Bin/b./
     B, Binti/bt/bte, A/L/a/l/s/o, Anak/ak), and parenthetical aliases
     ("Wong Mei Ling (Karen Wong)") for PERSON; and property-type +
     label diversity (landed/high-rise/commercial/industrial/village/
     East Malaysian/P.O. Box, crossed with Malay/English/bilingual/
     multi-line-no-label/inline labels) for ADDRESS.

Combines with v18 (not from scratch) — same reasoning as v18 combining
with v17: this is additive coverage for a specific found gap, not a
full regeneration.

Torch-free, same reasoning as build_v18_corpus.py.

Usage:
    python -m data.build_v19_corpus
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

# Reuse v18's parsing/writing/splitting machinery rather than
# duplicating it a second time.
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
# 1. Real names, from the user's Example_name.txt (91 names, 3 systems)
# =======================================================================

MALAY_NAMES = [
    "ADAM SYAFIQ BIN MOHD FAIZAL", "AFIQ ADIB BIN MUS MULIADI",
    "AHMAD ERFAN QALIFF BIN AZRUL FAIDZ", "BALQIS AN NUR BINTI MOHD FADLEE",
    "ELLYSHA DAMIA' BINTI MUHAMMAD HELMIE",
    "FATIMAH ZAHRAH BINTI MOHD HABIBULLAH", "HANNANI ZUHAIRAH BINTI ZAZALY",
    "MUHAMMAD ADHA ZIKRY BIN KAMAL ARIFIN", "MUHAMMAD AKIF BIN AHMAD SYAHIR",
    "MUHAMMAD AMMAR MUHKRIZ BIN MUHAMMAD FAZLI",
    "MUHAMMAD AQIL HAFIY BIN ABDULLAH HAILME",
    "MUHAMMAD ASYRAF BIN MOHD NAZIF", "MUHAMMAD FARIS RAIHAN BIN CHIK",
    "MUHAMMAD IRFAN BIN ARZNAN", "MUHAMMAD RAMADAN BIN RAISED",
    "MUHAMMAD SHAHRIN HAAZIM BIN SHAHRIZAL",
    "MUHAMMAD SYAFIQ BIN MOHD IRWAN", "MUHAMMAD SYAFIQ ZAQWAN BIN SHARIZAL",
    "MUHAMMAD UMAR RAEYF BIN MOHD SHUKRI", "NAYLI ATHIRAH BINTI MOHD FIKRI",
    "NUR ALIAH BATRISYIA BINTI AZMI", "NUR FARISYA KHAYRIN BINTI MOHD FAUZI",
    "NUR FATIN DARWISYAH BINTI MOHD FAIRUS",
    "NUR HAFIZAHTUL DIYANAH BINTI HAMDAN", "NURAIN BINTI AZHAR",
    "NURIN AMANINA BINTI ZAKI", "NURUL IMAN BINTI MOHD IDRUS",
    "NURZAHIDAH BINTI MUHAMAD ZAILAN", "PUTERI ADANI DAMIA BINTI AMIR ERWAN",
    "ZAKIR ZAKWAN BIN NOOR AFENDI",
]

CHINESE_NAMES = [
    "ANG SHENG LOONG", "CHIA XIN YING", "HEW ZHI QING", "KO JUN LONG",
    "KOK BOH LONG", "KOK JUN JIE", "LAM JING XUAN", "LEE CHI BIN",
    "LEE MUN LYE", "MARCUS LEE QIAO JIE", "TAN LI LONG", "WONG CAI XIAN",
    "WONG CAI ZHEN", "WONG WEN DE", "AUDREY YONG", "CHAI JIA YEE",
    "CHAI ZHE HAO", "CHIN XIAO XUAN", "CHONG JIA HE", "CHONG KE YAO",
    "CHONG KHAI EN", "CHONG YAN QING", "CHONG YING TUNG", "GOH JUN LIN",
    "KAREN TOH", "KHOO SI KEE", "LAHE KAI WOON", "LEE MIN YOUNG",
    "LEE QI MING", "LEN TECK KIONG", "LIM YI QI", "LING HUA KHANG",
    "LOO HOW JUN", "LOW YU HAO", "MAVIS TEO SYE DONN", "MAX TEE KAI YOON",
    "NG YU YAN", "RANDERS CHONG HIN WAI", "SIOW CONNIE", "TAI WEI LONG",
    "TAN HUI YU", "TAN YUKI", "TEY JIE NIN", "TIANG WEI QUAN",
    "WONG YUE QING", "YAP ZI XUAN",
]

INDIAN_NAMES = [
    "DASSHWINN MAGENDRAN", "ELEENA A/P MUNIYANDY",
    "KHEERTHIGA A/P THEVAKKUMAR", "KHUBEHRAN A/L MUNIANDYSAMY",
    "LAXSHANAA DEVI A/P PARAMESWARAN", "MANIESHA A/P KANAPATHY",
    "NAKULAN A/L ILAVARASAN", "PERINIYAAL A/P VUAIAN",
    "SHANMUGAPRIYA A/P DEVAN", "THEVAR HAKILESS A/P THEVE AKILAN",
    "UGENESWARAN A/L SHASHI KUMAR", "VINUSIYAA NAIDU A/P KARTIGESU NAIDU",
    "VISHAL A/L PIRABU", "YEAISHVANNRAJ A/L SAKTIRAJNANTHAN",
    "PARAMESWAR A/L SUPERAMANIAM", "EIMAIYARASI A/P RETHINASAMY",
    "GANA PRIYA A/P DEVAN", "JAMESHINA ANGEL A/P JAMES",
    "KAVEA A/P THEVAKKUMAR", "KOGILA A/P MURLEE",
    "LATHEESHAN A/L RAJINI", "MELINDA A/P ILAVARASAN",
    "RENUKA SHREE A/P MAHENDRAN", "SHASHNEESHA A/P ALAGAVAN",
    "THANARUBAN A/L MURUGAN", "KIRTHIKA SRI A/P GANESAN",
    "SIVEGANES A/L KUMARAVELL", "YASHWINI A/P TANIMALA",
    "YOGAVARSHINI A/P PUGANESWARAN", "DEEPADARSHINIY A/P SHASHI KUMAR",
    "DINAGARAN A/L MARUTHAIAN", "HAREESH VANAR A/L KOGILAVANAR",
    "KAVINASH A/L SURESH KUMAR", "KRISHNANRAJ A/L NAGARAJAN",
    "NITESH A/L KALIDASS", "NITTISH A/L JAISELAN",
    "PADMAN KUGHAAN A/L PARAMESWARAN",
    "PIRASHAANTHAN NAIDU A/L KARTIGESU NAIDU",
    "RESHMENAVI A/P RETHINASAMY", "STEPHEN A/L KESAVAN",
    "TARANESHVARAN A/L VISNU", "VARATESVARAN A/L MARAN",
    "VIMMAL MUTHUKUMARAN", "VISHAALIENA NAIR A/P PRABAKARAN",
    "VISHAL MUTHUKUMARAN",
]

ALL_NAMES = MALAY_NAMES + CHINESE_NAMES + INDIAN_NAMES


def _name_sentence(name_text: str, prefix_tokens: list[str] | None = None) -> Sentence:
    """One name -> one Sentence, entirely B-PERSON/I-PERSON, with
    optional leading O-tagged furniture (a bare list number, a table
    cell number) — exactly the shape a class-list/attendance-sheet row
    takes, which is the confirmed failing case."""
    tokens = list(prefix_tokens or [])
    labels = ["O"] * len(tokens)
    name_tokens = name_text.split(" ")
    for i, tok in enumerate(name_tokens):
        tokens.append(tok)
        labels.append("B-PERSON" if i == 0 else "I-PERSON")
    return Sentence(tokens=tokens, labels=labels)


def build_bare_name_list_sentences(rng: random.Random) -> list[Sentence]:
    """Each real name, in the three concrete shapes the class-list PDF
    actually used: bare standalone line, numbered list ("12. NAME"),
    and table row ("12  NAME" — BIL/NAMA column layout, space-separated
    since that's how the table's cell text extracts)."""
    out = []
    for i, name in enumerate(ALL_NAMES, start=1):
        out.append(_name_sentence(name))                              # bare line
        out.append(_name_sentence(name, [f"{i}."]))                    # numbered list
        out.append(_name_sentence(name, [str(i)]))                     # table row (BIL column)
    # A few multi-name table pages: several list rows concatenated into
    # one training "sentence" (== one page's worth of table), so the
    # model also sees consecutive B-PERSON spans back to back with no
    # separating sentence, matching a real multi-row table extraction.
    shuffled = list(ALL_NAMES)
    rng.shuffle(shuffled)
    for start in range(0, len(shuffled) - 4, 5):
        chunk = shuffled[start:start + 5]
        tokens, labels = [], []
        for i, name in enumerate(chunk, start=1):
            tokens.append(str(i)); labels.append("O")
            for j, tok in enumerate(name.split(" ")):
                tokens.append(tok)
                labels.append("B-PERSON" if j == 0 else "I-PERSON")
        out.append(Sentence(tokens=tokens, labels=labels))
    return out


# =======================================================================
# 2. PERSON label/pattern diversity — from POSSIBLE_FORMATTING.pdf
# =======================================================================

PERSON_LABEL_TEMPLATES_EN = [
    "Name: {P}", "Full Name: {P}", "Applicant Name: {P}", "Insured Name: {P}",
    "Received From: {P}", "Issued By: {P}", "Referring Doctor: {P}",
    "Employer Name: {P}",  # negative-adjacent context on purpose: label
                            # itself is fine as O either way, entity is still PERSON
]
PERSON_LABEL_TEMPLATES_MS = [
    "Nama Penuh: {P}", "Nama Pengguna: {P}", "Nama Pesakit: {P}",
    "Nama Tuan Rumah: {P}", "Nama Penyewa: {P}", "Nama Waris: {P}",
    "Nama Saksi: {P}",
]
PERSON_LABEL_TEMPLATES_BILINGUAL = [
    "Nama Pesakit / Patient Name: {P}", "Nama Pengguna / Customer Details: {P}",
    "Nama / Name: {P}",
]
PERSON_IMPLICIT_TEMPLATES = [
    "{P}", "{P}\nResume", "Kepada: {P}", "Yang benar,\n{P}",
]
PERSON_INLINE_TEMPLATES = [
    "issued to candidate {P} on the stated date .",
    "Received from {P} the sum of one thousand ringgit .",
    "This is to certify that {P} has completed the course .",
    "Policyholder Signature: {P}",
    "Tandatangan Penyewa: {P}",
]

HONORIFIC_PREFIXES = ["Ir.", "Dr.", "Puan", "Encik", "Assoc. Prof. Dr."]

# name -> alternate structural forms, applied to a handful of base
# Malay/Indian names from the pool (Bin/Binti/A-L/A-P abbreviation
# variants explicitly catalogued as separate real-world forms).
def _structural_variants(name: str) -> list[str]:
    variants = [name]
    lower_particles = {
        " BIN ": [" bin ", " b. ", " B "],
        " BINTI ": [" binti ", " bt ", " bte "],
        " A/L ": [" a/l ", " s/o "],
        " A/P ": [" a/p ", " d/o "],
    }
    for particle, alts in lower_particles.items():
        if particle in name:
            for alt in alts:
                variants.append(name.replace(particle, alt))
    return variants


def build_person_pattern_sentences(rng: random.Random) -> list[Sentence]:
    out = []
    sample_names = rng.sample(ALL_NAMES, min(40, len(ALL_NAMES)))

    for name in sample_names:
        for variant in _structural_variants(name)[:2]:  # cap: don't over-weight one base name
            template = rng.choice(
                PERSON_LABEL_TEMPLATES_EN + PERSON_LABEL_TEMPLATES_MS
                + PERSON_LABEL_TEMPLATES_BILINGUAL
            )
            text = template.replace("{P}", variant)
            out.append(_tag_template_as_person(text, variant))

    # Implicit header / inline narrative — no label at all
    for name in rng.sample(ALL_NAMES, min(20, len(ALL_NAMES))):
        template = rng.choice(PERSON_IMPLICIT_TEMPLATES + PERSON_INLINE_TEMPLATES)
        text = template.replace("{P}", name)
        out.append(_tag_template_as_person(text, name))

    # Honorifics
    for name in rng.sample(MALAY_NAMES, min(10, len(MALAY_NAMES))):
        prefix = rng.choice(HONORIFIC_PREFIXES)
        full = f"{prefix} {name}"
        out.append(_name_sentence(full))

    # Parenthetical alias: "Wong Mei Ling (Karen Wong)" — both spans
    # are real PERSON entities (a person's registered name and their
    # alias/preferred name), tagged as two separate B-PERSON spans.
    for name in rng.sample(CHINESE_NAMES, min(10, len(CHINESE_NAMES))):
        alias = " ".join(name.split(" ")[:2])  # crude but real-shaped: "Karen Wong"-style short form
        tokens = name.split(" ") + ["("] + alias.split(" ") + [")"]
        labels = (
            ["B-PERSON"] + ["I-PERSON"] * (len(name.split(" ")) - 1)
            + ["O"]
            + ["B-PERSON"] + ["I-PERSON"] * (len(alias.split(" ")) - 1)
            + ["O"]
        )
        out.append(Sentence(tokens=tokens, labels=labels))

    return out


def _tag_template_as_person(text: str, name: str) -> Sentence:
    """Tags every token of `name` as B-/I-PERSON wherever it appears in
    `text`, everything else O. Simple whole-phrase match since these
    are controlled templates, not free text."""
    tokens = text.replace("\n", " \n ").split(" ")
    name_tokens = name.split(" ")
    labels = ["O"] * len(tokens)
    n = len(name_tokens)
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n] == name_tokens:
            labels[i] = "B-PERSON"
            for k in range(1, n):
                labels[i + k] = "I-PERSON"
            break
    tokens = [t for t in tokens if t != "\n"]
    labels = [l for t, l in zip(text.replace("\n", " \n ").split(" "), labels) if t != "\n"]
    return Sentence(tokens=tokens, labels=labels)


# =======================================================================
# 3. ADDRESS property-type + label diversity — from POSSIBLE_FORMATTING.pdf
# =======================================================================

ADDRESS_VALUE_POOL = [
    "No. 12, Jalan SS 2/45, 47300 Petaling Jaya, Selangor",
    "No. 45, Jalan Taman Selasih 3, 09000 Kulim, Kedah",
    "Unit A-22-05, Residency Condominium, Jalan Bukit Bintang, 55100 Kuala Lumpur",
    "Block B-11-08, Pangsapuri Suria Jaya, Jalan Kinrara 4, 47180 Puchong, Selangor",
    "Unit 18-3A, Block B, Menara Service Residence, Jalan Kerinchi, 59200 Kuala Lumpur",
    "Suite 8-2, West Wing, No. 15, Lorong Perusahaan 2, 13600 Prai, Pulau Pinang",
    "Level 12, Tower B, Menara PJX, Jalan Timur, 46050 Petaling Jaya, Selangor",
    "No. 24-1, Jalan Telawi 3, Bangsar Baru, 59100 Kuala Lumpur",
    "Kawasan Perindustrian Bayan Lepas, Phase 4, 11900 Penang",
    "Lot 45, Kawasan Perindustrian Pasir Gudang, 81700 Pasir Gudang, Johor",
    "No. 89, Jalan Kampung Pasir Puteh, 31650 Ipoh, Perak",
    "Kampung Laut, Mukim Kuala Besut, 22200 Besut, Terengganu",
    "Kampung Long Lama, Baram, 98050 Miri, Sarawak",
    "Sublot 14, Lot 405, Block 11, Muara Tuang Land District, 94300 Kota Samarahan, Sarawak",
    "Sublot 45, Miri Waterfront Commercial Centre, Jalan Bendahara, 98000 Miri, Sarawak",
    "P.O. Box 10294, Pejabat Pos Besar, 88803 Kota Kinabalu, Sabah",
    "Lot 12-A, Jalan Ampang Utama 1/1, 68000 Ampang, Selangor",
    "No. 45, Jalan Kelawai, 10250 Georgetown, Pulau Pinang",
    "No. 12, Jalan Taman Selasih 3, 09000 Kulim, Kedah",
    "Lot 892, Kampung Pasir Puteh, 31650 Ipoh, Perak",
    "No. 18, Jalan Pandan Indah 4/2, 55100 Kuala Lumpur",
]

ADDRESS_LABEL_TEMPLATES_MS = [
    "Alamat Kediaman: {A}", "Alamat Premis: {A}", "Alamat Rumah: {A}",
    "Alamat Bekalan: {A}",
]
ADDRESS_LABEL_TEMPLATES_EN = [
    "Residential Address: {A}", "Office Address: {A}", "Billing Address: {A}",
    "Risk Address: {A}",
]
ADDRESS_NOLABEL_TEMPLATES = ["{A}"]
ADDRESS_INLINE_TEMPLATES = ["premises located at {A} ."]


def build_address_pattern_sentences(rng: random.Random) -> list[Sentence]:
    out = []
    for addr in ADDRESS_VALUE_POOL:
        template = rng.choice(
            ADDRESS_LABEL_TEMPLATES_MS + ADDRESS_LABEL_TEMPLATES_EN
            + ADDRESS_NOLABEL_TEMPLATES + ADDRESS_INLINE_TEMPLATES
        )
        text = template.replace("{A}", addr)
        sent = _tag_template_as_address(text, addr)
        sent, _ = normalize_address_lead_words(sent)  # keep v18's boundary fix applied here too
        out.append(sent)
    return out


def _tag_template_as_address(text: str, addr: str) -> Sentence:
    tokens = text.split(" ")
    addr_tokens = addr.split(" ")
    labels = ["O"] * len(tokens)
    n = len(addr_tokens)
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n] == addr_tokens:
            labels[i] = "B-ADDRESS"
            for k in range(1, n):
                labels[i + k] = "I-ADDRESS"
            break
    return Sentence(tokens=tokens, labels=labels)


# =======================================================================
# Combine with v18
# =======================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v18", default="data/processed/semi_synthetic_v18.txt")
    parser.add_argument("--name-list-repeat", type=int, default=3,
                         help="How many times to repeat the bare-name-list "
                              "sentences — small pool (91 names), needs "
                              "real weight against v18's ~7000 sentences.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--out", default="data/processed/semi_synthetic_v19.txt")
    parser.add_argument("--train-pool-out", default="data/processed/train_pool_v19.txt")
    parser.add_argument("--test-holdout-out", default="data/processed/test_holdout_v19.txt")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    v18 = parse_iob2_file(args.v18)
    print(f"v18 base: {len(v18)} sentences")

    bare_list = build_bare_name_list_sentences(rng) * args.name_list_repeat
    print(f"bare name-list sentences: {len(bare_list)} "
          f"({len(ALL_NAMES)} real names x 3 shapes x {args.name_list_repeat}x repeat, "
          f"plus multi-row table chunks)")

    person_patterns = build_person_pattern_sentences(rng)
    print(f"PERSON label/pattern diversity sentences: {len(person_patterns)}")

    address_patterns = build_address_pattern_sentences(rng)
    print(f"ADDRESS label/pattern diversity sentences: {len(address_patterns)}")

    combined = v18 + bare_list + person_patterns + address_patterns
    rng.shuffle(combined)

    before = entity_token_counts(v18)
    after = entity_token_counts(combined)
    print("\nEntity token counts, v18 -> v19:")
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
