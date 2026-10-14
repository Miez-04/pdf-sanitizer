"""
data/build_v23_corpus.py

Adds a LETTER / DOCUMENT-STYLE layer on top of the v22 corpus, aimed at the
failures seen on real Malaysian letters (measured on a 10-letter test
document with the v22 / v23_ns models):

  1. SPAN EDGES. The model finds the core of a name or address but stops
     early: a name ends right after "Bin"/"Binti" (the family name stays
     visible), the given name before it is dropped, an address loses its
     leading "No." / "Lot" / "PT209," / "Suite 13-02," or its tail
     "Darul Ehsan". The layer gives many examples where the span runs
     exactly from the first to the last token.
  2. CONTEXTS. Names appeared in places the training data rarely had:
     "Name (MyKad: 980412-03-5119)" header lines, "Dear Puan Siti,"
     salutations, recipient blocks with a job title, signatures after
     "Yours sincerely,", subject lines, and mentions inside sentences.
  3. TOKEN SHAPE. PyMuPDF keeps punctuation attached to words
     ("PAYONG," / "Kelantan." / "5119)"). The old generators mostly emit
     punctuation as separate tokens ("PAYONG" ","), so training and
     inference tokens differ. This layer writes attached punctuation, and
     --attach-base-ratio converts a share of the EXISTING sentences too.

Conventions in this layer
  * honorifics (Encik, Puan, Dr, Mr., Datuk ...) are labelled O; only the
    name itself is PERSON;
  * every name, address and number comes from the project's own generators
    (data/registries.py); no sentence or name is copied from any test
    document.

Torch-free, builds from the v22 file:

    python -m data.build_v23_corpus --v22 data/processed/semi_synthetic_v22_ns3k.txt ^
        --out data/processed/semi_synthetic_v23_ns3k.txt ^
        --train-pool-out data/processed/train_pool_v23_ns3k.txt ^
        --test-holdout-out data/processed/test_holdout_v23_ns3k.txt
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
from data.registries import (
    BUILDING_PREFIXES,
    _tokenize_address,
    generate_address,
    generate_nric,
    generate_person,
    generate_phone,
    load_postcode_registry,
    vary_state_in_address,
)

# ---------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------

_PUNCT_ONLY = re.compile(r"^[,.;:!?)]+$")
_CONNECTORS = {"bin", "binti", "bte", "bt", "b.", "a/l", "a/p", "d/o", "s/o", "anak", "ak", "@"}
_HONORIFIC_FIRST = {"encik", "en.", "tuan", "datuk", "dato'", "dato", "dr", "dr.", "haji", "ir", "cik",
                    "puan", "datin", "hajjah", "prof", "prof.", "mr", "mr.", "mrs", "mrs.", "ms", "ms."}

HONORIFICS = ["Encik", "En.", "Puan", "Cik", "Tuan", "Mr.", "Mrs.", "Ms.", "Dr.", "Datuk", "Datin",
              "Prof.", "Haji", "Hajjah", "Dato'", "Tan Sri", "Puan Sri", "Datuk Seri"]

ROLES = [
    "Chief Executive Officer", "HR Manager", "Branch Manager", "Operations Manager",
    "Marketing Manager", "Finance Manager", "Head of Product & Design", "Pengarah Urusan",
    "Pengurus Besar", "Guru Besar", "Setiausaha", "Pengerusi Persatuan Penduduk",
    "Customer Service Officer", "Administrative Executive", "Senior Engineer", "Project Manager",
    "Pegawai Tadbir", "Penolong Pengurus", "Director of Operations", "Head of Department",
]

# invented companies and generic institutions (deliberately NOT the ones in any test document)
ORGS = [
    "Bintang Utama Sdn Bhd", "Seri Maju Resources Berhad", "Asia Pacific Logistics Sdn Bhd",
    "Harmoni Prima Holdings Berhad", "TechNova Solutions Sdn Bhd", "Kencana Jaya Enterprise",
    "Permata Sakti Trading", "Lestari Bina Sdn Bhd", "Cahaya Timur Industries Sdn Bhd",
    "Global Reach Consulting Sdn Bhd", "Universiti Putra Malaysia", "Lembaga Hasil Dalam Negeri",
    "Jabatan Pendaftaran Negara", "Hospital Sultanah Aminah", "Pos Malaysia Berhad",
    "Majlis Perbandaran Seberang Perai", "Kementerian Pendidikan Malaysia", "Politeknik Ungku Omar",
    "Syarikat Air Selangor", "Klinik Kesihatan Bandar Baru",
]

SUBJECTS_EN = [
    "NOTICE OF RESIGNATION", "OFFER OF EMPLOYMENT", "REQUEST FOR LEAVE", "APPLICATION FOR TRANSFER",
    "CONFIRMATION OF EMPLOYMENT", "COMPLAINT REGARDING SERVICE", "REQUEST FOR STATEMENT OF ACCOUNT",
    "NOTICE OF INTENT TO VACATE", "APPLICATION FOR INDUSTRIAL TRAINING", "LETTER OF AUTHORISATION",
]
SUBJECTS_MS = [
    "PERMOHONAN CUTI REHAT", "SURAT LETAK JAWATAN", "PERMOHONAN PERTUKARAN", "ADUAN MENGENAI PERKHIDMATAN",
    "PERITAHUAN KETIDAKHADIRAN KE SEKOLAH", "PERMOHONAN PENGELUARAN SIMPANAN", "SURAT KUASA",
]

MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
             "October", "November", "December"]
MONTHS_MS = ["Januari", "Februari", "Mac", "April", "Mei", "Jun", "Julai", "Ogos", "September",
             "Oktober", "November", "Disember"]

ID_LABELS = ["(MyKad:", "(NRIC:", "(No. K/P:", "(IC No:", "(No. KP:", "(MyKad No:", "(I/C:"]
PHONE_LABELS = ["", "Tel:", "H/P:", "Mobile:", "Tel :", "No. Tel:", "Hubungi:"]
CLOSERS_EN = ["Yours sincerely,", "Yours faithfully,", "Best regards,", "Sincerely,", "Thank you."]
CLOSERS_MS = ["Yang benar,", "Yang menjalankan amanah,", "Sekian, terima kasih.", "Saya yang menurut perintah,"]

# sentences that mention people / numbers / addresses inside running text.
# {P} full name  {PF} given-name part  {NRIC} {PH} {ADDR} entities
# {HON} {ORG} {ROLE} {DATE} {ACC} {REG} plain (label O) fillers
BODY_EN = [
    "I am writing to authorize my spouse, {P} (MyKad: {NRIC}), to collect the documents on my behalf .",
    "This is to certify that {HON} {P} (MyKad: {NRIC}), a student from {ORG}, has completed the programme .",
    "Please accept this letter as formal notification that I am resigning from my position as {ROLE} .",
    "Kindly send the reply to {ADDR} or call {PH} for any clarification .",
    "The undersigned, {P}, hereby confirms the details stated above are true and correct .",
    "{HON} {PF} will be in charge of the project and can be reached at {PH} .",
    "Enclosed are my statements for Account No: {ACC} and the supporting documents for your assessment .",
    "We have received your application dated {DATE} and the officer, {HON} {PF}, will contact you shortly .",
    "I would like to request a copy of the agreement registered under {REG} at {ADDR} .",
    "The meeting will be chaired by {HON} {P}, {ROLE}, at our office in {ADDR} .",
]
BODY_MS = [
    "Saya ibu kepada murid {P} dari Kelas 4 Cemerlang ingin memaklumkan bahawa anak saya tidak dapat hadir .",
    "Dengan ini saya, {P} (No. K/P: {NRIC}), memohon pengeluaran daripada akaun saya .",
    "Surat ini adalah untuk mengesahkan bahawa {HON} {P} telah menyelesaikan latihan industri di {ORG} .",
    "Sila hubungi {HON} {PF} di talian {PH} untuk maklumat lanjut .",
    "Surat kuasa ini diberikan kepada {P} (No. K/P: {NRIC}) untuk mengambil dokumen bagi pihak saya .",
    "Alamat surat-menyurat saya ialah {ADDR} dan nombor telefon saya {PH} .",
    "Permohonan daripada {P} telah diterima pada {DATE} dan sedang diproses oleh {HON} {PF} .",
]

TEAM_TEMPLATES = [
    "{P} {ROLE}", "{P}, {ROLE}", "{ROLE}: {P}", "{ROLE} - {P}", "{HON} {P}, {ROLE}",
    "{P} ({ROLE})", "1. {P}, {ROLE}", "Prepared by: {P}", "Contact person: {P} ({PH})",
]


def _o(text: str) -> tuple[list[str], list[str]]:
    toks = text.split()
    return toks, ["O"] * len(toks)


def _entity(kind: str, tokens: list[str], trailing: str = "") -> tuple[list[str], list[str]]:
    toks = list(tokens)
    if trailing:
        toks[-1] = toks[-1] + trailing  # punctuation stays attached, like PyMuPDF words
    return toks, [f"B-{kind}"] + [f"I-{kind}"] * (len(toks) - 1)


# ---------------------------------------------------------------------
# Entity generators with the conventions of this layer
# ---------------------------------------------------------------------

def attach_punctuation_tokens(tokens: list[str]) -> list[str]:
    """['PAYONG', ',', 'KOTA'] -> ['PAYONG,', 'KOTA']. Punctuation-only tokens
    are glued to the previous token; a lone '(' is glued to the next one."""
    out: list[str] = []
    pending_open = ""
    for t in tokens:
        if _PUNCT_ONLY.match(t) and out:
            out[-1] += t
        elif t == "(":
            pending_open = "("
        else:
            out.append(pending_open + t)
            pending_open = ""
    return out


def _strip_honorific(tokens: list[str]) -> list[str]:
    toks = list(tokens)
    while len(toks) > 1:
        first = toks[0].lower()
        if first == "tan" and len(toks) > 2 and toks[1].lower() == "sri":
            toks = toks[2:]
        elif first in _HONORIFIC_FIRST or first in {"sri", "seri"}:
            toks = toks[1:]
        else:
            break
    return toks


def person_tokens(rng: random.Random) -> list[str]:
    toks, _ = generate_person(rng)
    toks = _strip_honorific(toks)
    if rng.random() < 0.5:  # connectors capitalised like real letters: Bin, Binti, A/L, A/P
        toks = [t.capitalize() if t.lower() in {"bin", "binti", "bte", "anak"} else
                (t.upper() if t.lower() in {"a/l", "a/p"} else t) for t in toks]
    if rng.random() < 0.3:
        toks = [t.upper() for t in toks]
    return toks


def given_name_part(rng: random.Random, tokens: list[str]) -> list[str]:
    """The part of a name used in a salutation: the given names before a
    Bin/Binti/A-L connector (up to 3 words); otherwise the first word."""
    for i, t in enumerate(tokens):
        if t.lower() in _CONNECTORS and i >= 1:
            return tokens[: min(i, rng.randint(1, 3))]
    return tokens[:1]


def address_tokens(rng: random.Random) -> list[str]:
    if rng.random() < 0.35:  # building-style address with explicit lead-in and code
        reg = load_postcode_registry()
        prefix = rng.choice(BUILDING_PREFIXES)
        code = rng.choice([str(rng.randint(1, 300)), f"{rng.randint(1, 30)}-{rng.randint(1, 40):02d}",
                           f"B-{rng.randint(1, 20)}-{rng.randint(1, 20):02d}", f"{rng.randint(1, 4)}{rng.choice('ABCD')}"])
        building = rng.choice(["", "", "Menara Bumi", "Wisma Maju", "Pangsapuri Seri Jaya", "Kondominium Harmoni",
                               "Plaza Damai", "Mercu 3", "Tower 1"])
        if reg:
            e = rng.choice(reg)
            text = f"{prefix} {code}, {building + ', ' if building else ''}{e['place']}, {e['code']} {e['city']}, {e['state']}"
            toks = vary_state_in_address(_tokenize_address(text), rng)
        else:
            toks, _ = generate_address(rng)
    else:
        toks, _ = generate_address(rng)
    if rng.random() < 0.8:
        toks = attach_punctuation_tokens(toks)
    if rng.random() < 0.25:
        toks = [t.upper() for t in toks]
    return toks


def nric_tokens(rng: random.Random) -> list[str]:
    toks, _ = generate_nric(rng)
    return list(toks)


def phone_tokens(rng: random.Random) -> list[str]:
    toks, _ = generate_phone(rng)
    return list(toks)


def _date(rng: random.Random, ms: bool) -> str:
    d = rng.choice([1, 2, 3, 4, 5, 8, 10, 11, 12, 15, 20, 21, 22, 28])
    suf = "th" if 11 <= d % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
    month = rng.choice(MONTHS_MS if ms else MONTHS_EN)
    return f"{d}{suf} {month} {rng.randint(2024, 2027)}" if not ms else f"{d} {month} {rng.randint(2024, 2027)}"


def _account(rng: random.Random) -> str:
    return rng.choice([f"{rng.randint(10, 99)}-{rng.randint(100, 999)}-{rng.randint(100000, 999999)}-{rng.randint(1, 9)}",
                       f"{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}",
                       f"{rng.randint(10000000, 99999999)}-{rng.randint(1, 9)}"])


def _reg(rng: random.Random) -> str:
    return rng.choice([f"20{rng.randint(10, 24)}{rng.randint(10000000, 99999999) % 10**8:08d}-{rng.choice('ABKMTWX')}",
                       f"PPM-{rng.randint(1, 9):03d}-{rng.randint(1, 16):02d}-{rng.randint(10, 28)}{rng.randint(1, 12):02d}{rng.randint(2010, 2024)}"])


# ---------------------------------------------------------------------
# Template renderer:  {P} {PF} {PU} {NRIC} {PH} {ADDR}  = entities
#                     {HON} {ORG} {ROLE} {DATE} {ACC} {REG} {SUBJ} = plain O text
# ---------------------------------------------------------------------

_SLOT = re.compile(r"^(\(?)\{(\w+)\}([^\w{]*)$")


def render(template: str, rng: random.Random, ms: bool = False, ctx: dict | None = None) -> Sentence:
    ctx = ctx if ctx is not None else {}
    tokens: list[str] = []
    labels: list[str] = []
    for raw in template.split():
        m = _SLOT.match(raw)
        if not m:
            t, l = _o(raw)
        else:
            opener, kind, trailing = m.groups()
            if opener:
                tokens.append("(")
                labels.append("O")
            if kind in ("P", "PU"):
                # {P} always draws a NEW person (and remembers it); {PU} is the same
                # kind of slot written in ALL CAPS
                if kind == "P" or "person" not in ctx:
                    ctx["person"] = person_tokens(rng)
                toks = [x.upper() for x in ctx["person"]] if kind == "PU" else ctx["person"]
                t, l = _entity("PERSON", toks, trailing)
            elif kind in ("S", "SU"):
                # {S} = the SAME person as the earlier {P} of this letter (a signature
                # repeats the writer's name); a new person if there was none
                if "person" not in ctx:
                    ctx["person"] = person_tokens(rng)
                toks = [x.upper() for x in ctx["person"]] if kind == "SU" else ctx["person"]
                t, l = _entity("PERSON", toks, trailing)
            elif kind == "PF":
                base = ctx.get("person") or person_tokens(rng)
                t, l = _entity("PERSON", given_name_part(rng, base), trailing)
            elif kind == "NRIC":
                t, l = _entity("NRIC", nric_tokens(rng), trailing)
            elif kind == "PH":
                t, l = _entity("PHONE", phone_tokens(rng), trailing)
            elif kind == "ADDR":
                t, l = _entity("ADDRESS", address_tokens(rng), trailing)
            else:
                text = {
                    "HON": lambda: rng.choice(HONORIFICS), "ORG": lambda: rng.choice(ORGS),
                    "ROLE": lambda: rng.choice(ROLES), "DATE": lambda: _date(rng, ms),
                    "ACC": lambda: _account(rng), "REG": lambda: _reg(rng),
                    "SUBJ": lambda: rng.choice(SUBJECTS_MS if ms else SUBJECTS_EN),
                    "IDL": lambda: rng.choice(ID_LABELS), "PHL": lambda: rng.choice(PHONE_LABELS),
                }[kind]() + trailing
                t, l = _o(text)
                if opener:  # '(' already emitted as its own token; glue it instead
                    tokens.pop(); labels.pop()
                    t[0] = "(" + t[0]
        tokens += t
        labels += l
    # an opening "(" emitted as its own token for entities is glued to the next token
    fixed_t: list[str] = []
    fixed_l: list[str] = []
    carry = ""
    for tok, lab in zip(tokens, labels):
        if tok == "(" and lab == "O":
            carry = "("
            continue
        fixed_t.append(carry + tok)
        fixed_l.append(lab)
        carry = ""
    return Sentence(tokens=fixed_t, labels=fixed_l)


def _cat(parts: list[Sentence]) -> Sentence:
    t: list[str] = []
    l: list[str] = []
    for p in parts:
        t += p.tokens
        l += p.labels
    return Sentence(tokens=t, labels=l)


# ---------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------

def letter_block(rng: random.Random) -> Sentence:
    ms = rng.random() < 0.35
    ctx: dict = {}
    sender_options = [
        "{P} ({IDL} {NRIC}) {ADDR} | {PH} {DATE}",
        "{P} (MyKad: {NRIC}) {ADDR} {PH} {DATE}",
        "{P} {ADDR} | {PHL} {PH} {DATE}",
        "{PU} (MyKad: {NRIC} | EPF Acc: {ACC}) {ADDR} | {PH} {DATE}",
        "{ORG} (Co. Reg: {REG}) {ADDR} | Tel: {PH} {DATE}",
        "{P} ({IDL} {NRIC}) {ADDR} {DATE}",
        "{ADDR} {DATE}",
    ]
    recipient_options = [
        "{HON} {P}, {ROLE} {ORG}, {ADDR}",
        "{ROLE} {ORG}, {ADDR}",
        "{HON} {P} (Property Owner) {ADDR}",
        "{ORG}, {ADDR}",
        "{HON} {P}, {ROLE} {ORG}",
    ]
    parts = [render(rng.choice(sender_options), rng, ms, ctx)]
    ctx_recipient: dict = {}
    parts.append(render(rng.choice(recipient_options), rng, ms, ctx_recipient))
    if rng.random() < 0.7:
        parts.append(render("RE: {SUBJ}" if rng.random() < 0.6 else "{SUBJ}", rng, ms))
    sal = rng.choice(
        ["Dear {HON} {PF},", "Dear {HON} {PF},", "Dear Sir/Madam,", "Dear Hiring Manager,"]
        if not ms else ["Tuan/Puan,", "Kepada {HON} {PF},", "Dear {HON} {PF},"]
    )
    parts.append(render(sal, rng, ms, ctx_recipient if "{PF}" in sal else {}))
    for _ in range(rng.randint(1, 2)):
        parts.append(render(rng.choice(BODY_MS if ms else BODY_EN), rng, ms, ctx if rng.random() < 0.5 else {}))
    parts.append(render(rng.choice(CLOSERS_MS if ms else CLOSERS_EN), rng, ms))
    sig = rng.choice(["{S}", "{S}", "{SU}", "{S} {ROLE}, {ORG}", "{S} {ROLE}"])
    parts.append(render(sig, rng, ms, ctx))  # same person as the sender: the signature repeats the name
    return _cat(parts)


def mention_sentence(rng: random.Random) -> Sentence:
    ms = rng.random() < 0.4
    return render(rng.choice(BODY_MS if ms else BODY_EN), rng, ms)


def team_list(rng: random.Random) -> Sentence:
    rows = [render(rng.choice(TEAM_TEMPLATES), rng, False) for _ in range(rng.randint(2, 5))]
    return _cat(rows)


def id_table(rng: random.Random) -> Sentence:
    ms = rng.random() < 0.5
    lines = ["Nama : {P} No. K/P : {NRIC} No. Tel : {PH}", "Name : {P} IC No : {NRIC} H/P : {PH}",
             "Nama: {PU} No. K/P: {NRIC}, {ADDR}", "Applicant : {P} ({IDL} {NRIC}) Mobile : {PH}"]
    return render(rng.choice(lines), rng, ms)


def negative_numbers(rng: random.Random) -> Sentence:
    """Number-like strings that are NOT a phone or an NRIC (account, EPF, MC,
    company registration): label O, so the model does not learn 'digits with
    hyphens = phone'."""
    lines = ["Account No: {ACC} .", "EPF Acc: {ACC} )", "Reference : MC No: {ACC} issued {DATE}",
             "Company Reg: {REG} (Co. Reg) .", "Invoice No: {ACC} dated {DATE} ."]
    return render(rng.choice(lines), rng, False)


FAMILIES = [
    (letter_block, 0.50), (mention_sentence, 0.20), (team_list, 0.12), (id_table, 0.10), (negative_numbers, 0.08),
]


def build_layer(rng: random.Random, count: int) -> list[Sentence]:
    funcs = [f for f, _ in FAMILIES]
    weights = [w for _, w in FAMILIES]
    out = []
    for _ in range(count):
        out.append(rng.choices(funcs, weights)[0](rng))
    return out


# ---------------------------------------------------------------------
# Attach punctuation in EXISTING sentences
# ---------------------------------------------------------------------

def attach_punctuation_sentence(sent: Sentence) -> Sentence:
    """Glue standalone punctuation tokens (, . ; : ! ? ) ) to the previous
    token and keep the PREVIOUS token's label, as a PDF word would be
    labelled ('Kelantan.' stays inside the address). A punctuation token that
    starts an entity (label B-) is left alone."""
    toks: list[str] = []
    labs: list[str] = []
    for tok, lab in zip(sent.tokens, sent.labels):
        if toks and _PUNCT_ONLY.match(tok) and not lab.startswith("B-"):
            toks[-1] += tok
        else:
            toks.append(tok)
            labs.append(lab)
    return Sentence(tokens=toks, labels=labs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v22", default="data/processed/semi_synthetic_v22_ns3k.txt")
    ap.add_argument("--letter-count", type=int, default=1200,
                    help="number of new letter/document-style sequences. Moderate on purpose: v21 "
                         "regressed after a large narrow batch. On the v22_ns corpus, 2500 would make "
                         "the new layer about 60%% of all ADDRESS tokens, 1200 about 45%%.")
    ap.add_argument("--attach-base-ratio", type=float, default=0.5,
                    help="share of EXISTING sentences whose standalone punctuation is glued to words")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=123)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--out", default="data/processed/semi_synthetic_v23_ns3k.txt")
    ap.add_argument("--train-pool-out", default="data/processed/train_pool_v23_ns3k.txt")
    ap.add_argument("--test-holdout-out", default="data/processed/test_holdout_v23_ns3k.txt")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = parse_iob2_file(args.v22)
    print(f"v22 base: {len(base)} sentences")

    attached = 0
    base_out: list[Sentence] = []
    for s in base:
        if rng.random() < args.attach_base_ratio:
            s = attach_punctuation_sentence(s)
            attached += 1
        base_out.append(s)
    print(f"base sentences with punctuation attached: {attached}")

    layer = build_layer(rng, args.letter_count)
    for s in layer:
        assert len(s.tokens) == len(s.labels)
    print(f"letter/document-style sequences: {len(layer)}")

    combined = base_out + layer
    rng.shuffle(combined)

    before, after = entity_token_counts(base), entity_token_counts(combined)
    print("\nEntity token counts, v22 -> v23:")
    for tag in sorted(set(before) | set(after)):
        print(f"  {tag:8s} {before.get(tag, 0):6d} -> {after.get(tag, 0):6d}")

    write_iob2(combined, args.out)
    print(f"\nWrote {len(combined)} sentences -> {args.out}")
    train_pool, test_holdout = split_by_skeleton(combined, args.test_ratio, args.split_seed)
    assert not ({skeleton(s) for s in train_pool} & {skeleton(s) for s in test_holdout}), "skeleton leak"
    write_iob2(train_pool, args.train_pool_out)
    write_iob2(test_holdout, args.test_holdout_out)
    print(f"train pool:   {len(train_pool)} -> {args.train_pool_out}")
    print(f"test holdout: {len(test_holdout)} -> {args.test_holdout_out}")


if __name__ == "__main__":
    main()
