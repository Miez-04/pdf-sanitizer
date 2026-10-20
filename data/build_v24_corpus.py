"""
data/build_v24_corpus.py

Adds the patterns that v25 still missed on a real Malay guideline document
(measured on the v23 training pool and on that document):

  1. ADDRESS WITHOUT A POSTCODE. Only 44 of 4,180 training addresses (1.1%)
     had none, so "Jalan Ampang, Kuala Lumpur" or "Kota Bharu, Kelantan" was
     almost never seen. This layer writes street-led and "City, State" forms,
     many derived from the real company address pool with the postcode removed.
  2. LONG NAMES. Only 4% of training names had 6+ tokens and 0.7% had 7+, but
     real names run to "Wan Mohd. Zawawi bin Wan Abd Rahman" or
     "Huzaifah binti Mohd Zain @ Mad Zain". Spans were cut off or merged.
  3. NAMES IN A ROW. Credit lists print names one after another with no
     separator ("Latifah binti Md Yatim Rabiah binti Alias ..."); only 45
     training spans started right after another name. The layer teaches that
     a new B-PERSON starts each name.
  4. BRACKETED SIGNATURES such as "(HASANUDDIN BIN OTHMAN) Penolong Pendaftar"
     and "(u.p.: Puan Anis binti X)".
  5. HARD NEGATIVES for 1: a city or state that is NOT an address (publication
     place "Kuala Lumpur: Arkib Negara", campus "UTM Johor Bahru", institution
     "Dewan Bandaraya Kuala Lumpur", a title "Permaisuri Johor", a city in
     prose). Without them the model would learn "city name = address".

Independence: this builder imports nothing from regex_engine. The address heuristic (and its
places table) is an optional product feature; the Bi-LSTM-CRF learns these addresses from data.

Policy note: a bare "City, State" is a coarse location, not a street address.
It is labelled ADDRESS here because the project asked for it; set
--bare-pair-share 0 to leave those out.

Conventions are the same as data/build_v23_corpus.py (honorifics labelled O,
punctuation attached to words, every name/number from the project's own
generators). Torch-free. Builds on the v23 corpus:

    python -m data.build_v24_corpus --v23 data/processed/semi_synthetic_v23_ns3k.txt ^
        --out data/processed/semi_synthetic_v24_ns3k.txt ^
        --train-pool-out data/processed/train_pool_v24_ns3k.txt ^
        --test-holdout-out data/processed/test_holdout_v24_ns3k.txt
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
from data.build_v23_corpus import (
    CLOSERS_EN,
    CLOSERS_MS,
    HONORIFICS,
    ORGS,
    ROLES,
    _cat,
    _entity,
    _o,
    attach_punctuation_tokens,
    letter_block,
    mention_sentence,
    person_tokens,
    render,
)
from data.registries import (
    INDIAN_FATHER,
    INDIAN_FIRST,
    MALAY_FEMALE_FIRST,
    MALAY_MALE_FIRST,
    MALAY_PATRONYM,
    _tokenize_address,
    load_company_address_pool,
    load_postcode_registry,
    vary_state_in_address,
)

# ---------------------------------------------------------------------
# 1. Long names
# ---------------------------------------------------------------------

_LINEAGE = ["Wan", "Nik", "Tengku", "Syed", "Sharifah", "Raja", "Che", "Megat", "Nor", "Nur"]
_MOHD = ["Mohd", "Mohd.", "Md", "Muhammad", "Mohamad", "Mohamed"]
_ABD = ["Abd", "Abd.", "Abdul"]


def long_person_tokens(rng: random.Random) -> list[str]:
    """A name of 5 to 10 tokens: lineage prefixes, Mohd/Abd forms, compound given
    and father's names, and sometimes an '@ alias'."""
    for _ in range(20):
        style = rng.choice(["malay_m", "malay_m", "malay_f", "malay_f", "alias_m", "alias_f", "indian", "indian_alias"])
        if style.startswith("malay") or style.startswith("alias"):
            female = style.endswith("_f") or style == "malay_f"
            pool = MALAY_FEMALE_FIRST if female else MALAY_MALE_FIRST
            given: list[str] = []
            if rng.random() < 0.45:
                given.append(rng.choice(_LINEAGE))
            if not female and rng.random() < 0.55:
                given.append(rng.choice(_MOHD))
            given += [rng.choice(pool) for _ in range(rng.choice([1, 2, 2]))]
            father: list[str] = []
            if rng.random() < 0.4:
                father.append(rng.choice(_LINEAGE))
            if rng.random() < 0.4:
                father.append(rng.choice(_ABD))
            father += [rng.choice(MALAY_PATRONYM) for _ in range(rng.choice([1, 2]))]
            conn = ["Binti" if female else "Bin"]
            toks = given + conn + father
            if style.startswith("alias"):
                toks += ["@"] + [rng.choice(pool) for _ in range(rng.choice([1, 2]))]
        else:
            given = [rng.choice(INDIAN_FIRST) for _ in range(rng.choice([1, 2]))]
            toks = given + [rng.choice(["A/L", "A/P"])] + [rng.choice(INDIAN_FATHER) for _ in range(rng.choice([1, 2]))]
            if style == "indian_alias":
                toks += ["@", rng.choice(INDIAN_FIRST)]
        if len(toks) >= 5:
            break
    if rng.random() < 0.5:
        toks = [t.lower() if t in {"Bin", "Binti"} else t for t in toks]
    if rng.random() < 0.3:
        toks = [t.upper() for t in toks]
    return toks


def mixed_person_tokens(rng: random.Random, p_long: float = 0.35) -> list[str]:
    """A long name with probability p_long, otherwise the ordinary generator. Kept
    well below 1.0 so long names grow from ~4% to roughly 15% of all names, not to a
    majority (a model that sees mostly long names starts to run spans on)."""
    return long_person_tokens(rng) if rng.random() < p_long else person_tokens(rng)


def _mostly_long(rng: random.Random) -> list[str]:
    return mixed_person_tokens(rng, 0.6)


# ---------------------------------------------------------------------
# 2. Addresses without a postcode
# ---------------------------------------------------------------------

_ROAD_TYPES = ["Jalan", "Jalan", "Jalan", "Lorong", "Persiaran", "Lebuh", "Lebuhraya", "Jalan"]
_ROAD_NAMES = [
    "Ampang", "Mawar", "Kenanga", "Cempaka", "Tun Razak", "Bukit Bintang", "Pahlawan", "Sultan Ismail",
    "Tuanku Abdul Rahman", "Raja Laut", "Setia", "Bunga Raya", "Perdana", "Kenari", "Dato' Onn", "Masjid",
    "Sekolah", "Pasar", "Stesen", "Tengah", "Besar", "Baru", "Merdeka", "Harmoni", "Sentosa", "Damai",
    "Utama", "Indah", "Teratai", "Melati", "Seri Aman", "Mahkota", "Pantai", "Telok Kumbar", "Gombak",
]
_AREAS = ["Taman Indah", "Taman Seri Murni", "Kampung Baru", "Bandar Sunway", "Seksyen 7", "Taman Bahagia",
          "Kampung Pasir", "Bandar Utama", "Taman Melati", "Kampung Sungai Pinang"]
_BUILDINGS = ["", "", "", "Menara Bumi", "Wisma Maju", "Pangsapuri Seri Jaya", "Kondominium Harmoni", "Plaza Damai"]
_PREFIXES = ["No.", "No.", "Lot", "Unit", "Suite", "Tingkat", "PT", "Blok"]


def _street(rng: random.Random) -> str:
    code = rng.choice(["", "", f" {rng.randint(1, 20)}", f" {rng.randint(1, 9)}/{rng.randint(1, 30)}",
                       f" SS {rng.randint(1, 25)}/{rng.randint(1, 80)}"])
    return f"{rng.choice(_ROAD_TYPES)} {rng.choice(_ROAD_NAMES)}{code}"


def _from_pool_without_postcode(rng: random.Random) -> list[str] | None:
    pool = load_company_address_pool()
    if not pool:
        return None
    toks = [t for t in _tokenize_address(rng.choice(pool)) if not re.fullmatch(r"\d{5}", t)]
    return toks if len(toks) >= 4 else None


def address_no_postcode_tokens(rng: random.Random, bare_pair_share: float = 0.15) -> list[str]:
    reg = load_postcode_registry()
    roll = rng.random()
    toks: list[str] | None = None
    if roll < 0.40:
        toks = _from_pool_without_postcode(rng)
    if toks is None and reg:
        e = rng.choice(reg)
        city, state = e["city"], e["state"]
        if roll >= 1 - bare_pair_share:
            text = f"{city}, {state}"                                  # bare pair
        else:
            form = rng.randint(1, 5)
            code = rng.choice([str(rng.randint(1, 300)), f"{rng.randint(1, 30)}-{rng.randint(1, 40):02d}"])
            if form == 1:
                text = f"{_street(rng)}, {city}"
            elif form == 2:
                text = f"{_street(rng)}, {city}, {state}"
            elif form == 3:
                text = f"{rng.choice(_PREFIXES)} {code}, {_street(rng)}, {rng.choice(_AREAS)}, {city}, {state}"
            elif form == 4:
                b = rng.choice(_BUILDINGS)
                text = f"{rng.choice(_PREFIXES)} {code}, {b + ', ' if b else ''}{_street(rng)}, {city}"
            else:
                text = f"{rng.choice(_AREAS)}, {city}, {state}"
        toks = _tokenize_address(text)
    elif toks is None:
        toks = _tokenize_address(f"{_street(rng)}, Kota Bharu, Kelantan")
    toks = vary_state_in_address(toks, rng)
    if rng.random() < 0.8:
        toks = attach_punctuation_tokens(toks)
    if rng.random() < 0.25:
        toks = [t.upper() for t in toks]
    return toks


def _np_slot(share: float):
    return lambda rng: ("ADDRESS", address_no_postcode_tokens(rng, share))


# ---------------------------------------------------------------------
# 3. Hard negatives: city / state words that are NOT an address
# ---------------------------------------------------------------------

def _registry_cities() -> list[str]:
    """City names from the project's own postal registry (data/raw/postalcode.json), so the
    training data does not depend on the address heuristic or its places table."""
    reg = load_postcode_registry()
    cities = sorted({e["city"] for e in reg}) if reg else []
    return cities or ["Kota Bharu", "Kuala Lumpur", "Johor Bahru", "Petaling Jaya", "Georgetown", "Kajang"]


_CITIES = _registry_cities()
_PUBLISHERS = ["Dewan Bahasa dan Pustaka", "Arkib Negara Malaysia", "Penerbit Universiti Malaya", "Prentice Hall",
               "Pearson Malaysia", "Utusan Publications", "Oxford Fajar", "Institut Terjemahan Negara"]
_INSTITUTIONS = ["Hospital", "Universiti Teknologi", "Dewan Bandaraya", "Majlis Perbandaran", "Klinik Kesihatan",
                 "Politeknik", "Kolej Komuniti", "Mahkamah Sesyen", "Balai Polis", "Pejabat Pos"]
_TITLES = ["Permaisuri", "Sultan", "Sultanah", "Tengku Mahkota", "Raja Muda", "Tunku Mahkota"]
_STATES = ["Johor", "Kelantan", "Selangor", "Perak", "Pahang", "Kedah", "Terengganu", "Perlis", "Melaka"]


def negative_place(rng: random.Random) -> Sentence:
    city = rng.choice(_CITIES)
    kind = rng.randint(0, 6)
    year = rng.randint(1998, 2024)
    text = [
        f"{city} : {rng.choice(_PUBLISHERS)} .",
        f"({year}) . Tatabahasa Dewan Edisi ke-{rng.randint(2, 5)} . {city} : {rng.choice(_PUBLISHERS)} .",
        f"Pejabat Pendaftar UTM {city} kerana telah mengambil inisiatif untuk menambah baik proses .",
        f"{rng.choice(_INSTITUTIONS)} {city} akan mengadakan majlis penyampaian anugerah .",
        f"Mesyuarat akan diadakan di {city} selama {rng.randint(2, 4)} hari bermula bulan depan .",
        f"{rng.choice(_TITLES)} {rng.choice(_STATES)} berkenan mencemar duli merasmikan majlis itu .",
        f"Kampus UTM {city} menawarkan program pengajian baharu .",
    ][kind]
    toks, labs = _o(text)
    return Sentence(tokens=toks, labels=labs)


# ---------------------------------------------------------------------
# 4. Names in a row, bracketed signatures
# ---------------------------------------------------------------------

_LIST_LABELS = ["Pengerusi", "Ketua Editor :", "Editor :", "Pembaca Pruf :", "Grafik :", "Percetakan :",
                "Penaung", "Setiausaha :", "Ahli :", "Panel :", "Disediakan oleh :", "Penyelaras :", "Sidang Redaksi"]


def name_list(rng: random.Random) -> Sentence:
    n = rng.randint(2, 6)
    mode = rng.choice(["contiguous", "contiguous", "labelled", "numbered"])
    parts: list[Sentence] = []
    if rng.random() < 0.5:
        parts.append(Sentence(*_o(rng.choice(["SIDANG REDAKSI", "Jawatankuasa Penerbitan", "Panel Penilai", "Senarai Nama"]))))
    for i in range(n):
        # With no separator between names, only a name that has a Bin/Binti/A-L connector has a
        # learnable boundary; informal two-word names run together are ambiguous even to a person.
        # So contiguous lists use connector names ~80% of the time; labelled/numbered lists any.
        who = (long_person_tokens(rng) if rng.random() < 0.8 else person_tokens(rng)) if mode == "contiguous" \
            else mixed_person_tokens(rng, 0.5)
        toks, labs = _entity("PERSON", who)
        if mode == "labelled" or (mode == "contiguous" and rng.random() < 0.25):
            parts.append(Sentence(*_o(rng.choice(_LIST_LABELS))))
        elif mode == "numbered":
            parts.append(Sentence(*_o(f"{i + 1}.")))
        parts.append(Sentence(toks, labs))
    return _cat(parts)


def bracket_signature(rng: random.Random) -> Sentence:
    template = rng.choice([
        "({PU}) {ROLE}", "({P}) {ROLE} {ORG}", "({PU})", "b.p. ({PU})", "u.p.: {HON} {P}", "(u.p.: {HON} {P})",
        "s.k. {HON} {P}", "{CLOSER} ({PU}) {ROLE}",
    ])
    ms = "{CLOSER}" in template and rng.random() < 0.5
    template = template.replace("{CLOSER}", rng.choice(CLOSERS_MS if ms else CLOSERS_EN))
    return render(template, rng, ms, person_fn=_mostly_long)


# ---------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------

def long_name_letter(rng: random.Random) -> Sentence:
    return letter_block(rng, person_fn=_mostly_long)


def long_name_mention(rng: random.Random) -> Sentence:
    return mention_sentence(rng, person_fn=_mostly_long)


def no_postcode_letter(rng: random.Random, bare_pair_share: float = 0.15) -> Sentence:
    return letter_block(rng, person_fn=mixed_person_tokens, extra_slots={"ADDR": _np_slot(bare_pair_share)})


def no_postcode_line(rng: random.Random, bare_pair_share: float = 0.15) -> Sentence:
    template = rng.choice([
        "Alamat : {ADDR}", "Alamat surat-menyurat : {ADDR}", "Beralamat di {ADDR}", "{ADDR}",
        "Address : {ADDR} Tel : {PH}", "Sila hantar ke {ADDR} .", "Pejabat kami terletak di {ADDR} .",
        "{HON} {P}, {ROLE} {ORG}, {ADDR}",
    ])
    return render(template, rng, False, person_fn=mixed_person_tokens, extra_slots={"ADDR": _np_slot(bare_pair_share)})


FAMILIES = [
    (long_name_letter, 0.22), (long_name_mention, 0.08), (name_list, 0.14), (bracket_signature, 0.08),
    (no_postcode_letter, 0.17), (no_postcode_line, 0.11), (negative_place, 0.20),
]


def build_layer(rng: random.Random, count: int, bare_pair_share: float = 0.15) -> list[Sentence]:
    out: list[Sentence] = []
    funcs = [f for f, _ in FAMILIES]
    weights = [w for _, w in FAMILIES]
    for _ in range(count):
        f = rng.choices(funcs, weights)[0]
        out.append(f(rng, bare_pair_share) if f in (no_postcode_letter, no_postcode_line) else f(rng))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v23", default="data/processed/semi_synthetic_v23_ns3k.txt")
    ap.add_argument("--layer-count", type=int, default=2000,
                    help="new sequences. Moderate and spread over 7 families on purpose (v21 regressed after a "
                         "large narrow batch).")
    ap.add_argument("--bare-pair-share", type=float, default=0.15,
                    help="share of no-postcode addresses that are only 'City, State' (0 = none)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--split-seed", type=int, default=123)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--out", default="data/processed/semi_synthetic_v24_ns3k.txt")
    ap.add_argument("--train-pool-out", default="data/processed/train_pool_v24_ns3k.txt")
    ap.add_argument("--test-holdout-out", default="data/processed/test_holdout_v24_ns3k.txt")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = parse_iob2_file(args.v23)
    layer = build_layer(rng, args.layer_count, args.bare_pair_share)
    for s in layer:
        assert len(s.tokens) == len(s.labels)
    combined = base + layer
    rng.shuffle(combined)
    print(f"v23 base: {len(base)} | new layer: {len(layer)} | total: {len(combined)}")
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
