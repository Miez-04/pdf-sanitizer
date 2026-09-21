"""
data.registries

Implements FYP Section 3.2.2 Step 2 ("Programmatic PII Injection")'s
three named registries:
  - Multi-ethnic Naming Registry
  - Public Sector Corporate Enterprise Directory (-> phone generation)
  - National Postal and Geographic Registry (-> address generation)

Every generator returns (span_tokens, entity_type) — a list of
whitespace-level tokens (matching pdf_ingestion's tokenization
granularity, same reasoning as the "Tan Sri" split fix in
models/dataset.py) plus the IOB2 entity label to tag them with.

All generation is seeded (pass a random.Random instance) for
reproducible corpus builds.
"""

from __future__ import annotations

import random
import re

# ---------------------------------------------------------------------
# Multi-ethnic Naming Registry
# ---------------------------------------------------------------------

MALAY_MALE_FIRST = [
    "Ahmad", "Muhammad", "Mohd", "Amir", "Hafiz", "Farid", "Zulkifli",
    "Ismail", "Rahman", "Syafiq", "Aiman", "Danial", "Haziq", "Iskandar",
    "Khairul", "Nabil", "Rizal", "Shahrul", "Zaid", "Faiz",
    # Expanded after a real OOV failure ("Ahmed" — alternate spelling
    # of "Ahmad" — went unrecognized end to end): more common given
    # names AND common alternate transliterations, since Malaysian
    # documents don't standardize spelling (Ahmed/Ahmad, Mohamad/
    # Muhammad/Mohammed all appear in real records).
    "Ahmed", "Mohamad", "Mohamed", "Mohammed", "Azman", "Azlan",
    "Firdaus", "Hakim", "Hazim", "Idris", "Irfan", "Kamal", "Luqman",
    "Naim", "Ridzuan", "Saiful", "Syahmi", "Zack", "Zafran", "Adam",
    "Akmal", "Amirul", "Asyraf", "Fahmi", "Haziq", "Imran", "Rayyan",
]
MALAY_FEMALE_FIRST = [
    "Nurul", "Siti", "Aisyah", "Nadhirah", "Farah", "Aina", "Zulaikha",
    "Hidayah", "Aida", "Balqis", "Fatimah", "Izzati", "Nabila", "Sofia",
    "Amirah", "Diana", "Husna", "Iman", "Maya", "Qistina",
    # Expanded alongside MALAY_MALE_FIRST for the same OOV reason.
    "Zainah", "Zainab", "Mariam", "Maryam", "Aisyah", "Alia", "Ain",
    "Damia", "Elina", "Farhana", "Hanani", "Insyirah", "Liyana",
    "Nurhaliza", "Puteri", "Raihan", "Sarah", "Syafiqah", "Wardina",
    "Yasmin", "Zara", "Adriana", "Batrisyia",
]
MALAY_PATRONYM = [
    "Abdullah", "Ismail", "Rahman", "Zulkifli", "Hassan", "Ibrahim",
    "Kassim", "Yusof", "Osman", "Ariffin", "Latiff", "Mansor", "Aziz",
    "Rashid", "Salleh", "Talib", "Wahab", "Zain",
    # Expanded alongside the given-name lists above.
    "Zainah", "Bakar", "Hamid", "Idris", "Jamil", "Karim", "Majid",
    "Nordin", "Omar", "Rahim", "Samad", "Sulaiman", "Yaakob", "Zaman",
]

CHINESE_SURNAMES = [
    "Tan", "Lim", "Lee", "Wong", "Ng", "Chan", "Ong", "Goh", "Teo",
    "Chong", "Yap", "Low", "Chua", "Koh", "Sim", "Loh",
    "Lau", "Liew", "Ho", "Yeoh", "Khoo", "Ang", "Chin",
]
CHINESE_GIVEN = [
    "Wei Ming", "Chee Keong", "Mei Ling", "Chin Huat", "Siew Fong",
    "Kok Wai", "Li Ying", "Boon Hock", "Hui Min", "Jia Wen", "Zhi Hao",
    "Yan Ting", "Kah Wai", "Su Lin",
    "Yee Ling", "Kar Mun", "Wai Kit", "Xin Yi", "Cheng Yew", "Poh Choo",
    "Mei Yee", "Wan Ling", "Chee Hong", "Sook Yee",
]

INDIAN_FIRST = [
    "Arjun", "Kumar", "Suresh", "Priya", "Deepa", "Ravi", "Anand",
    "Kavitha", "Vijay", "Muthu", "Ganesh", "Lakshmi", "Sathish", "Meera",
    "Devendran", "Chandra", "Kalai", "Nirmala", "Raj", "Selvi",
    "Thevan", "Uma", "Yamuna", "Balan", "Gopal", "Malar",
]
INDIAN_FATHER = [
    "Ravi", "Subramaniam", "Krishnan", "Muthusamy", "Chandran",
    "Rajendran", "Naidu", "Pillai", "Sivam", "Kumaran",
    "Govindasamy", "Maniam", "Nathan", "Perumal", "Sivalingam",
    "Thevar", "Veerasamy",
]

HONORIFICS_MALE = ["Encik", "Tuan", "Datuk", "Dato'", "Tan Sri", "Dr", "Haji", "Ir"]
HONORIFICS_FEMALE = ["Cik", "Puan", "Datin", "Puan Sri", "Dr", "Hajjah"]
HONORIFICS_NEUTRAL = ["Dr", "Prof"]  # safe to prefix either gendered form


ENGLISH_GIVEN_NAMES = [
    "John", "Peter", "Michael", "David", "James", "Andrew", "Daniel",
    "Mary", "Susan", "Grace", "Jessica", "Sarah", "Amanda", "Rachel",
    "Kevin", "Steven", "Eric", "Alvin", "Vincent", "Nicholas",
]


CHINESE_CHARACTER_NAMES = [
    "林国良", "黄丽达", "陈美玲", "李俊杰", "王秀英", "张建国",
    "刘雅婷", "吴伟明", "郑志强", "何嘉欣",
]

# East Malaysian (Sarawak/Sabah) native naming: "Anak"/"ak" ("child
# of") replaces bin/binti/a-l/a-p — a completely distinct naming
# convention this project had never covered at all until a real
# formatting reference catalogue flagged it. First names are very
# often English/Christian given names.
EAST_MALAYSIAN_FIRST = [
    "Stephanie", "Marilyn", "Patricia", "Grace", "Richard", "Joseph",
    "Simon", "Jenny", "Alice", "Thomas", "Willie", "Michael", "Susan",
    "Robert", "Linda", "Charles", "Doreen", "Edwin", "Florence",
]
EAST_MALAYSIAN_FATHER = [
    "Jinggut", "Jalong", "Nyalang", "Tuah", "Jimbun", "Anggat",
    "Sagan", "Rentap", "Bilong", "Empaling", "Gunggu", "Lasah",
]


def generate_person(rng: random.Random) -> tuple[list[str], str]:
    """Returns (tokens, "PERSON"). Roughly matches ethnic distribution
    visible in the existing dataset (Malay/Chinese/Indian, honorifics
    on ~50% of instances, matching Table 3.4's honorific-inclusive
    PERSON entity definition).

    Given/patronym names are sometimes COMPOUND (2 words) — real Malay
    names are very often "Muhammad Hakimi bin Muhammad Jauhari" (5
    tokens), not just "Ahmad bin Abdullah" (3 tokens).

    CRITICALLY, real Malaysian documents constantly use SHORTENED/
    INFORMAL name forms — and testing against a real batch of resumes
    and letters showed these aren't a minority variant, they're the
    DOMINANT form in this document type: every single name across
    multiple real resumes/letters used an informal form ("R Theva",
    "Ahmed Zainah", "John Lee", "Dr. Iskandar Ishak"). An earlier
    version of this generator produced informal forms only ~30-35% of
    the time within each ethnicity branch, which under-represented
    them relative to what real documents actually show — probabilities
    below are set to match observed reality (~55-60%), not guessed.
      - Malay: "bin"/"binti" is very often DROPPED entirely in real
        documents ("Ahmad Zainal", not "Ahmad bin Zainal").
      - Indian: the father's name is very often abbreviated to a bare
        INITIAL with the "a/l"/"a/p" marker dropped too ("R Theva",
        not "Theva a/l Ravi").
      - Chinese: English-educated Malaysian Chinese very often use
        WESTERN given-name-first order with an English given name
        ("John Lee"), not just the traditional Surname-Given order.
    """
    ethnicity = rng.choice(["malay_m", "malay_f", "chinese", "indian", "east_malaysian"])
    include_honorific = rng.random() < 0.5

    def _malay_given(pool: list[str]) -> list[str]:
        if rng.random() < 0.35:
            return [rng.choice(pool), rng.choice(pool)]
        return [rng.choice(pool)]

    def _malay_patronym() -> list[str]:
        if rng.random() < 0.25:
            return [rng.choice(MALAY_MALE_FIRST), rng.choice(MALAY_PATRONYM)]
        return [rng.choice(MALAY_PATRONYM)]

    def _malay_marker(full_word: str, abbrev_forms: list[str]) -> list[str]:
        """55% dropped entirely, 25% full word, 20% abbreviated
        ("b." / "bt" / "bte") — abbreviations found via a real
        formatting reference catalogue, not previously covered."""
        roll = rng.random()
        if roll < 0.55:
            return []
        elif roll < 0.80:
            return [full_word]
        else:
            return [rng.choice(abbrev_forms)]

    if ethnicity == "malay_m":
        given = _malay_given(MALAY_MALE_FIRST)
        patronym = _malay_patronym()
        marker = _malay_marker("bin", ["b.", "B."])
        name = given + marker + patronym
        honorific = rng.choice(HONORIFICS_MALE) if include_honorific else None
    elif ethnicity == "malay_f":
        given = _malay_given(MALAY_FEMALE_FIRST)
        patronym = _malay_patronym()
        marker = _malay_marker("binti", ["bt", "bt.", "bte", "bte."])
        name = given + marker + patronym
        honorific = rng.choice(HONORIFICS_FEMALE) if include_honorific else None
    elif ethnicity == "chinese":
        surname = rng.choice(CHINESE_SURNAMES)
        # Raised from 0.35 -> 0.55.
        if rng.random() < 0.55:
            # Western given-first order with an English given name —
            # the "John Lee" pattern, common among English-educated
            # Malaysian Chinese and completely absent before.
            name = [rng.choice(ENGLISH_GIVEN_NAMES), surname]
        else:
            given_parts = rng.choice(CHINESE_GIVEN).split(" ")
            if len(given_parts) == 2 and rng.random() < 0.3:
                # Hyphenated given name ("Wei-Jian") rather than
                # space-separated — a real, previously-uncovered
                # variant. Kept as ONE token (no internal space),
                # matching how it would appear as a single PyMuPDF word.
                given_parts = [f"{given_parts[0]}-{given_parts[1]}"]
            name = [surname, *given_parts]

        # Parenthetical nickname or Chinese-character name — found
        # missing entirely via real PDF testing: "Leong Ka Wai
        # (Alvin)" and "Lim Kok Leong (林国良)" both had their
        # parenthetical portion left completely unredacted (in one
        # case the WHOLE name, main part included, was missed), since
        # the generator had never produced this structure at all.
        parenthetical_roll = rng.random()
        if parenthetical_roll < 0.20:
            name = name + [f"({rng.choice(ENGLISH_GIVEN_NAMES)})"]
        elif parenthetical_roll < 0.35:
            name = name + [f"({rng.choice(CHINESE_CHARACTER_NAMES)})"]

        honorific = (
            rng.choice(HONORIFICS_MALE + HONORIFICS_FEMALE) if include_honorific else None
        )
    elif ethnicity == "indian":
        father = rng.choice(INDIAN_FATHER)
        if rng.random() < 0.55:
            # Initial-based shorthand with the marker dropped
            # entirely — "R Theva", a very common real form, not just
            # the fully-expanded "Theva a/l Ravindran".
            name = [f"{father[0]}", rng.choice(INDIAN_FIRST)]
        else:
            # "s/o"/"d/o" (son/daughter of) alongside the more
            # traditional "a/l"/"a/p" — both appear in real documents,
            # found via a formatting reference catalogue.
            marker = rng.choice(["a/l", "a/p", "s/o", "d/o"])
            name = [rng.choice(INDIAN_FIRST), marker, father]
        honorific = rng.choice(HONORIFICS_MALE + HONORIFICS_FEMALE) if include_honorific else None
    else:  # east_malaysian
        given = rng.choice(EAST_MALAYSIAN_FIRST)
        father = rng.choice(EAST_MALAYSIAN_FATHER)
        marker = rng.choice(["Anak", "ak"])
        name = [given, marker, father]
        honorific = None  # Western/Christian first names rarely take a Malay honorific

    # Alias format: "Ahmad bin Ali @ Abu" — the "@" symbol marking an
    # alternate/preferred name. A distinct real convention (seen in
    # official documents) this generator never produced. Applied
    # across any ethnicity at low probability, appended after the
    # main name.
    if rng.random() < 0.08:
        alias_pool = MALAY_MALE_FIRST + MALAY_FEMALE_FIRST + ENGLISH_GIVEN_NAMES
        name = name + ["@", rng.choice(alias_pool)]

    tokens = (honorific.split(" ") if honorific else []) + name
    return tokens, "PERSON"


# ---------------------------------------------------------------------
# Public Sector Corporate Enterprise Directory -> phone generation
# ---------------------------------------------------------------------

_MOBILE_PREFIXES = ["010", "011", "012", "013", "014", "016", "017", "018", "019"]
_LANDLINE_PREFIXES = ["03", "04", "05", "06", "07", "08", "09"]


def _digits(rng: random.Random, n: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def generate_phone(rng: random.Random) -> tuple[list[str], str]:
    """Generates a phone string matching regex_engine.patterns exactly
    (mobile 01X-XXXXXXX / 011-XXXXXXXX, landline 03-XXXXXXXX /
    0X-XXXXXXX, with ~20% using the +60 international form) — so the
    injected gold label and Tier 1's own regex agree by construction."""
    use_intl = rng.random() < 0.2
    is_mobile = rng.random() < 0.7

    if is_mobile:
        prefix = rng.choice(_MOBILE_PREFIXES)
        subscriber = _digits(rng, 8) if prefix == "011" else _digits(rng, 7)
    else:
        prefix = rng.choice(_LANDLINE_PREFIXES)
        subscriber = _digits(rng, 8) if prefix == "03" else _digits(rng, 7)

    local_number = f"{prefix}-{subscriber}"
    if use_intl:
        text = "+60" + local_number[1:]  # drop leading 0, prepend +60
    else:
        text = local_number

    return [text], "PHONE"


# ---------------------------------------------------------------------
# NRIC generation
# ---------------------------------------------------------------------

_VALID_MONTHS_DAYS = [(m, d) for m in range(1, 13) for d in (1, 10, 15, 20, 28)]
_VALID_PB_CODES = [f"{i:02d}" for i in range(1, 17)]  # 01-16, safely valid per JPN


def generate_nric(rng: random.Random) -> tuple[list[str], str]:
    yy = rng.randint(0, 99)
    mm, dd = rng.choice(_VALID_MONTHS_DAYS)
    pb = rng.choice(_VALID_PB_CODES)
    sn = _digits(rng, 4)
    text = f"{yy:02d}{mm:02d}{dd:02d}-{pb}-{sn}"
    return [text], "NRIC"


# ---------------------------------------------------------------------
# National Postal and Geographic Registry -> address generation
# ---------------------------------------------------------------------

ROAD_TYPES = ["Jalan", "Lorong", "Persiaran", "Lebuh"]
ROAD_NAMES = ["Merdeka", "Aman", "Bahagia", "Damai", "Sentosa", "Harmoni", "Indah", "Utama"]
AREA_TYPES = ["Taman", "Kampung", "Bandar", "Seksyen"]
AREA_NAMES = ["Bahagia", "Melati", "Impian", "Ceria", "Permai", "Jaya", "Sri Aman"]
STATES_AND_CITIES = [
    ("50480", "Kuala Lumpur"),
    ("43000", "Kajang, Selangor"),
    ("11900", "Bayan Lepas, Penang"),
    ("80000", "Johor Bahru, Johor"),
    ("93450", "Kuching, Sarawak"),
    ("88000", "Kota Kinabalu, Sabah"),
    ("15000", "Kota Bharu, Kelantan"),
    ("25000", "Kuantan, Pahang"),
]

BUILDING_PREFIXES = ["No.", "Lot", "Suite", "Unit", "Level", "Tingkat", "Blok", "Block", "PT"]


# ---------------------------------------------------------------------
# National Postal and Geographic Registry + real company address pool.
# Loaded lazily from real government/registry data (data/raw/) — see
# load_postcode_registry / load_company_address_pool. Falls back to the
# small hardcoded STATES_AND_CITIES list above if those files aren't
# present, so this module still works (with reduced realism) in
# environments that don't have the raw registry files, e.g. CI.
# ---------------------------------------------------------------------

_POSTCODE_REGISTRY_CACHE: list[dict] | None = None
_COMPANY_ADDRESS_POOL_CACHE: list[str] | None = None

DEFAULT_POSTCODE_PATH = "data/raw/postalcode.json"
DEFAULT_COMPANY_ADDRESS_PATH = "data/raw/company_addresses.csv"

_JUNK_ADDRESS_VALUES = {"", "tiada", "tiada maklumat", "-", "n/a"}
_TRAILING_CONTACT_RE = re.compile(
    r"\s*[\(,]?\s*(?:T\s*:|TEL\s*[:.]?|NO\.?\s*TEL|F\s*:|FAKS).*$", re.IGNORECASE
)


def load_postcode_registry(path: str = DEFAULT_POSTCODE_PATH) -> list[dict]:
    global _POSTCODE_REGISTRY_CACHE
    if _POSTCODE_REGISTRY_CACHE is not None:
        return _POSTCODE_REGISTRY_CACHE
    try:
        import json

        with open(path, encoding="utf-8") as f:
            _POSTCODE_REGISTRY_CACHE = json.load(f)["data"]
    except (FileNotFoundError, KeyError, ValueError):
        _POSTCODE_REGISTRY_CACHE = []
    return _POSTCODE_REGISTRY_CACHE


def load_company_address_pool(path: str = DEFAULT_COMPANY_ADDRESS_PATH) -> list[str]:
    """Real, government-published company addresses (Section 3.2.2's
    Postal and Geographic Registry, applied to real Malaysian direct-
    sales company license records). This is the actual source of
    address SHAPE diversity — No./Lot/Suite/Unit/Level/Tingkat/Wisma/
    Block prefixes, multi-line building+floor+unit compounds — that a
    single synthetic template can't approximate."""
    global _COMPANY_ADDRESS_POOL_CACHE
    if _COMPANY_ADDRESS_POOL_CACHE is not None:
        return _COMPANY_ADDRESS_POOL_CACHE

    pool: list[str] = []
    try:
        import csv

        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                raw = row.get("ALAMAT LESEN", "").strip()
                cleaned = _clean_company_address(raw)
                if cleaned:
                    pool.append(cleaned)
    except FileNotFoundError:
        pool = []

    _COMPANY_ADDRESS_POOL_CACHE = pool
    return _COMPANY_ADDRESS_POOL_CACHE


def _clean_company_address(raw: str) -> str | None:
    """Strips embedded phone/fax suffixes (found in ~2% of rows —
    verified by grep against the raw CSV during development) and
    rejects junk placeholder values ('Tiada', 'Tiada Maklumat', empty)."""
    text = _TRAILING_CONTACT_RE.sub("", raw)
    text = re.sub(r"\s+", " ", text).strip().rstrip(".,")
    if text.lower() in _JUNK_ADDRESS_VALUES or len(text) < 15:
        return None
    return text


_ADDR_TOKEN_RE = re.compile(r"\w+(?:['/]\w+)*|[^\w\s]")


def _tokenize_address(text: str) -> list[str]:
    return _ADDR_TOKEN_RE.findall(text)


def generate_address(rng: random.Random) -> tuple[list[str], str]:
    """
    Two sources, mirroring what real Malaysian addresses actually look
    like (confirmed against 1,695 real company license addresses —
    prefixes seen: No., Lot, Suite, Unit, Level, Tingkat, Wisma, Block,
    Bilik, Blok, Plot, Penthouse, PT — NOT just "No. X, Jalan Y"):

    - ~65%: a REAL address string sampled verbatim from the company
      address pool (load_company_address_pool). Genuine shape
      diversity, since it's actual registry data, not invented.
    - ~35%: synthetically composed using a REAL postcode/place/city/
      state combination from load_postcode_registry, with a randomized
      building-type prefix. Exists so entity volume isn't capped by
      the ~1,695 real addresses when a corpus build needs more
      instances than that, while staying geographically valid (real
      postcode <-> place <-> city <-> state, not a fabricated pairing).

    Falls back to the small hardcoded STATES_AND_CITIES list if the
    real registry files aren't present in this environment.
    """
    pool = load_company_address_pool()
    if pool and rng.random() < 0.65:
        text = rng.choice(pool)
        return _tokenize_address(text), "ADDRESS"

    registry = load_postcode_registry()
    prefix = rng.choice(BUILDING_PREFIXES)
    number = rng.randint(1, 200)
    # ~25% of the time, drop the prefix word entirely — a bare "45
    # Jalan X, postcode City" with no "No./Lot/..." lead-in at all is
    # a common, simple home-address shape a real resume test showed
    # was completely missing from training (every synthetic address
    # always had an explicit prefix token before this fix).
    include_prefix = rng.random() < 0.75

    if registry:
        entry = rng.choice(registry)
        place, city, state, poskod = entry["place"], entry["city"], entry["state"], entry["code"]
        text = (
            f"{prefix} {number}, {place}, {poskod} {city}, {state}"
            if include_prefix
            else f"{number} {place}, {poskod} {city}, {state}"
        )
    else:
        poskod, city = rng.choice(STATES_AND_CITIES)
        road = f"{rng.choice(ROAD_TYPES)} {rng.choice(ROAD_NAMES)} {rng.randint(1, 20)}"
        area = f"{rng.choice(AREA_TYPES)} {rng.choice(AREA_NAMES)}"
        text = (
            f"{prefix} {number}, {road}, {area}, {poskod} {city}"
            if include_prefix
            else f"{number} {road}, {poskod} {city}"
        )

    return _tokenize_address(text), "ADDRESS"


ENTITY_GENERATORS = {
    "PERSON": generate_person,
    "PHONE": generate_phone,
    "NRIC": generate_nric,
    "ADDRESS": generate_address,
}
