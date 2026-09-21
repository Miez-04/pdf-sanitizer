"""
data.document_templates

Generalizes the resume-shape fix into a proper multi-document-type
system. Two problems this solves together:

1. Document SHAPE diversity: a resume header, a formal letter's
   signature block, and a report's "prepared by" line all place PII
   completely differently. Training only on flowing news-prose
   sentences (however many) never teaches the model these shapes.

2. Sequence-length mismatch: at INFERENCE (pipeline.inference.
   TierTwoPredictor.predict_page), the model runs on an ENTIRE PAGE
   as one sequence — potentially many visual lines concatenated. At
   TRAINING, every example built by inject_connector_clause /
   inject_form_fragment has been a single short line. The model has
   never actually been trained on a sequence resembling real
   page-level inference input. generate_document_fragment() below
   concatenates several realistic document lines into ONE training
   example, closing that gap directly.

Each generator returns a Sentence spanning multiple logical "lines"
of a document type, joined with a distinct O-labelled newline-marker
token so the model can learn line-boundary structure the same way it
would see it in real extracted text (PyMuPDF's word order does not
preserve visual line breaks as a token, so this is an approximation —
see the NEWLINE_MARKER docstring below for the honest caveat).
"""

from __future__ import annotations

import random

from data.registries import ENTITY_GENERATORS
from data.entity_mutation import tokenize, _strip_trailing_honorific
from models.dataset import Sentence

# pdf_ingestion's real Token stream has NO explicit line-break token —
# line structure is implicit in each token's bbox y-coordinate, not in
# the token text itself. Using a literal placeholder word here would
# teach the model a token that never exists in real inference input,
# which is worse than the alternative: just concatenate lines with a
# single space, exactly as pdf_ingestion.extractor's page_text does
# (see its "cursor = char_end + 1" joining logic). This IS how the
# model actually receives multi-line page text at inference, so it's
# the correct choice, not a compromise.
def _join_lines(lines: list[Sentence]) -> Sentence:
    tokens: list[str] = []
    labels: list[str] = []
    for line in lines:
        tokens.extend(line.tokens)
        labels.extend(line.labels)
    return Sentence(tokens=tokens, labels=labels)


def _field_line(label_text: str, entity_type: str, rng: random.Random) -> Sentence:
    """One 'Label : value' line, e.g. 'Tel : 012-3456789'."""
    entity_tokens, resolved_type = ENTITY_GENERATORS[entity_type](rng)
    label_tokens = tokenize(label_text)
    tokens = label_tokens + entity_tokens
    labels = (
        ["O"] * len(label_tokens)
        + [f"B-{resolved_type}"]
        + [f"I-{resolved_type}"] * (len(entity_tokens) - 1)
    )
    return Sentence(tokens=tokens, labels=labels)


def _bare_line(entity_type: str, rng: random.Random, upper: bool = False) -> Sentence:
    """A line that's JUST the entity, no label — e.g. a standalone
    resume name header, or an address directly under it."""
    entity_tokens, resolved_type = ENTITY_GENERATORS[entity_type](rng)
    if upper:
        entity_tokens = [t.upper() for t in entity_tokens]
    labels = [f"B-{resolved_type}"] + [f"I-{resolved_type}"] * (len(entity_tokens) - 1)
    return Sentence(tokens=entity_tokens, labels=labels)


def _plain_line(text: str) -> Sentence:
    tokens = tokenize(text)
    return Sentence(tokens=tokens, labels=["O"] * len(tokens))


# ---------------------------------------------------------------------
# Resume: name header (often all-caps, no label) + address + phone(s) + IC
# ---------------------------------------------------------------------

def generate_resume_header(rng: random.Random, lang: str = "en") -> Sentence:
    lines = [_bare_line("PERSON", rng, upper=rng.random() < 0.5)]

    if rng.random() < 0.6:
        lines.append(_bare_line("ADDRESS", rng))

    phone_labels = (
        ["House phone :", "Mobile phone :", "H/P :", "Tel :"]
        if lang == "en"
        else ["No. Telefon :", "H/P :", "Tel :"]
    )
    for _ in range(rng.randint(1, 2)):
        lines.append(_field_line(rng.choice(phone_labels), "PHONE", rng))

    ic_labels = ["IC number :", "I/C Number :", "NRIC :"] if lang == "en" else ["No. K/P :"]
    if rng.random() < 0.7:
        lines.append(_field_line(rng.choice(ic_labels), "NRIC", rng))

    return _join_lines(lines)


# ---------------------------------------------------------------------
# Personal-particulars form block (table-style: several "Label value"
# rows in a row, as seen in the real resume PDF's "PERSONAL PARTICULARS"
# section)
# ---------------------------------------------------------------------

def generate_form_block(rng: random.Random, lang: str = "en") -> Sentence:
    field_generators_en = [
        ("Name", "PERSON"), ("Address", "ADDRESS"),
        ("Tel (Home) / (H/P)", "PHONE"), ("IC Number", "NRIC"),
    ]
    field_generators_ms = [
        ("Nama", "PERSON"), ("Alamat", "ADDRESS"),
        ("No. Telefon", "PHONE"), ("No. K/P", "NRIC"),
    ]
    fields = field_generators_en if lang == "en" else field_generators_ms
    rng.shuffle(fields)
    chosen = fields[: rng.randint(2, len(fields))]

    lines = [_field_line(f"{label} :", entity_type, rng) for label, entity_type in chosen]
    return _join_lines(lines)


# ---------------------------------------------------------------------
# Formal letter: sender/recipient block + signature block at the end
# ---------------------------------------------------------------------

_LETTER_OPENERS_EN = [
    "Dear Sir / Madam ,", "To Whom It May Concern ,", "Dear Hiring Manager ,",
]
_LETTER_CLOSERS_EN = ["Yours faithfully ,", "Yours sincerely ,", "Best regards ,"]
_LETTER_OPENERS_MS = ["Tuan / Puan ,", "Kepada Sesiapa Yang Berkenaan ,"]
_LETTER_CLOSERS_MS = ["Yang benar ,", "Sekian , terima kasih ."]


def generate_letter_fragment(rng: random.Random, lang: str = "en") -> Sentence:
    openers = _LETTER_OPENERS_EN if lang == "en" else _LETTER_OPENERS_MS
    closers = _LETTER_CLOSERS_EN if lang == "en" else _LETTER_CLOSERS_MS

    lines = [_plain_line(rng.choice(openers))]

    body_en = [
        "I am writing to formally apply for the position advertised on your website .",
        "This letter serves as confirmation of the details discussed during our meeting .",
        "Please find enclosed the requested documents for your review .",
    ]
    body_ms = [
        "Saya menulis surat ini untuk memohon jawatan yang diiklankan di laman web tuan .",
        "Surat ini adalah untuk mengesahkan butiran yang dibincangkan semasa mesyuarat kami .",
    ]
    lines.append(_plain_line(rng.choice(body_en if lang == "en" else body_ms)))

    lines.append(_plain_line(rng.choice(closers)))
    lines.append(_bare_line("PERSON", rng))

    if rng.random() < 0.5:
        contact_label = "Contact :" if lang == "en" else "Hubungi :"
        lines.append(_field_line(contact_label, "PHONE", rng))

    return _join_lines(lines)


# ---------------------------------------------------------------------
# Report / memo: "Prepared by" / "To / From / Date" style contact lines
# ---------------------------------------------------------------------

def generate_report_contact_block(rng: random.Random, lang: str = "en") -> Sentence:
    lines = []
    if lang == "en":
        lines.append(_field_line("Prepared by :", "PERSON", rng))
        if rng.random() < 0.5:
            lines.append(_field_line("Contact Person :", "PERSON", rng))
        lines.append(_field_line("For enquiries , please contact :", "PHONE", rng))
    else:
        lines.append(_field_line("Disediakan oleh :", "PERSON", rng))
        lines.append(_field_line("Untuk pertanyaan , sila hubungi :", "PHONE", rng))
    return _join_lines(lines)


def generate_memo_header(rng: random.Random, lang: str = "en") -> Sentence:
    lines = []
    if lang == "en":
        lines.append(_field_line("To :", "PERSON", rng))
        lines.append(_field_line("From :", "PERSON", rng))
        lines.append(_plain_line("Subject : Internal Policy Update"))
    else:
        lines.append(_field_line("Kepada :", "PERSON", rng))
        lines.append(_field_line("Daripada :", "PERSON", rng))
        lines.append(_plain_line("Perkara : Kemaskini Polisi Dalaman"))
    return _join_lines(lines)


# ---------------------------------------------------------------------
# Biodata: Malaysian personal-particulars format, heavy on DECOY fields
# (gender, race, religion, marital status, age) mixed in with the real
# PII fields. This is specifically a PRECISION trainer: not every
# personal-sounding label-value pair in a biodata block is one of the
# four target entity types, and the model needs negative examples of
# that shape to avoid over-firing on things like "Bangsa : Melayu".
# ---------------------------------------------------------------------

_DECOY_GENDER_EN = ["Male", "Female"]
_DECOY_GENDER_MS = ["Lelaki", "Perempuan"]
_DECOY_RACE_MS = ["Melayu", "Cina", "India", "Iban", "Kadazan"]
_DECOY_RELIGION_MS = ["Islam", "Buddha", "Hindu", "Kristian", "Sikh"]
_DECOY_MARITAL_EN = ["Single", "Married"]
_DECOY_MARITAL_MS = ["Bujang", "Berkahwin"]
_DECOY_NATIONALITY_EN = ["Malaysian"]
_DECOY_NATIONALITY_MS = ["Warganegara Malaysia"]


def _decoy_field_line(label_text: str, values: list[str], rng: random.Random) -> Sentence:
    label_tokens = tokenize(label_text)
    value_tokens = tokenize(rng.choice(values))
    return Sentence(
        tokens=label_tokens + value_tokens,
        labels=["O"] * (len(label_tokens) + len(value_tokens)),
    )


def generate_biodata_block(rng: random.Random, lang: str = "en") -> Sentence:
    real_fields_en = [
        ("Name :", "PERSON"), ("Address :", "ADDRESS"),
        ("Tel (Home) / (H/P) :", "PHONE"), ("IC No :", "NRIC"),
    ]
    real_fields_ms = [
        ("Nama :", "PERSON"), ("Alamat :", "ADDRESS"),
        ("No. Telefon :", "PHONE"), ("No. K/P :", "NRIC"),
    ]
    real_fields = real_fields_en if lang == "en" else real_fields_ms
    lines = [_field_line(label, entity_type, rng) for label, entity_type in real_fields]

    decoys_en = [
        ("Age :", lambda: [str(rng.randint(18, 60))]),
        ("Gender :", lambda: [rng.choice(_DECOY_GENDER_EN)]),
        ("Marital Status :", lambda: [rng.choice(_DECOY_MARITAL_EN)]),
        ("Nationality :", lambda: [rng.choice(_DECOY_NATIONALITY_EN)]),
    ]
    decoys_ms = [
        ("Umur :", lambda: [str(rng.randint(18, 60))]),
        ("Jantina :", lambda: [rng.choice(_DECOY_GENDER_MS)]),
        ("Bangsa :", lambda: [rng.choice(_DECOY_RACE_MS)]),
        ("Agama :", lambda: [rng.choice(_DECOY_RELIGION_MS)]),
        ("Status Perkahwinan :", lambda: [rng.choice(_DECOY_MARITAL_MS)]),
    ]
    decoys = decoys_en if lang == "en" else decoys_ms
    rng.shuffle(decoys)
    for label_text, value_fn in decoys[: rng.randint(2, len(decoys))]:
        lines.append(_decoy_field_line(label_text, value_fn(), rng))

    rng.shuffle(lines)  # real Malaysian biodata forms don't fix field order
    return _join_lines(lines)


# ---------------------------------------------------------------------
# Bill / invoice: billed-to name, address, contact, plus decoy invoice
# reference and amount — hard negatives so the model doesn't learn
# "any digit-heavy field near a name = PII".
# ---------------------------------------------------------------------

def _decoy_invoice_ref(rng: random.Random) -> str:
    """Deliberately shaped to NOT match NRIC/PHONE regex (no 6-2-4
    digit-dash pattern, no phone-prefix digits)."""
    return f"INV-{rng.randint(2020, 2025)}-{rng.randint(10000, 99999)}"


def _decoy_amount(rng: random.Random) -> str:
    return f"RM{rng.randint(50, 9999)}.{rng.randint(0, 99):02d}"


def generate_bill_block(rng: random.Random, lang: str = "en") -> Sentence:
    billed_to_label = "Billed To :" if lang == "en" else "Ditagih Kepada :"
    addr_label = "Address :" if lang == "en" else "Alamat :"
    contact_label = "Contact :" if lang == "en" else "Hubungi :"
    inv_label = "Invoice No :" if lang == "en" else "No. Invois :"
    amount_label = "Amount Due :" if lang == "en" else "Jumlah Perlu Dibayar :"

    lines = [
        _field_line(billed_to_label, "PERSON", rng),
        _field_line(addr_label, "ADDRESS", rng),
        _field_line(contact_label, "PHONE", rng),
        _decoy_field_line(inv_label, [_decoy_invoice_ref(rng)], rng),
        _decoy_field_line(amount_label, [_decoy_amount(rng)], rng),
    ]
    return _join_lines(lines)


# ---------------------------------------------------------------------
# Academic/institutional document: student name, lecturer name, and
# course/matric decoy codes that must NOT be tagged NRIC/PHONE/ADDRESS
# despite being digit-heavy or appearing right next to a name — added
# after a real absence-letter PDF test showed the model tagging a
# course code ("LCC500") as ADDRESS and, in an earlier run, a matric
# number as NRIC. Institution names ("Universiti Teknologi Mara") are
# also included as decoys since they're proper nouns but not PII.
# ---------------------------------------------------------------------

def _decoy_matric_no(rng: random.Random) -> str:
    """Shaped to NOT match NRIC's 6-2-4 dash pattern or PHONE's prefix
    rules — a bare run of digits, as real matric/student IDs are."""
    return str(rng.randint(2020000000, 2026999999))


def _decoy_course_code(rng: random.Random) -> str:
    letters = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(rng.choice([2, 4, 6])))
    return f"{letters}{rng.randint(100, 599)}{rng.choice(['', 'A', 'B'])}"


_INSTITUTIONS = [
    "Universiti Teknologi Mara", "Universiti Malaya", "Universiti Putra Malaysia",
    "Universiti Kebangsaan Malaysia", "Universiti Sains Malaysia",
    "Universiti Teknologi Malaysia", "Kolej Universiti Islam Antarabangsa Selangor",
]


def generate_academic_block(rng: random.Random, lang: str = "en") -> Sentence:
    lines = [_bare_line("PERSON", rng, upper=rng.random() < 0.5)]

    lines.append(
        _decoy_field_line("", [_decoy_matric_no(rng)], rng)
    )
    lines.append(
        _decoy_field_line("", [_decoy_course_code(rng)], rng)
    )

    lecturer_prefix = "DR." if rng.random() < 0.5 else ("PROF." if lang == "en" else "PN.")
    lecturer_line = _bare_line("PERSON", rng, upper=rng.random() < 0.5)
    lecturer_line = Sentence(
        tokens=[lecturer_prefix] + lecturer_line.tokens,
        labels=["O"] + lecturer_line.labels,
    )
    lines.append(lecturer_line)

    role_text = rng.choice(
        ["LECTURER OF", "SUPERVISOR FOR", "COURSE COORDINATOR FOR"]
        if lang == "en"
        else ["PENSYARAH BAGI", "PENYELIA BAGI"]
    )
    lines.append(_decoy_field_line(role_text, [_decoy_course_code(rng)], rng))
    lines.append(_plain_line(rng.choice(_INSTITUTIONS).upper()))

    return _join_lines(lines)


DOCUMENT_GENERATORS = {
    "resume": generate_resume_header,
    "form": generate_form_block,
    "letter": generate_letter_fragment,
    "report": generate_report_contact_block,
    "memo": generate_memo_header,
    "biodata": generate_biodata_block,
    "bill": generate_bill_block,
    "academic": generate_academic_block,
}


def generate_random_document_fragment(rng: random.Random, lang: str = "en") -> Sentence:
    doc_type = rng.choice(list(DOCUMENT_GENERATORS.keys()))
    return DOCUMENT_GENERATORS[doc_type](rng, lang=lang)
