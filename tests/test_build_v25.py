"""tests/test_build_v25.py - table / heading layer."""

import random
from pathlib import Path

from data.build_v25_corpus import (
    CITIZEN, FAMILIES, GENDERS, STATUS, build_layer, course_table_only, headings_only, invoice_table,
    label_row, label_value_short, transcript_full, value_row,
)


def _spans(sent, typ):
    out, cur = [], None
    for i, lab in enumerate(sent.labels):
        if lab == f"B-{typ}":
            cur = [i, i + 1]
            out.append(cur)
        elif lab == f"I-{typ}" and cur is not None:
            cur[1] = i + 1
        else:
            cur = None
    return [(a, b) for a, b in out]


def _valid_iob2(labels):
    prev = "O"
    for lab in labels:
        if lab.startswith("I-") and (prev == "O" or prev[2:] != lab[2:]):
            return False
        prev = lab
    return True


def test_layer_is_well_formed_deterministic_and_bounded():
    a = build_layer(random.Random(1), 300)
    b = build_layer(random.Random(1), 300)
    assert [(s.tokens, s.labels) for s in a] == [(s.tokens, s.labels) for s in b]
    for s in a:
        assert len(s.tokens) == len(s.labels) and s.tokens and _valid_iob2(s.labels)
        assert len(s.tokens) <= 300
        assert "{" not in " ".join(s.tokens)
    assert len(FAMILIES) == 5


def test_headings_label_rows_and_course_tables_are_entirely_O():
    rng = random.Random(2)
    for _ in range(150):
        assert set(headings_only(rng).labels) == {"O"}
        assert set(label_row(rng).labels) == {"O"}
        assert set(course_table_only(rng).labels) == {"O"}


def test_value_row_marks_only_name_nric_and_address():
    rng = random.Random(3)
    split_addresses = 0
    non_entity = set(GENDERS) | {w for c in CITIZEN for w in c.split()} | {w for s in STATUS for w in s.split()}
    for _ in range(300):
        s = value_row(rng)
        persons = _spans(s, "PERSON")
        assert len(persons) == 1
        a, b = persons[0]
        assert not any(t in non_entity or t.isdigit() for t in s.tokens[a:b])      # no gender word / number inside a name
        assert len(_spans(s, "NRIC")) <= 1
        for i, (t, lab) in enumerate(zip(s.tokens, s.labels)):
            if t in GENDERS or (t.isdigit() and len(t) == 10):
                assert lab == "O"                                                  # gender word, student number: O
        split_addresses += len(_spans(s, "ADDRESS")) == 2
    assert split_addresses > 40                                                    # address often split in two cells


def test_invoice_has_billing_entities_and_O_items():
    rng = random.Random(4)
    for _ in range(100):
        s = invoice_table(rng)
        assert _spans(s, "PERSON") and _spans(s, "ADDRESS")
        text = " ".join(t for t, l in zip(s.tokens, s.labels) if l == "O")
        assert "SUBTOTAL" in text and "TOTAL" in text


def test_transcripts_contain_long_O_runs_and_real_pii():
    rng = random.Random(5)
    longest_o = 0
    for _ in range(80):
        s = transcript_full(rng)
        assert _spans(s, "PERSON")
        run = best = 0
        for lab in s.labels:
            run = run + 1 if lab == "O" else 0
            best = max(best, run)
        longest_o = max(longest_o, best)
    assert longest_o >= 60


def test_builder_does_not_depend_on_the_rule_engine():
    source = (Path(__file__).resolve().parents[1] / "data" / "build_v25_corpus.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in source.splitlines() if l.lstrip().startswith(("import ", "from ")))
    assert "regex_engine" not in code
