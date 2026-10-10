import pytest

from eval.metrics import (
    token_level_report, detection_only_report, confusion_matrix,
    entity_level_report, to_type,
)

GOLD = [["B-PERSON", "I-PERSON", "O", "B-NRIC", "O", "B-ADDRESS", "I-ADDRESS"]]
PRED = [["B-PERSON", "O",        "B-PERSON", "B-NRIC", "O", "B-ADDRESS", "O"]]


def test_to_type_strips_prefix():
    assert to_type("B-PERSON") == "PERSON" and to_type("I-ADDRESS") == "ADDRESS"
    assert to_type("O") == "O"


def test_token_level_counts_by_hand():
    r = token_level_report(GOLD, PRED)["per_type"]
    # PERSON: gold tokens 0,1 ; pred tokens 0,2 -> TP 1, FP 1, FN 1
    assert (r["PERSON"]["tp"], r["PERSON"]["fp"], r["PERSON"]["fn"]) == (1, 1, 1)
    assert r["PERSON"]["precision"] == 0.5 and r["PERSON"]["recall"] == 0.5
    # NRIC exact
    assert r["NRIC"]["f1"] == 1.0
    # ADDRESS: gold 5,6 ; pred 5 -> TP 1, FP 0, FN 1
    assert (r["ADDRESS"]["tp"], r["ADDRESS"]["fp"], r["ADDRESS"]["fn"]) == (1, 0, 1)
    # PHONE: no gold, no pred -> zeros, no ZeroDivisionError
    assert r["PHONE"]["f1"] == 0.0


def test_micro_and_accuracy():
    rep = token_level_report(GOLD, PRED)
    assert (rep["micro"]["tp"], rep["micro"]["fp"], rep["micro"]["fn"]) == (3, 1, 2)
    # 7 tokens, tokens 0,3,4,5 match -> 4/7
    assert rep["token_accuracy"] == pytest.approx(4 / 7)


def test_detection_only_ignores_wrong_type():
    gold = [["B-PERSON", "O"]]
    pred = [["B-ADDRESS", "O"]]  # wrong type but still redacted -> not a leak
    d = detection_only_report(gold, pred)
    assert (d["tp"], d["fp"], d["fn"]) == (1, 0, 0)


def test_confusion_matrix_rows_are_gold():
    cm = confusion_matrix(GOLD, PRED)
    assert cm["PERSON"]["PERSON"] == 1 and cm["PERSON"]["O"] == 1
    assert cm["O"]["PERSON"] == 1 and cm["ADDRESS"]["O"] == 1


def test_entity_level_is_stricter_than_token_level():
    pytest.importorskip("seqeval")
    ent = entity_level_report(GOLD, PRED)
    assert ent["micro_f1"] < token_level_report(GOLD, PRED)["micro"]["f1"]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        token_level_report([["O"]], [["O", "O"]])
