"""tests/test_probe_context.py - the page-length probe (no model needed)."""

from eval.probe_context import (
    _repair_iob2, compare_page, run_probe, spans_of, windowed_labels,
)


def _fake_predictor(max_len: int):
    """A pretend model that tags the name 'Ahmad Bin Ibrahim' as PERSON, but only
    when it is given at most max_len tokens (simulates long-input failure)."""
    def predict(tokens: list[str]) -> list[str]:
        labels = ["O"] * len(tokens)
        if len(tokens) > max_len:
            return labels
        for i in range(len(tokens) - 2):
            if tokens[i:i + 3] == ["Ahmad", "Bin", "Ibrahim"]:
                labels[i], labels[i + 1], labels[i + 2] = "B-PERSON", "I-PERSON", "I-PERSON"
        return labels
    return predict


def _page(n_before: int) -> list[str]:
    return ["filler"] * n_before + ["Ahmad", "Bin", "Ibrahim", "(MyKad:"] + ["filler"] * 30


def test_windowed_finds_what_the_full_page_misses():
    tokens = _page(100)
    predict = _fake_predictor(max_len=60)
    full = predict(tokens)
    win = windowed_labels(tokens, 40, predict)
    assert spans_of(tokens, full, "PERSON") == []
    assert [t for _, _, t in spans_of(tokens, win, "PERSON")] == ["Ahmad Bin Ibrahim"]


def test_short_page_gives_identical_results():
    tokens = _page(5)
    predict = _fake_predictor(max_len=60)
    assert windowed_labels(tokens, 80, predict) == predict(tokens)


def test_name_at_document_start_and_end_is_not_lost():
    predict = _fake_predictor(max_len=60)
    start = ["Ahmad", "Bin", "Ibrahim"] + ["x"] * 120
    end = ["x"] * 120 + ["Ahmad", "Bin", "Ibrahim"]
    for tokens in (start, end):
        found = [t for _, _, t in spans_of(tokens, windowed_labels(tokens, 40, predict), "PERSON")]
        assert found == ["Ahmad Bin Ibrahim"]


def test_dangling_i_tag_is_repaired():
    assert _repair_iob2(["O", "I-PERSON", "I-PERSON"]) == ["O", "B-PERSON", "I-PERSON"]
    assert _repair_iob2(["B-PERSON", "I-PERSON"]) == ["B-PERSON", "I-PERSON"]


def test_report_lists_the_difference_and_totals():
    report = run_probe([_page(100)], _fake_predictor(60), 40)
    assert "+ windowed only: Ahmad Bin Ibrahim" in report
    assert "PERSON: spans found FULL page = 0   WINDOWED = 1" in report


def test_compare_page_marks_boundary_differences():
    tokens = ["A", "B", "C", "D"]
    full = ["B-PERSON", "I-PERSON", "O", "O"]
    win = ["B-PERSON", "I-PERSON", "I-PERSON", "O"]
    both = compare_page(tokens, full, win)["PERSON"]["both"]
    assert len(both) == 1 and "[window: A B C]" in both[0]


def test_show_spans_lists_confidence_and_flags_spans_below_the_gate():
    tokens = _page(5)
    predict = _fake_predictor(max_len=60)
    confs = [0.99] * len(tokens)
    for i, w in enumerate(tokens):          # the name spans tokens 5..7; make one token weak
        if w == "Bin":
            confs[i] = 0.55
    report = run_probe([tokens], predict, 80, conf_lookup=lambda t: confs, gate=0.7, show_spans=True)
    assert "[PERSON] min conf 0.55  Ahmad Bin Ibrahim  <-- BELOW GATE 0.7" in report
    assert "PERSON: spans found FULL page = 1   WINDOWED = 1   (below gate: 1)" in report


def test_show_spans_off_by_default_keeps_the_old_report():
    report = run_probe([_page(5)], _fake_predictor(60), 80)
    assert "[PERSON]" not in report and "below gate" not in report


def test_report_states_which_gate_value_was_used():
    report = run_probe([_page(5)], _fake_predictor(60), 80, conf_lookup=lambda t: [0.9] * len(t), gate=0.3, show_spans=True)
    assert report.startswith("confidence gate used for the BELOW GATE flags: 0.3")
