"""
eval/probe_context.py

Question this answers: is the model weak because of the TRAINING DATA, or
because it is given a whole page (200+ tokens) at once while it was trained
on short sequences (median 19 tokens, max 150)?

How: take the tokens of a real document (an iob2 dump you already have, no
PDF needed) and label every page twice with the SAME checkpoint:

  FULL      the whole page as one sequence (what the pipeline does today)
  WINDOWED  overlapping short windows (default 40 tokens, stride 20); each
            token takes the label from the window where it sits furthest
            from the window edge

Then it lists the PERSON / ADDRESS spans found only when windowed.

How to read the result:
  * windowed finds clearly MORE spans  -> the model can recognise them in a
    short context; the page length hurts. Fix: chunk pages at inference.
    No new data needed for those spans.
  * windowed finds about the same      -> the model does not know these
    patterns even in a short context. Fix: better training data.

Usage (from the folder that contains eval/, models/, pipeline/):
    python -m eval.probe_context --dump "letters_iob2.txt" ^
        --checkpoint models/checkpoints_v23_ns/best_model.pt --window 40 --show-spans

The dump is only used for its TOKENS; its labels are ignored. This is a model-only
probe: no regex, no heuristic, no filters.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

ENTITIES = ("PERSON", "ADDRESS")
_INF = 10**9


def windowed_labels(
    tokens: list[str],
    window: int,
    predict_labels: Callable[[list[str]], list[str]],
) -> list[str]:
    """Label `tokens` using overlapping windows.
    Input: token list, window size, predict_labels(tokens)->labels (one per token).
    Output: list of IOB2 labels, same length as tokens.
    Each token takes the label from the window where its distance to the
    nearest window edge is largest. A window edge that is the start or end of
    the page counts as infinitely far, because there is no truncated context
    there."""
    n = len(tokens)
    if n == 0:
        return []
    if window >= n:
        return predict_labels(tokens)
    stride = max(window // 2, 1)
    starts = list(range(0, n - window + 1, stride))
    if starts[-1] + window < n:
        starts.append(n - window)  # make sure the last window reaches the end
    best_dist = [-1] * n
    chosen = ["O"] * n
    for s in starts:
        e = s + window
        labels = predict_labels(tokens[s:e])
        assert len(labels) == e - s, "predictor returned the wrong number of labels"
        for i in range(s, e):
            left = _INF if s == 0 else i - s
            right = _INF if e == n else e - 1 - i
            d = min(left, right)
            if d > best_dist[i]:
                best_dist[i] = d
                chosen[i] = labels[i - s]
    return _repair_iob2(chosen)


def _repair_iob2(labels: list[str]) -> list[str]:
    """A label taken from a different window can leave 'I-X' without a 'B-X'
    before it. Turn such a dangling I-X into B-X."""
    out = list(labels)
    for i, lab in enumerate(out):
        if lab.startswith("I-"):
            prev = out[i - 1] if i > 0 else "O"
            if prev == "O" or prev[2:] != lab[2:]:
                out[i] = "B-" + lab[2:]
    return out


def spans_of(tokens: list[str], labels: list[str], entity: str) -> list[tuple[int, int, str]]:
    """Returns [(start, end_exclusive, text)] for every span of `entity`."""
    out: list[tuple[int, int, str]] = []
    start = None
    for i, lab in enumerate(labels + ["O"]):
        if lab == f"B-{entity}" or (lab == f"I-{entity}" and start is None):
            if start is not None:
                out.append((start, i, " ".join(tokens[start:i])))
            start = i
        elif lab != f"I-{entity}":
            if start is not None:
                out.append((start, i, " ".join(tokens[start:i])))
            start = None
    return out


def compare_page(
    tokens: list[str],
    full_labels: list[str],
    win_labels: list[str],
) -> dict[str, dict[str, list[str]]]:
    """For each entity: texts found in both / only FULL / only WINDOWED
    (compared by token range overlap, so a longer or shorter span still counts
    as 'found' but is listed as a boundary difference)."""
    result: dict[str, dict[str, list[str]]] = {}
    for ent in ENTITIES:
        f = spans_of(tokens, full_labels, ent)
        w = spans_of(tokens, win_labels, ent)
        both, only_f, only_w = [], [], []
        for fs, fe, ft in f:
            match = next((x for x in w if x[0] < fe and fs < x[1]), None)
            if match is None:
                only_f.append(ft)
            elif (fs, fe) == (match[0], match[1]):
                both.append(ft)
            else:
                both.append(f"{ft}   [window: {match[2]}]")
        for ws, we, wt in w:
            if not any(x[0] < we and ws < x[1] for x in f):
                only_w.append(wt)
        result[ent] = {"both": both, "only_full": only_f, "only_windowed": only_w}
    return result


def run_probe(
    pages_tokens: list[list[str]],
    predict_labels: Callable[[list[str]], list[str]],
    window: int,
    conf_lookup: Callable[[list[str]], list[float] | None] | None = None,
    gate: float | None = None,
    show_spans: bool = False,
) -> str:
    """Runs the comparison on every page; returns the report text.
    show_spans: also list EVERY span the model found on the full page.
    conf_lookup(tokens) -> per-token confidences of the last full-page call
    (or None); with `gate`, spans whose minimum confidence is below it are
    flagged, because the pipeline's confidence gate would delete them."""
    lines: list[str] = []
    if show_spans:
        lines.append(f"confidence gate used for the BELOW GATE flags: {gate}")
    total = {e: {"both": 0, "only_full": 0, "only_windowed": 0} for e in ENTITIES}
    below = {e: 0 for e in ENTITIES}
    for pi, tokens in enumerate(pages_tokens, 1):
        full = predict_labels(tokens)
        confs = conf_lookup(tokens) if conf_lookup else None
        win = windowed_labels(tokens, window, predict_labels)
        cmp = compare_page(tokens, full, win)
        lines.append(f"--- page {pi}: {len(tokens)} tokens")
        if show_spans:
            for ent in ENTITIES:
                for s, e, text in spans_of(tokens, full, ent):
                    conf_txt, flag = "", ""
                    if confs is not None:
                        mc = min(confs[s:e])
                        conf_txt = f"min conf {mc:.2f}  "
                        if gate is not None and mc < gate:
                            flag = f"  <-- BELOW GATE {gate}"
                            below[ent] += 1
                    lines.append(f"    [{ent}] {conf_txt}{text}{flag}")
        for ent in ENTITIES:
            c = cmp[ent]
            for k in total[ent]:
                total[ent][k] += len(c[k])
            lines.append(
                f"  {ent}: both={len(c['both'])}  only FULL={len(c['only_full'])}  "
                f"only WINDOWED={len(c['only_windowed'])}"
            )
            for t in c["only_windowed"]:
                lines.append(f"      + windowed only: {t}")
            for t in c["only_full"]:
                lines.append(f"      - full only:     {t}")
    lines.append("")
    lines.append(f"=== TOTAL (window={window} tokens)")
    for ent in ENTITIES:
        t = total[ent]
        full_n = t["both"] + t["only_full"]
        win_n = t["both"] + t["only_windowed"]
        extra = f"   (below gate: {below[ent]})" if (show_spans and gate is not None) else ""
        lines.append(f"  {ent}: spans found FULL page = {full_n}   WINDOWED = {win_n}{extra}")
    lines.append("")
    lines.append("Windowed >> full  => page length hurts (chunk at inference).")
    lines.append("Windowed ~= full  => the model does not know the pattern (needs data).")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="iob2 dump file; only its tokens are used")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="also save the report to this file")
    ap.add_argument("--show-spans", action="store_true",
                    help="list every span found on the full page, with its minimum confidence")
    ap.add_argument("--gate", type=float, default=None,
                    help="confidence threshold to flag against (default: the pipeline's MIN_MODEL_CONFIDENCE, else 0.7)")
    args = ap.parse_args()

    # heavy imports only here, so the pure functions above can be tested without torch
    from eval.compare_dumps import read_dump
    from eval.synthetic_page import sentence_to_page
    from models.dataset import Sentence
    from pipeline.inference import TierTwoPredictor

    predictor = TierTwoPredictor(args.checkpoint, device=args.device)

    conf_cache: dict[tuple[str, ...], list[float]] = {}

    def predict_labels(tokens: list[str]) -> list[str]:
        page = sentence_to_page(Sentence(tokens=list(tokens), labels=["O"] * len(tokens)))
        labels, conf = predictor.predict_page_with_confidence(page)
        conf_cache[tuple(tokens)] = list(conf)
        return labels

    gate = args.gate
    if gate is None:
        try:
            from pipeline.conflict_resolution import MIN_MODEL_CONFIDENCE as gate
        except ImportError:  # the original_hybrid folder has no confidence gate
            gate = 0.7

    pages_tokens, _ = read_dump(args.dump)
    report = run_probe(
        pages_tokens, predict_labels, args.window,
        conf_lookup=lambda toks: conf_cache.get(tuple(toks)),
        gate=gate, show_spans=args.show_spans,
    )
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
