"""
pdf-sanitizer/eval/metrics.py

Pure-Python metric functions for Chapter 4 (no torch / PyMuPDF import).

Inputs everywhere:
    gold, pred : list[list[str]]   one inner list of IOB2 tag strings per
                                   sentence/page, e.g. ["B-PERSON","I-PERSON","O"]
                                   len(gold[i]) must equal len(pred[i]).
"""

from __future__ import annotations

from collections import Counter

ENTITY_TYPES = ["PERSON", "ADDRESS", "NRIC", "PHONE"]  # list[str]
LABEL_ORDER = ["O"] + ENTITY_TYPES                      # list[str], confusion-matrix order


def to_type(tag: str) -> str:
    """'B-PERSON' -> 'PERSON', 'I-ADDRESS' -> 'ADDRESS', 'O' -> 'O'.
    Input: str. Output: str. Drops the B-/I- prefix so tokens are compared
    by entity category only (token-level metrics, report Sec 3.4.1)."""
    return tag if tag == "O" else tag.split("-", 1)[1]


def _flatten_types(seqs: list[list[str]]) -> list[str]:
    """list of tag lists -> one flat list of entity-type strings."""
    return [to_type(tag) for seq in seqs for tag in seq]


def _prf(tp: int, fp: int, fn: int) -> dict:
    """Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R).
    A zero denominator gives 0.0 (not an exception)."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def token_level_report(gold: list[list[str]], pred: list[list[str]]) -> dict:
    """Token-level TP/FP/FN per entity type (report Sec 3.4.1 definitions):
        TP: gold type == t and pred type == t
        FP: pred type == t and gold type != t
        FN: gold type == t and pred type != t
    TN is not used in P/R. 'micro' sums TP/FP/FN over the 4 entity types
    (the O class is excluded). 'macro' averages the 4 per-type F1 values.
    'token_accuracy' = correct tokens / all tokens INCLUDING O, so it is
    inflated by the O majority: report it, but do not use it alone."""
    _check_shapes(gold, pred)
    g, p = _flatten_types(gold), _flatten_types(pred)
    per_type = {}
    for t in ENTITY_TYPES:
        tp = sum(1 for a, b in zip(g, p) if a == t and b == t)
        fp = sum(1 for a, b in zip(g, p) if b == t and a != t)
        fn = sum(1 for a, b in zip(g, p) if a == t and b != t)
        per_type[t] = _prf(tp, fp, fn)
        per_type[t]["support"] = tp + fn  # number of gold tokens of this type
    micro = _prf(
        sum(v["tp"] for v in per_type.values()),
        sum(v["fp"] for v in per_type.values()),
        sum(v["fn"] for v in per_type.values()),
    )
    macro_f1 = sum(v["f1"] for v in per_type.values()) / len(ENTITY_TYPES)
    correct = sum(1 for a, b in zip(g, p) if a == b)
    return {
        "per_type": per_type,
        "micro": micro,
        "macro_f1": macro_f1,
        "token_accuracy": correct / len(g) if g else 0.0,
        "n_tokens": len(g),
    }


def detection_only_report(gold: list[list[str]], pred: list[list[str]]) -> dict:
    """Type-agnostic redaction view: a token counts as detected if gold != O
    and pred != O, whatever the entity type. FN here = a sensitive token that
    would be LEFT VISIBLE (a leak). Wrong-type tokens are still redacted, so
    they are not leaks. Output: same dict shape as one _prf() result."""
    _check_shapes(gold, pred)
    g, p = _flatten_types(gold), _flatten_types(pred)
    tp = sum(1 for a, b in zip(g, p) if a != "O" and b != "O")
    fp = sum(1 for a, b in zip(g, p) if a == "O" and b != "O")
    fn = sum(1 for a, b in zip(g, p) if a != "O" and b == "O")
    return _prf(tp, fp, fn)


def confusion_matrix(gold: list[list[str]], pred: list[list[str]]) -> dict:
    """Token confusion matrix. Output: dict[gold_type][pred_type] -> int,
    over LABEL_ORDER. Row = gold label, column = predicted label."""
    _check_shapes(gold, pred)
    counts = Counter(zip(_flatten_types(gold), _flatten_types(pred)))
    return {g: {p: counts.get((g, p), 0) for p in LABEL_ORDER} for g in LABEL_ORDER}


def entity_level_report(gold: list[list[str]], pred: list[list[str]]) -> dict:
    """Exact-span entity metrics via seqeval (a predicted entity counts only
    if its type AND both boundaries match gold). Stricter than token-level;
    this is what eval/run_eval.py already prints. Output: dict with
    'per_type' (seqeval report dict) and 'micro_f1' (float)."""
    from seqeval.metrics import classification_report, f1_score  # lazy import
    _check_shapes(gold, pred)
    return {
        "per_type": classification_report(gold, pred, output_dict=True, zero_division=0),
        "micro_f1": f1_score(gold, pred),
    }


def _check_shapes(gold: list[list[str]], pred: list[list[str]]) -> None:
    if len(gold) != len(pred):
        raise ValueError(f"gold has {len(gold)} sequences, pred has {len(pred)}")
    for i, (a, b) in enumerate(zip(gold, pred)):
        if len(a) != len(b):
            raise ValueError(f"sequence {i}: gold {len(a)} tags vs pred {len(b)} tags")


def format_markdown(name: str, gold: list[list[str]], pred: list[list[str]]) -> str:
    """Build a Markdown block (tables) you can paste into Chapter 4.
    Input: set name + gold/pred. Output: str."""
    tok = token_level_report(gold, pred)
    det = detection_only_report(gold, pred)
    cm = confusion_matrix(gold, pred)
    try:
        ent_f1 = f"{entity_level_report(gold, pred)['micro_f1']:.4f}"
    except ImportError:  # seqeval is in requirements.txt; tolerate a missing install
        ent_f1 = "n/a (seqeval not installed)"

    out = [f"### {name}  ({tok['n_tokens']} tokens)", ""]
    out += ["**Token-level (report Sec 3.4.1)**", "",
            "| Entity | TP | FP | FN | Precision | Recall | F1 | Support |",
            "|---|---|---|---|---|---|---|---|"]
    for t in ENTITY_TYPES:
        v = tok["per_type"][t]
        out.append(f"| {t} | {v['tp']} | {v['fp']} | {v['fn']} | {v['precision']:.4f} | "
                   f"{v['recall']:.4f} | {v['f1']:.4f} | {v['support']} |")
    m = tok["micro"]
    out.append(f"| **micro** | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.4f} | "
               f"{m['recall']:.4f} | {m['f1']:.4f} | |")
    out += ["", f"Macro F1 = {tok['macro_f1']:.4f}; token accuracy (includes O) = "
                f"{tok['token_accuracy']:.4f}", ""]
    out += ["**Detection-only (any entity type; FN = left visible)**", "",
            f"Precision {det['precision']:.4f} | Recall {det['recall']:.4f} | F1 {det['f1']:.4f} "
            f"(TP {det['tp']}, FP {det['fp']}, FN {det['fn']})", ""]
    out += ["**Token confusion matrix (rows = gold, columns = predicted)**", "",
            "| gold \\ pred | " + " | ".join(LABEL_ORDER) + " |",
            "|---|" + "---|" * len(LABEL_ORDER)]
    for g in LABEL_ORDER:
        out.append(f"| {g} | " + " | ".join(str(cm[g][p]) for p in LABEL_ORDER) + " |")
    out += ["", f"**Entity-level exact-span micro F1 (seqeval)** = {ent_f1}", ""]
    return "\n".join(out)
