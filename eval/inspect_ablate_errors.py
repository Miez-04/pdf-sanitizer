# pdf-sanitizer/eval/inspect_ablate_errors.py
import argparse

from models.dataset import parse_iob2_file
from eval.synthetic_page import sentence_to_page
from pdf_ingestion.schema import DocumentTokens
from pipeline.conflict_resolution import resolve_page, address_heuristic
from regex_engine.matcher import build_regex_mask_registry


def spans(labels, entity):
    """
    Input : labels (list of str, IOB2 tags), entity (str, e.g. 'ADDRESS')
    Output: set of (start int, end int) tuples, end exclusive, one per span of that entity
    """
    out = set()
    i = 0
    while i < len(labels):
        if labels[i] == "B-" + entity:
            j = i + 1
            while j < len(labels) and labels[j] == "I-" + entity:
                j += 1
            out.add((i, j))
            i = j
        else:
            i += 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()
    try:
        sentences = parse_iob2_file(args.data)
    except OSError as e:
        print(f"Cannot read {args.data}: {e}")
        return

    lost_addr, extra_phone = [], []   # list of str, printed at the end
    with address_heuristic(False):    # heuristic OFF for this run
        for sent in sentences:
            page = sentence_to_page(sent)
            doc = DocumentTokens(source_path="<inspect>", pages=[page])
            registry = build_regex_mask_registry(doc)
            resolve_page(page, registry, list(sent.labels), None)   # gold labels act as perfect Tier 2
            comb = [t.label for t in page.tokens]

            for a, z in spans(sent.labels, "ADDRESS") - spans(comb, "ADDRESS"):
                lost_addr.append(" ".join(sent.tokens[a:z]))         # gold ADDRESS span not reproduced
            for a, z in spans(comb, "PHONE") - spans(sent.labels, "PHONE"):
                ctx = " ".join(sent.tokens[max(0, a - 3):z + 3])      # 3 tokens of context each side
                extra_phone.append(f"[{' '.join(sent.tokens[a:z])}]  <- {ctx}")

    print(f"gold ADDRESS spans lost: {len(lost_addr)}")
    for s in lost_addr[:args.limit]:
        print("  ", s)
    print(f"false-positive PHONE spans: {len(extra_phone)}")
    for s in extra_phone[:args.limit]:
        print("  ", s)


if __name__ == "__main__":
    main()