# pdf-sanitizer/eval/ablate_combine.py
import argparse
from seqeval.metrics import classification_report

from models.dataset import parse_iob2_file
from eval.synthetic_page import sentence_to_page
from pdf_ingestion.schema import DocumentTokens
from pipeline.conflict_resolution import resolve_page, address_heuristic
from regex_engine.matcher import build_regex_mask_registry


def run(sentences, heuristic_on):
    """
    Input : sentences (list of Sentence objects with .tokens and .labels), heuristic_on (bool)
    Output: tuple (gold list of list of str, combined list of list of str)
    Tier 2 is replaced by the gold labels, so any difference comes from the combine step.
    """
    gold_all, comb_all = [], []
    with address_heuristic(heuristic_on):                           # context manager: switch the heuristic for this block only
        for sent in sentences:
            page = sentence_to_page(sent)                           # PageTokens built from the sentence
            doc = DocumentTokens(source_path="<ablate>", pages=[page])
            registry = build_regex_mask_registry(doc)               # Tier 1 regex matches
            resolve_page(page, registry, list(sent.labels), None)   # gold labels act as perfect Tier 2
            gold_all.append(list(sent.labels))
            comb_all.append([t.label for t in page.tokens])         # labels after conflict resolution
    return gold_all, comb_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    try:
        sentences = parse_iob2_file(args.data)
    except OSError as e:
        print(f"Cannot read {args.data}: {e}")
        return
    for flag in (True, False):
        gold, comb = run(sentences, flag)
        print(f"\n##### perfect Tier 2 + combine step, address_heuristic={flag}")
        print(classification_report(gold, comb, digits=4))


if __name__ == "__main__":
    main()