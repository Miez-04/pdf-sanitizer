"""
data/build_corpus.py

Run from the project root:

    python -m data.build_corpus \
        --merged data/raw/merged.txt \
        --malay-news data/raw/news-30k.json \
        --malay-sample-size 2000 \
        --out data/processed/semi_synthetic_v2.txt

Produces an IOB2 file in the same format as the original
latest_train_dataset.txt (token label per line, blank line between
sentences), plus a stats printout mirroring Table 3.4 of the report
and the skeleton-diversity check used earlier in this project to
diagnose the original dataset's template-memorization problem.

Scoping decisions (stated plainly, not buried):
- English baseline (merged.txt) gets real PERSON-replacement injection
  via spaCy where a sentence contains a detectable name; NRIC/PHONE/
  ADDRESS always use connector-clause injection (these don't occur
  naturally in news prose).
- Malay baseline (news-30k.json) gets connector-clause injection only,
  for ALL entity types, including PERSON. There is no reliable Malay
  NER available here, so replacement-based injection isn't attempted
  on Malay text — attempting it with an English-tuned model would
  silently produce wrong entity boundaries, which is worse than not
  doing it. The real-sentence diversity gain from the Malay corpus
  still applies to the connector clause's carrier context.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import spacy

from data.entity_mutation import (
    inject_by_replacement,
    inject_connector_clause,
    inject_form_fragment,
    tokenize,
)
from data.document_templates import DOCUMENT_GENERATORS
from models.dataset import Sentence

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def naive_sentence_split(text: str) -> list[str]:
    """Language-agnostic sentence splitter (works for both English and
    Malay, since Malay uses the same terminal punctuation). Not
    linguistically perfect — doesn't handle abbreviations like 'Dr.'
    specially — but errs toward over-splitting, which just yields
    slightly shorter carrier sentences, not incorrect labels."""
    text = text.replace("\r\n", " ").strip()
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if 20 <= len(p.strip()) <= 220]


def load_english_baseline(path: str | Path) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in raw.split("\r\n\r\n") if p.strip()]
    sentences: list[str] = []
    for para in paragraphs:
        sentences.extend(naive_sentence_split(para))
    return sentences


def load_malay_baseline(path: str | Path, sample_size: int, rng: random.Random) -> list[str]:
    articles = json.loads(Path(path).read_text(encoding="utf-8"))
    rng.shuffle(articles)
    sentences: list[str] = []
    for article in articles[: sample_size]:
        text = article.get("text", "")
        sentences.extend(naive_sentence_split(text))
    return sentences


def skeleton(sent: Sentence) -> str:
    out = []
    for tok, lab in zip(sent.tokens, sent.labels):
        if lab == "O":
            out.append(tok)
        elif lab.startswith("B-"):
            out.append(f"<{lab[2:]}>")
    return " ".join(out)


def write_iob2(sentences: list[Sentence], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for tok, lab in zip(sent.tokens, sent.labels):
                f.write(f"{tok} {lab}\n")
            f.write("\n")


def print_stats(sentences: list[Sentence], label: str) -> None:
    skeletons = [skeleton(s) for s in sentences]
    unique = len(set(skeletons))
    entity_counts = Counter()
    for s in sentences:
        for lab in s.labels:
            if lab.startswith("B-"):
                entity_counts[lab[2:]] += 1

    print(f"--- {label} ---")
    print(f"sentences: {len(sentences)}")
    print(f"unique skeletons: {unique}  (avg {len(sentences)/max(unique,1):.1f} sentences/skeleton)")
    print(f"entity instances: {dict(entity_counts)}")
    print(f"total tokens: {sum(len(s) for s in sentences)}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", default="data/raw/merged.txt")
    parser.add_argument("--malay-news", default="data/raw/news-30k.json")
    parser.add_argument("--malay-sample-size", type=int, default=2000)
    parser.add_argument("--out", default="data/processed/semi_synthetic_v2.txt")
    parser.add_argument("--target-per-entity", type=int, default=1100)
    parser.add_argument(
        "--form-fragment-ratio", type=float, default=0.3,
        help="Fraction of each entity type's instances generated as standalone "
             "resume/form-style fragments (no base sentence) rather than injected "
             "into prose. Added after real resume PDFs showed the model had never "
             "seen bare 'AHMED ZAINAH' / 'Tel : ...' style fragments at all.",
    )
    parser.add_argument(
        "--document-fragment-count", type=int, default=800,
        help="Number of multi-field document blocks (resume/biodata/letter/report/"
             "memo/bill) to generate, split across languages. Each block contributes "
             "SEVERAL co-occurring entities at once (name+address+phone+IC together), "
             "which single-entity fragments can't teach — this is what trains entity "
             "boundary discipline in a realistic multi-field context, plus decoy "
             "fields (biodata's Gender/Race/Religion, bill's Invoice No/Amount) for "
             "precision.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    nlp = spacy.load("en_core_web_sm")

    en_sentences = load_english_baseline(args.merged)
    ms_sentences = load_malay_baseline(args.malay_news, args.malay_sample_size, rng)
    print(f"loaded {len(en_sentences)} English baseline sentences")
    print(f"loaded {len(ms_sentences)} Malay baseline sentences")

    entity_types = ["PERSON", "PHONE", "NRIC", "ADDRESS"]
    output: list[Sentence] = []
    entity_counts = Counter()

    # Multi-field document blocks FIRST — resume/biodata/letter/report/
    # memo/bill, each with several co-occurring entities plus decoy
    # fields. Contributes a VARYING number of each entity type per
    # block (not 1-to-1), so entity_counts is incremented by actually
    # scanning each generated block's labels, not assumed.
    doc_generators = list(DOCUMENT_GENERATORS.items())
    for _ in range(args.document_fragment_count):
        doc_type, gen_fn = rng.choice(doc_generators)
        lang = rng.choice(["en", "ms"])
        sent = gen_fn(rng, lang=lang)
        output.append(sent)
        for lab in sent.labels:
            if lab.startswith("B-"):
                entity_counts[lab[2:]] += 1

    # Form-fragment injection next: standalone resume/form-style
    # fragments (bare "AHMED ZAINAH", "Tel : 012-3456789") with no
    # real base sentence at all. Respects what the document-block
    # phase above already contributed — target is a CEILING on this
    # style's share, not an unconditional addition, or entity counts
    # would blow past target_per_entity once document blocks are
    # already contributing a substantial share themselves.
    for entity_type in entity_types:
        form_target = int(args.target_per_entity * args.form_fragment_ratio)
        while entity_counts[entity_type] < form_target:
            lang = rng.choice(["en", "ms"])
            sent = inject_form_fragment(entity_type, rng, lang=lang)
            output.append(sent)
            entity_counts[entity_type] += 1

    # English: try replacement for PERSON first, fall back to connector.
    en_pool = en_sentences[:]
    rng.shuffle(en_pool)
    idx = 0
    while entity_counts["PERSON"] < args.target_per_entity and idx < len(en_pool):
        text = en_pool[idx]
        idx += 1
        sent = inject_by_replacement(nlp, text, rng)
        if sent is None:
            sent = inject_connector_clause(text, "PERSON", rng, lang="en")
        output.append(sent)
        entity_counts["PERSON"] += 1

    for entity_type in ["PHONE", "NRIC", "ADDRESS"]:
        while entity_counts[entity_type] < args.target_per_entity and idx < len(en_pool):
            text = en_pool[idx]
            idx += 1
            sent = inject_connector_clause(text, entity_type, rng, lang="en")
            output.append(sent)
            entity_counts[entity_type] += 1

    # Malay: connector-clause injection for everything, topping up
    # whichever entity types are still under target after English.
    ms_pool = ms_sentences[:]
    rng.shuffle(ms_pool)
    idx = 0
    while any(entity_counts[e] < args.target_per_entity for e in entity_types) and idx < len(ms_pool):
        entity_type = rng.choice(
            [e for e in entity_types if entity_counts[e] < args.target_per_entity]
        )
        text = ms_pool[idx]
        idx += 1
        sent = inject_connector_clause(text, entity_type, rng, lang="ms")
        output.append(sent)
        entity_counts[entity_type] += 1

    rng.shuffle(output)
    write_iob2(output, args.out)

    print_stats(output, f"semi_synthetic_v2 ({args.out})")
    print(f"final entity counts: {dict(entity_counts)}")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
