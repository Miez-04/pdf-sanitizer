"""
data/real_samples_2.py

Second real-document batch (data/raw/samples_2.pdf) — 16 pages of
medical/hospital, banking/insurance, and tenancy documents, extending
data/real_samples.py's approach to document types the first batch
(resumes/letters/reports/invoices) didn't cover. Same strategy:
NRIC/PHONE/ADDRESS auto-tagged via this project's own regex_engine,
PERSON hand-curated.
"""

from __future__ import annotations

from data.real_samples import build_real_sample_sentences, write_iob2

PERSON_SPANS_2: dict[int, list[str]] = {
    0: ["CHIN KAH WAI", "BALAN A/L RAMAN", "Dr. Faridah Binti Hassan"],
    1: ["SITI ZUBAIDAH BINTI MUSTAFFA", "Dr. K. Subramaniam"],
    2: ["ANTHONY A/L DASS", "Dr. Ahmad Faisal Bin Razali"],
    3: ["NURUL AIN BINTI ISMAIL", "ISMAIL BIN KASSIM", "Dr. Tan Chin Hock"],
    4: ["DAYANG KU MAIMUNAH"],
    5: ["KAREN TEOH MEI SHAN", "Karen Teoh Mei Shan"],
    6: ["DEVAN A/L MUTHU", "SARASWATHI A/P DEVAN", "ANAND A/L DEVAN"],
    7: ["HALIMAH BINTI SULAIMAN"],
    8: ["BONG JIN CHUNG", "Bong Jin Chung"],
    9: ["AHMAD SHAHIR BIN MUSTAFA"],
    10: ["CHUA KIM HOCK", "NOR HASLINA BINTI YUSOF"],
    11: ["SANTHI A/P RAMAKRISHNAN", "Ahmad Faisal Bin Razali"],
    12: ["LOGANATHAN A/L BALU", "TAN SIEW LING"],
    13: ["STEPHANIE ANAK RICHMOND"],
    14: ["ZULKIFLI BIN MANGSOR", "LIM CHEE SENG", "Lim Chee Seng"],
}


if __name__ == "__main__":
    import sys

    pages_json = sys.argv[1] if len(sys.argv) > 1 else "/tmp/samples2_pages.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/real_samples_2.txt"

    sentences = build_real_sample_sentences(pages_json, person_spans=PERSON_SPANS_2)
    write_iob2(sentences, out_path)

    from collections import Counter

    counts = Counter()
    for s in sentences:
        for lab in s.labels:
            if lab.startswith("B-"):
                counts[lab[2:]] += 1
    print(f"pages processed: {len(sentences)}")
    print(f"entity counts: {dict(counts)}")
    print(f"saved to {out_path}")
