# Hybrid PDF Sanitizer (Regex + Bi-LSTM-CRF)

A two-tier document sanitization system for Malaysian PDFs. Combines a
deterministic Regex engine (MyKad NRIC, phone numbers) with a
Bi-LSTM-CRF sequence labeller (names, addresses) to detect and redact
Personally Identifiable Information (PII) at the vector level, with a
human-in-the-loop review step before final output.

## Architecture

```
PDF Ingestion → [Tier 1: Regex | Tier 2: Bi-LSTM-CRF] → Conflict
Resolution Gate → HITL Review → Spatial Mapping → Vector Redaction
```

- **Tier 1 (Regex):** matches rigid structural identifiers — MyKad
  NRIC (`YYMMDD-PB-###G`), mobile/landline numbers.
- **Tier 2 (Bi-LSTM-CRF):** context-aware labelling for names
  (including Bin/Binti/A-L/A-P honorifics) and addresses.
- **Conflict resolution:** regex match always wins over the neural
  prediction for a given token.
- **HITL:** detected entities are rendered as bounding boxes for
  human accept/reject/manual-override before the final redaction pass.

## Project structure

```
pdf-sanitizer/
├── data/
│   ├── raw/            # baseline Malay/English corpora (gitignored)
│   └── processed/      # IOB2-tagged train/test splits (gitignored)
├── pdf_ingestion/       # PyMuPDF token stream + spatial coordinate table
├── regex_engine/        # Tier 1: MyKad / phone pattern matching
├── models/               # Bi-LSTM-CRF model code + checkpoints (gitignored)
├── pipeline/            # conflict resolution, spatial mapping, redaction
├── ui/                  # HITL review workspace
├── eval/                # Precision / Recall / F1 evaluation scripts
├── tests/
├── notebooks/           # exploration / training notebooks
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Roadmap

1. [x] Repo scaffold
2. [x] PDF ingestion & spatial layer (`pdf_ingestion/`)
3. [x] Regex engine (`regex_engine/`)
4. [ ] Semi-synthetic dataset pipeline (`data/`)
5. [ ] Bi-LSTM-CRF model + training (`models/`)
6. [ ] Conflict resolution & vector redaction (`pipeline/`)
7. [ ] Human-in-the-loop review UI (`ui/`)
8. [ ] Evaluation (Precision / Recall / F1) (`eval/`)

## Status

Early scaffold — see roadmap above.
