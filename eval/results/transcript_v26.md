### v26_ns on transcript  (406 tokens)

**Token-level (report Sec 3.4.1)**

| Entity    | TP | FP | FN |Precision|Recall |   F1   | Support |
|---        |--- |--- |--- |---     |---     |---     |---      |
| PERSON    | 3  | 10 |  2 | 0.2308 | 0.6000 | 0.3333 | 5       |
| ADDRESS   | 10 | 37 |  1 | 0.2128 | 0.9091 | 0.3448 | 11      |
| NRIC      | 1  | 0  |  0 | 1.0000 | 1.0000 | 1.0000 | 1       |
| PHONE     | 0  | 0  |  0 | 0.0000 | 0.0000 | 0.0000 | 0       |
| **micro** | 14 | 47 |  3 | 0.2295 | 0.8235 | 0.3590 |         |

Macro F1 = 0.4195; token accuracy (includes O) = 0.8768

**Detection-only (any entity type; FN = left visible)**

Precision 0.2295 | Recall 0.8235 | F1 0.3590 (TP 14, FP 47, FN 3)

**Token confusion matrix (rows = gold, columns = predicted)**

| gold \ pred | O | PERSON | ADDRESS | NRIC | PHONE |
|-------------|---|--------|---------|------|-------|
| O           |342|    0   |    37   |   0  |   0   |
| PERSON      | 2 |    3   |    0    |   0  |   0   |
| ADDRESS     | 1 |    0   |   10    |   0  |   0   |
| NRIC        | 0 |    0   |    0    |   1  |   0   |
| PHONE       | 0 |    0   |    0    |   0  |   0   |

**Entity-level exact-span micro F1 (seqeval)** = 0.1538
