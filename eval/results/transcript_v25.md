### v25_ns on transcript  (406 tokens)

**Token-level (report Sec 3.4.1)**

| Entity    | TP | FP | FN | Precision | Recall |   F1   | Support |
|---        |--- |--- |--- |---        |---     |---     |---      |
| PERSON    | 5  | 4  |  0 |  0.5556   | 1.0000 | 0.7143 | 5 |
| ADDRESS   | 11 | 8  |  0 |  0.5789   | 1.0000 | 0.7333 | 11|
| NRIC      | 1  | 0  |  0 |  1.0000   | 1.0000 | 1.0000 | 1 |
| PHONE     | 0  | 0  |  0 |  0.0000   | 0.0000 | 0.0000 | 0 |
| **micro** | 17 | 12 |  0 |  0.5862   | 1.0000 | 0.7391 |   |

Macro F1 = 0.6119; token accuracy (includes O) = 0.9704

**Detection-only (any entity type; FN = left visible)**

Precision 0.5862 | Recall 1.0000 | F1 0.7391 (TP 17, FP 12, FN 0)

**Token confusion matrix (rows = gold, columns = predicted)**

| gold \ pred | O | PERSON | ADDRESS | NRIC | PHONE |
|---          |---|---     |---      |---   |---    |
| O           |377|   4    |    8    |  0   |   0   |
| PERSON      | 0 |   5    |    0    |  0   |   0   |
| ADDRESS     | 0 |   0    |   11    |  0   |   0   |
| NRIC        | 0 |   0    |    0    |  1   |   0   |
| PHONE       | 0 |   0    |    0    |  0   |   0   |

**Entity-level exact-span micro F1 (seqeval)** = 0.5000
