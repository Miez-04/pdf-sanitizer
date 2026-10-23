# pdf-sanitizer/eval/audit_dataset.py
import re
import sys
from collections import Counter

def load_blocks(path):
    """
    Input : path (str) to a file with 'token TAG' per line (space separated), blank line between blocks.
    Output: list of blocks; each block is a list of (token str, tag str) tuples.
    """
    blocks = []   # list of list of (str, str)
    cur = []      # list of (str, str)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():                      # blank line ends a block
                if cur:
                    blocks.append(cur)
                    cur = []
                continue
            parts = line.rsplit(" ", 1)               # list of str: [token, tag]; rsplit keeps tokens with spaces intact
            if len(parts) == 2:
                cur.append((parts[0], parts[1]))
    if cur:
        blocks.append(cur)
    return blocks

def person_spans(block):
    """
    Input : block (list of (str, str))
    Output: list of lists of (str, str), one list per PERSON span
    """
    out = []
    i = 0
    while i < len(block):
        if block[i][1] == "B-PERSON":
            j = i + 1
            while j < len(block) and block[j][1] == "I-PERSON":
                j += 1
            out.append(block[i:j])
            i = j
        else:
            i += 1
    return out

def main(path):
    try:
        blocks = load_blocks(path)
    except OSError as e:
        print(f"Cannot read {path}: {e}")
        return

    # 1. Single-letter PERSON spans (letters plus optional punctuation)
    initials = Counter()   # Counter: span text str -> int
    for b in blocks:
        for sp in person_spans(b):
            text = " ".join(w for w, _ in sp)
            if re.fullmatch(r"[A-Za-z][,.;:]*", text):
                initials[text] += 1
    print("single-letter PERSON spans:", sum(initials.values()))
    print(initials.most_common(20))

    # 2. Exact duplicate blocks
    counts = Counter(tuple(b) for b in blocks)        # tuple of tuples is hashable
    dups = sum(1 for v in counts.values() if v > 1)
    print("duplicated blocks:", dups, "of", len(counts), "unique")

if __name__ == "__main__":
    main(sys.argv[1])   # usage: python eval/audit_dataset.py data/processed/semi_synthetic_v25_ns3k.txt