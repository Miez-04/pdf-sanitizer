# pdf-sanitizer/eval/split_stats.py
import sys
from collections import Counter

def load_blocks(path):
    """
    Input : path (str), file with 'token TAG' per line, blank line between blocks.
    Output: list of blocks; each block is a list of (token str, tag str) tuples.
    """
    blocks = []   # list of list of (str, str)
    cur = []      # list of (str, str)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                if cur:
                    blocks.append(cur)
                    cur = []
                continue
            parts = line.rsplit(" ", 1)               # list of str: [token, tag]
            if len(parts) == 2:
                cur.append((parts[0], parts[1]))
    if cur:
        blocks.append(cur)
    return blocks

def summarize(path):
    """
    Input : path (str)
    Output: dict with block count, zero-entity share, span counts per type, O share
    """
    blocks = load_blocks(path)                                          # list of blocks
    tags = Counter(t for b in blocks for _, t in b)                     # Counter: tag str -> int
    total = sum(tags.values())                                          # int
    spans = Counter(t[2:] for b in blocks for _, t in b if t.startswith("B-"))   # Counter: type str -> int
    zero = sum(1 for b in blocks if all(t == "O" for _, t in b))        # int
    return {
        "blocks": len(blocks),
        "zero_entity_%": round(100 * zero / max(len(blocks), 1), 1),
        "O_%": round(100 * tags["O"] / max(total, 1), 1),
        "PERSON": spans["PERSON"],
        "ADDRESS": spans["ADDRESS"],
        "NRIC": spans["NRIC"],
        "PHONE": spans["PHONE"],
    }

def main(folder):
    for name in ("train", "val", "test"):
        try:
            print(name, summarize(f"{folder}/{name}.txt"))
        except OSError as e:
            print(f"Cannot read {folder}/{name}.txt: {e}")

if __name__ == "__main__":
    main(sys.argv[1])   # usage: python eval\split_stats.py data\processed