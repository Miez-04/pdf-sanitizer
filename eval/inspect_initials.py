# pdf-sanitizer/eval/inspect_initials.py
import re
import sys
from collections import defaultdict

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

def main(path, window=4, max_per_group=6):
    try:
        blocks = load_blocks(path)
    except OSError as e:
        print(f"Cannot read {path}: {e}")
        return

    groups = defaultdict(list)   # dict: span text str -> list of context strings
    for bi, block in enumerate(blocks):
        i = 0
        while i < len(block):
            if block[i][1] == "B-PERSON":
                j = i + 1
                while j < len(block) and block[j][1] == "I-PERSON":
                    j += 1
                text = " ".join(w for w, _ in block[i:j])          # str: span text
                if re.fullmatch(r"[A-Za-z][,.;:]*", text):
                    left = " ".join(w for w, _ in block[max(0, i - window):i])      # str: tokens before span
                    right = " ".join(w for w, _ in block[j:j + window])             # str: tokens after span
                    groups[text].append(f"block {bi}: {left} [[{text}]] {right}")
                i = j
            else:
                i += 1

    for text, ctxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"=== {text!r} ({len(ctxs)} spans) ===")
        for c in ctxs[:max_per_group]:
            print("  ", c)

if __name__ == "__main__":
    main(sys.argv[1])