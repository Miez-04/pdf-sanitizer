# pdf-sanitizer/eval/dedupe_blocks.py
import sys

def main(src, dst):
    """
    Input : src (str) path of the dataset, dst (str) path of the output file.
    Output: writes dst containing each distinct block once, in first-seen order.
    """
    try:
        with open(src, encoding="utf-8", errors="replace") as f:
            raw_lines = [ln.rstrip("\r\n") for ln in f]       # list of str
    except OSError as e:
        print(f"Cannot read {src}: {e}")
        return

    blocks = []   # list of list of str (each str is one 'token TAG' line)
    cur = []      # list of str
    for ln in raw_lines:
        if not ln.strip():
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(ln)
    if cur:
        blocks.append(cur)

    seen = set()      # set of tuple of str (hashable block fingerprint)
    kept = []         # list of list of str
    for b in blocks:
        key = tuple(b)
        if key in seen:
            continue                                          # skip repeat
        seen.add(key)
        kept.append(b)

    try:
        with open(dst, "w", encoding="utf-8", newline="\n") as f:
            for b in kept:
                f.write("\n".join(b) + "\n\n")                # one blank line between blocks
    except OSError as e:
        print(f"Cannot write {dst}: {e}")
        return

    print(f"blocks in: {len(blocks)}  blocks out: {len(kept)}  removed: {len(blocks) - len(kept)}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])