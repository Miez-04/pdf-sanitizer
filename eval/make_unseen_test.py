# pdf-sanitizer/eval/make_unseen_test.py
import sys

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

def spans_of(block, entity_type):
    """
    Input : block (list of (str, str)), entity_type (str)
    Output: list of str, the space-joined text of each span of that type in the block
    """
    out = []   # list of str
    i = 0
    while i < len(block):
        if block[i][1] == "B-" + entity_type:
            j = i + 1
            while j < len(block) and block[j][1] == "I-" + entity_type:
                j += 1
            out.append(" ".join(w for w, _ in block[i:j]))
            i = j
        else:
            i += 1
    return out

def write_blocks(blocks, path):
    """
    Input : blocks (list of blocks), path (str)
    Output: writes 'token TAG' lines with a blank line between blocks.
    """
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for b in blocks:
            for w, t in b:
                f.write(f"{w} {t}\n")
            f.write("\n")

def main(train_path, test_path, out_path):
    try:
        train = load_blocks(train_path)
        test = load_blocks(test_path)
    except OSError as e:
        print(f"Cannot read input file: {e}")
        return

    seen = {"PERSON": set(), "ADDRESS": set()}   # dict: type str -> set of str seen in train
    for b in train:
        for ty in seen:
            seen[ty].update(spans_of(b, ty))

    unseen = []   # list of blocks with no PERSON/ADDRESS string from train
    for b in test:
        leaked = any(s in seen[ty] for ty in seen for s in spans_of(b, ty))   # bool
        if not leaked:
            unseen.append(b)

    try:
        write_blocks(unseen, out_path)
    except OSError as e:
        print(f"Cannot write {out_path}: {e}")
        return
    print(f"test blocks: {len(test)}  unseen-entity blocks: {len(unseen)} "
          f"({100 * len(unseen) / max(len(test), 1):.1f}%)  -> {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: python eval\\make_unseen_test.py <train file> <test file> <output file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])