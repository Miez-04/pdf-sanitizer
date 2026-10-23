# pdf-sanitizer/eval/split_dataset.py
import random
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

def entity_strings(blocks, entity_type):
    """
    Input : blocks (list of blocks), entity_type (str, e.g. 'PERSON')
    Output: set of str, each str is the space-joined text of one entity span
    """
    out = set()   # set of str
    for b in blocks:
        i = 0
        while i < len(b):
            if b[i][1] == "B-" + entity_type:
                j = i + 1
                while j < len(b) and b[j][1] == "I-" + entity_type:
                    j += 1
                out.add(" ".join(w for w, _ in b[i:j]))
                i = j
            else:
                i += 1
    return out

def write_blocks(blocks, path):
    """
    Input : blocks (list of blocks), path (str)
    Output: writes one 'token TAG' line per token, blank line between blocks.
    """
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for b in blocks:
            for w, t in b:
                f.write(f"{w} {t}\n")
            f.write("\n")

def main(src, out_dir, seed=42, train_frac=0.8, val_frac=0.1):
    try:
        blocks = load_blocks(src)
    except OSError as e:
        print(f"Cannot read {src}: {e}")
        return

    # Stratify by whether a block has any entity, so every split keeps the same share of zero-entity blocks
    with_e = [b for b in blocks if any(t != "O" for _, t in b)]      # list of blocks
    no_e = [b for b in blocks if all(t == "O" for _, t in b)]        # list of blocks

    rng = random.Random(seed)   # random.Random object, fixed seed gives the same split on every run
    train, val, test = [], [], []
    for group in (with_e, no_e):
        rng.shuffle(group)                                            # in-place shuffle of a list
        n = len(group)                                                # int
        n_train = int(n * train_frac)                                 # int
        n_val = int(n * val_frac)                                     # int
        train += group[:n_train]
        val += group[n_train:n_train + n_val]
        test += group[n_train + n_val:]
    for s in (train, val, test):
        rng.shuffle(s)

    try:
        write_blocks(train, f"{out_dir}/train.txt")
        write_blocks(val, f"{out_dir}/val.txt")
        write_blocks(test, f"{out_dir}/test.txt")
    except OSError as e:
        print(f"Cannot write to {out_dir}: {e}")
        return

    print(f"train: {len(train)}  val: {len(val)}  test: {len(test)}")

    # Leakage report: share of test entity strings that also occur in train
    for ty in ("PERSON", "ADDRESS"):
        tr = entity_strings(train, ty)       # set of str
        te = entity_strings(test, ty)        # set of str
        overlap = len(te & tr)               # int: strings present in both sets
        print(f"{ty}: unique test spans {len(te)}, also in train {overlap} "
              f"({100 * overlap / max(len(te), 1):.1f}%)")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])   # usage: python eval\split_dataset.py <dedup file> <output folder>