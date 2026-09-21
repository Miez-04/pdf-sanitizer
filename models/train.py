"""
models/train.py

CLI entry point. Run from the project root:

    python -m models.train --data data/raw/latest_train_dataset.txt

Splits sentences (not tokens) into train/val so no sentence's tokens
leak across the split. Saves the best-val-F1 checkpoint plus the three
vocabs (needed to numericalize new text identically at inference) into
models/checkpoints/ (gitignored — see .gitignore, models/*.pt).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from seqeval.metrics import f1_score, classification_report
from torch.utils.data import DataLoader

from models.bilstm_crf import BiLSTMCRF
from models.dataset import PIIDataset, collate_fn, parse_iob2_file
from models.vocab import CharVocab, LabelVocab, WordVocab


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def train_val_split(sentences: list, val_ratio: float, seed: int):
    rng = random.Random(seed)
    shuffled = sentences[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:cut], shuffled[cut:]


def _skeleton(sent) -> str:
    out = []
    for tok, lab in zip(sent.tokens, sent.labels):
        if lab == "O":
            out.append(tok)
        elif lab.startswith("B-"):
            out.append(f"<{lab[2:]}>")
    return " ".join(out)


def skeleton_train_val_split(sentences: list, val_ratio: float, seed: int):
    """Groups sentences by skeleton (entity-masked structure) and
    assigns whole groups to train or val, so val never contains a
    template already seen in train — the honest generalization split
    diagnosed as missing from the original random sentence-level
    split (see conversation history: 72.9% of val sentences shared an
    exact skeleton with a train sentence under the naive split)."""
    from collections import defaultdict

    groups: dict[str, list] = defaultdict(list)
    for s in sentences:
        groups[_skeleton(s)].append(s)

    skeleton_keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(skeleton_keys)

    cut = int(len(skeleton_keys) * (1 - val_ratio))
    train_keys, val_keys = skeleton_keys[:cut], skeleton_keys[cut:]

    train_sents = [s for k in train_keys for s in groups[k]]
    val_sents = [s for k in val_keys for s in groups[k]]
    return train_sents, val_sents


def decode_batch_labels(
    label_ids: torch.Tensor, mask: torch.Tensor, label_vocab: LabelVocab
) -> list[list[str]]:
    """Gold labels, unpadded per-example, as strings — for seqeval."""
    out = []
    for row, m in zip(label_ids.tolist(), mask.tolist()):
        seq_len = sum(m)
        out.append([label_vocab.idx2label[i] for i in row[:seq_len]])
    return out


def evaluate(model, loader, label_vocab, device) -> tuple[float, str]:
    model.eval()
    all_gold, all_pred = [], []
    with torch.no_grad():
        for word_ids, char_ids, label_ids, mask in loader:
            word_ids, char_ids = word_ids.to(device), char_ids.to(device)
            label_ids, mask = label_ids.to(device), mask.to(device)

            pred_indices = model.decode(word_ids, char_ids, mask)
            pred_labels = [
                [label_vocab.idx2label[i] for i in seq] for seq in pred_indices
            ]
            gold_labels = decode_batch_labels(label_ids, mask, label_vocab)

            all_pred.extend(pred_labels)
            all_gold.extend(gold_labels)

    f1 = f1_score(all_gold, all_pred)
    report = classification_report(all_gold, all_pred, digits=4)
    return f1, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/latest_train_dataset.txt")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split", choices=["random", "skeleton"], default="skeleton",
        help="'skeleton' (default) holds out entire unseen sentence templates for "
             "val, giving an honest generalization score. 'random' is the original "
             "sentence-level split — kept only for reproducing the old, inflated number.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    sentences = parse_iob2_file(args.data)
    splitter = skeleton_train_val_split if args.split == "skeleton" else train_val_split
    train_sents, val_sents = splitter(sentences, args.val_ratio, args.seed)
    print(f"split={args.split}  train sentences: {len(train_sents)}  val sentences: {len(val_sents)}")

    # Vocabs are built from TRAIN only — building from val too would
    # leak val-only words into the model's known vocabulary and
    # overstate real generalization.
    train_tokens = [t for s in train_sents for t in s.tokens]
    all_labels = [l for s in sentences for l in s.labels]  # label SET must be global

    word_vocab = WordVocab.build(train_tokens)
    char_vocab = CharVocab.build(train_tokens)
    label_vocab = LabelVocab.build(all_labels)

    train_ds = PIIDataset(train_sents, word_vocab, char_vocab, label_vocab)
    val_ds = PIIDataset(val_sents, word_vocab, char_vocab, label_vocab)

    def _collate(batch):
        return collate_fn(batch, word_vocab.pad_id, char_vocab.pad_id, label_pad_id=0)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate
    )

    model = BiLSTMCRF(
        word_vocab_size=len(word_vocab),
        char_vocab_size=len(char_vocab),
        num_labels=len(label_vocab),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for word_ids, char_ids, label_ids, mask in train_loader:
            word_ids, char_ids = word_ids.to(device), char_ids.to(device)
            label_ids, mask = label_ids.to(device), mask.to(device)

            optimizer.zero_grad()
            loss = model.loss(word_ids, char_ids, label_ids, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_f1, val_report = evaluate(model, val_loader, label_vocab, device)
        print(f"epoch {epoch:2d}  train_loss={avg_loss:.4f}  val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "word_vocab": {
                        "token2idx": word_vocab.token2idx,
                        "idx2token": word_vocab.idx2token,
                    },
                    "char_vocab": {
                        "char2idx": char_vocab.char2idx,
                        "idx2char": char_vocab.idx2char,
                    },
                    "label_vocab": {
                        "label2idx": label_vocab.label2idx,
                        "idx2label": label_vocab.idx2label,
                    },
                    "val_f1": val_f1,
                    "epoch": epoch,
                },
                checkpoint_dir / "best_model.pt",
            )
            with open(checkpoint_dir / "best_report.txt", "w") as f:
                f.write(val_report)
            print(f"  -> new best (val_f1={val_f1:.4f}), checkpoint saved")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"no improvement for {args.patience} epochs, stopping early")
                break

    print(f"best val_f1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
