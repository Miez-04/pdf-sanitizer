"""
models.dataset

Parses the project's IOB2 training file (blank-line-separated
sentences, "<token(s)> <LABEL>" per line) and exposes it as a PyTorch
Dataset ready for batching.

Corpus quirk handled here (verified against latest_train_dataset.txt):
some lines carry a *compound* token field, e.g. "Tan Sri B-PERSON"
where "Tan Sri" is one honorific unit in the annotation tool's eyes.
At inference time, pdf_ingestion.extractor splits purely on
whitespace (PyMuPDF word tokens), so "Tan" and "Sri" arrive as two
separate Token objects. Training on the merged 3-column form would
teach the model a token granularity it will never see at inference,
silently degrading real-world accuracy. split_compound_tokens()
below normalizes every line to single-word granularity before
training, so train-time and inference-time tokenization match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import Dataset

from models.vocab import CharVocab, LabelVocab, WordVocab


@dataclass(slots=True)
class Sentence:
    tokens: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tokens)


def _split_compound_line(token_field: str, label: str) -> list[tuple[str, str]]:
    """
    Turn one raw corpus line into 1+ (word, label) pairs at
    single-whitespace-token granularity.

    - "Ahmad B-PERSON"      -> [("Ahmad", "B-PERSON")]
    - "Tan Sri B-PERSON"    -> [("Tan", "B-PERSON"), ("Sri", "I-PERSON")]
      (first sub-word keeps the original tag; any B- continues as I-
      of the same entity type, since they were one annotated span)
    - "Tan Sri O"           -> [("Tan", "O"), ("Sri", "O")]
    """
    words = token_field.split(" ")
    if len(words) == 1:
        return [(words[0], label)]

    pairs = [(words[0], label)]
    if label.startswith("B-"):
        continuation = "I-" + label[2:]
    else:
        continuation = label  # "O" or already "I-..." stays as-is
    pairs.extend((w, continuation) for w in words[1:])
    return pairs


def parse_iob2_file(path: str | Path) -> list[Sentence]:
    sentences: list[Sentence] = []
    current = Sentence()

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n").rstrip("\r")

            if not line.strip():
                if len(current) > 0:
                    sentences.append(current)
                    current = Sentence()
                continue

            *token_parts, label = line.split(" ")
            token_field = " ".join(token_parts)
            if not token_field:
                # Malformed line (e.g. only a label, no token) — skip
                # rather than silently inject an empty-string token.
                continue

            for word, tag in _split_compound_line(token_field, label):
                current.tokens.append(word)
                current.labels.append(tag)

    if len(current) > 0:
        sentences.append(current)

    return sentences


class PIIDataset(Dataset):
    """Numericalized (word_ids, char_ids, label_ids) triples, one per
    sentence. Padding/batching happens in collate_fn, not here, so a
    single sentence can still be inspected/debugged un-padded."""

    def __init__(
        self,
        sentences: list[Sentence],
        word_vocab: WordVocab,
        char_vocab: CharVocab,
        label_vocab: LabelVocab,
        max_word_len: int = 20,
    ):
        self.sentences = sentences
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.label_vocab = label_vocab
        self.max_word_len = max_word_len

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int):
        sent = self.sentences[idx]

        word_ids = torch.tensor(
            [self.word_vocab.encode(t) for t in sent.tokens], dtype=torch.long
        )
        label_ids = torch.tensor(
            [self.label_vocab.encode(l) for l in sent.labels], dtype=torch.long
        )

        # Char ids: (seq_len, max_word_len), truncated/padded per word.
        char_pad = self.char_vocab.pad_id
        char_rows = []
        for tok in sent.tokens:
            ids = self.char_vocab.encode_word(tok)[: self.max_word_len]
            ids = ids + [char_pad] * (self.max_word_len - len(ids))
            char_rows.append(ids)
        char_ids = torch.tensor(char_rows, dtype=torch.long)

        return word_ids, char_ids, label_ids


def collate_fn(batch, word_pad_id: int, char_pad_id: int, label_pad_id: int = 0):
    """
    Pads a list of (word_ids, char_ids, label_ids) to the batch's max
    sequence length. Returns:
      word_ids:  torch.LongTensor (batch, max_seq_len)
      char_ids:  torch.LongTensor (batch, max_seq_len, max_word_len)
      label_ids: torch.LongTensor (batch, max_seq_len)
      mask:      torch.BoolTensor (batch, max_seq_len) — True where real token
    """
    max_seq_len = max(w.size(0) for w, _, _ in batch)
    max_word_len = batch[0][1].size(1)
    batch_size = len(batch)

    word_ids = torch.full((batch_size, max_seq_len), word_pad_id, dtype=torch.long)
    char_ids = torch.full(
        (batch_size, max_seq_len, max_word_len), char_pad_id, dtype=torch.long
    )
    label_ids = torch.full((batch_size, max_seq_len), label_pad_id, dtype=torch.long)
    mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

    for i, (w, c, l) in enumerate(batch):
        seq_len = w.size(0)
        word_ids[i, :seq_len] = w
        char_ids[i, :seq_len, :] = c
        label_ids[i, :seq_len] = l
        mask[i, :seq_len] = True

    return word_ids, char_ids, label_ids, mask
