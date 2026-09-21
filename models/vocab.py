"""
models.vocab

Three independent vocabularies, all built from the training corpus
itself (no external embeddings assumed — offline system, no internet
access at train or inference time per the FYP's local-only constraint):

- WordVocab:  lowercased surface form -> int, for the word embedding table.
- CharVocab:  single character -> int, for the char-BiLSTM feature that
              helps with OOV Malay/English-mixed names and numeric
              patterns (NRIC/phone digit shapes) the word embedding
              alone won't generalize to.
- LabelVocab: IOB2 tag string -> int, e.g. "B-PERSON" -> 3.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass(slots=True)
class WordVocab:
    token2idx: dict[str, int] = field(default_factory=dict)
    idx2token: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, tokens: Iterable[str], min_freq: int = 1) -> "WordVocab":
        counts = Counter(t.lower() for t in tokens)
        vocab = cls()
        vocab.idx2token = [PAD_TOKEN, UNK_TOKEN] + sorted(
            tok for tok, c in counts.items() if c >= min_freq
        )
        vocab.token2idx = {tok: i for i, tok in enumerate(vocab.idx2token)}
        return vocab

    def encode(self, token: str) -> int:
        return self.token2idx.get(token.lower(), self.token2idx[UNK_TOKEN])

    @property
    def pad_id(self) -> int:
        return self.token2idx[PAD_TOKEN]

    def __len__(self) -> int:
        return len(self.idx2token)


@dataclass(slots=True)
class CharVocab:
    char2idx: dict[str, int] = field(default_factory=dict)
    idx2char: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, tokens: Iterable[str]) -> "CharVocab":
        chars: set[str] = set()
        for tok in tokens:
            chars.update(tok)
        vocab = cls()
        vocab.idx2char = [PAD_TOKEN, UNK_TOKEN] + sorted(chars)
        vocab.char2idx = {c: i for i, c in enumerate(vocab.idx2char)}
        return vocab

    def encode_word(self, token: str) -> list[int]:
        unk = self.char2idx[UNK_TOKEN]
        return [self.char2idx.get(c, unk) for c in token]

    @property
    def pad_id(self) -> int:
        return self.char2idx[PAD_TOKEN]

    def __len__(self) -> int:
        return len(self.idx2char)


@dataclass(slots=True)
class LabelVocab:
    label2idx: dict[str, int] = field(default_factory=dict)
    idx2label: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, labels: Iterable[str]) -> "LabelVocab":
        # No PAD/UNK here on purpose: pytorch-crf needs a dense, exact
        # label space (a stray UNK label would be a silent modelling
        # bug, not something to fall back on). Padding is handled via
        # a boolean mask in the dataset/collate step instead.
        vocab = cls()
        vocab.idx2label = sorted(set(labels))
        vocab.label2idx = {lab: i for i, lab in enumerate(vocab.idx2label)}
        return vocab

    def encode(self, label: str) -> int:
        return self.label2idx[label]

    def __len__(self) -> int:
        return len(self.idx2label)
