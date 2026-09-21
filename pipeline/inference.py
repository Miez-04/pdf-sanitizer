"""
pipeline.inference

Loads a models/checkpoints/*.pt checkpoint (model weights + the three
vocabs it was trained with — see models/train.py) and runs Tier-2
inference over a DocumentTokens produced by pdf_ingestion.

Critical detail: vocabs are rebuilt FROM THE CHECKPOINT, not re-derived
from a data file. word_vocab/char_vocab index assignments are only
meaningful relative to the specific model_state they were saved with;
rebuilding them from a (possibly different, possibly reordered) corpus
file would silently misalign word IDs with embedding rows.
"""

from __future__ import annotations

from pathlib import Path

import torch

from models.bilstm_crf import BiLSTMCRF
from models.vocab import CharVocab, LabelVocab, WordVocab
from pdf_ingestion.schema import DocumentTokens, PageTokens

MODEL_SOURCE = "model"


def _vocab_from_checkpoint_dict(cls, saved: dict):
    vocab = cls()
    for key, value in saved.items():
        setattr(vocab, key, value)
    return vocab


class TierTwoPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        checkpoint = torch.load(
            str(checkpoint_path), map_location=self.device, weights_only=False
        )

        self.word_vocab = _vocab_from_checkpoint_dict(WordVocab, checkpoint["word_vocab"])
        self.char_vocab = _vocab_from_checkpoint_dict(CharVocab, checkpoint["char_vocab"])
        self.label_vocab = _vocab_from_checkpoint_dict(LabelVocab, checkpoint["label_vocab"])

        self.model = BiLSTMCRF(
            word_vocab_size=len(self.word_vocab),
            char_vocab_size=len(self.char_vocab),
            num_labels=len(self.label_vocab),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.val_f1 = checkpoint.get("val_f1")
        self.trained_epoch = checkpoint.get("epoch")

    def _encode_page(self, page: PageTokens, max_word_len: int = 20):
        word_ids = torch.tensor(
            [self.word_vocab.encode(t.text) for t in page.tokens], dtype=torch.long
        )
        char_pad = self.char_vocab.pad_id
        rows = []
        for t in page.tokens:
            ids = self.char_vocab.encode_word(t.text)[:max_word_len]
            ids = ids + [char_pad] * (max_word_len - len(ids))
            rows.append(ids)
        char_ids = torch.tensor(rows, dtype=torch.long)
        mask = torch.ones(len(page.tokens), dtype=torch.bool)

        return (
            word_ids.unsqueeze(0).to(self.device),
            char_ids.unsqueeze(0).to(self.device),
            mask.unsqueeze(0).to(self.device),
        )

    def predict_page(self, page: PageTokens) -> list[str]:
        """Returns one IOB2 label string per token in page.tokens, in
        order. Does NOT mutate tokens — pipeline.conflict_resolution
        decides which of these predictions actually get applied."""
        if not page.tokens:
            return []

        word_ids, char_ids, mask = self._encode_page(page)
        with torch.no_grad():
            decoded = self.model.decode(word_ids, char_ids, mask)
        label_indices = decoded[0]
        return [self.label_vocab.idx2label[i] for i in label_indices]

    def predict_document(self, document: DocumentTokens) -> list[list[str]]:
        return [self.predict_page(page) for page in document.pages]
