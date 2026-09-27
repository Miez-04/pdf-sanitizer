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
        decides which of these predictions actually get applied.
        Thin wrapper over predict_page_with_confidence() for callers
        that only want labels."""
        labels, _confidences = self.predict_page_with_confidence(page)
        return labels

    def predict_page_with_confidence(
        self, page: PageTokens
    ) -> tuple[list[str], list[float]]:
        """Returns (labels, confidences) — one IOB2 label AND one float
        in [0, 1] per token, in order. confidence[i] is the CRF
        marginal probability of the CHOSEN label at position i (i.e.
        P(label_i = decoded_label_i | whole sequence)) — it already
        accounts for what neighboring tokens were predicted, via the
        forward-backward pass in models.bilstm_crf.BiLSTMCRF.marginals().
        This is what pipeline.conflict_resolution uses to decide
        whether a model-only prediction is trustworthy enough to reach
        redaction un-gated, or should be downgraded — see
        MIN_MODEL_CONFIDENCE / _reject_low_confidence_model_predictions
        there."""
        if not page.tokens:
            return [], []

        word_ids, char_ids, mask = self._encode_page(page)
        with torch.no_grad():
            decoded = self.model.decode(word_ids, char_ids, mask)
            marginals = self.model.marginals(word_ids, char_ids, mask)  # (1, S, L)

        label_indices = decoded[0]
        labels = [self.label_vocab.idx2label[i] for i in label_indices]
        confidences = [
            marginals[0, t, label_idx].item()
            for t, label_idx in enumerate(label_indices)
        ]
        return labels, confidences

    def predict_document(self, document: DocumentTokens) -> list[list[str]]:
        """Labels only — back-compat wrapper, see predict_page."""
        labels_by_page, _confidences_by_page = self.predict_document_with_confidence(document)
        return labels_by_page

    def predict_document_with_confidence(
        self, document: DocumentTokens
    ) -> tuple[list[list[str]], list[list[float]]]:
        labels_by_page: list[list[str]] = []
        confidences_by_page: list[list[float]] = []
        for page in document.pages:
            labels, confidences = self.predict_page_with_confidence(page)
            labels_by_page.append(labels)
            confidences_by_page.append(confidences)
        return labels_by_page, confidences_by_page
