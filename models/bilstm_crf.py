"""
models.bilstm_crf

Architecture (Lample et al. 2016 style, minus pretrained embeddings —
this system trains from scratch on the project's own corpus, offline):

    char_ids (batch, seq_len, max_word_len)
        -> char embedding -> char-BiLSTM -> last hidden state per word
        -> char_repr (batch, seq_len, 2*char_hidden_dim)

    word_ids (batch, seq_len)
        -> word embedding -> word_repr (batch, seq_len, word_emb_dim)

    concat(char_repr, word_repr) -> word-BiLSTM -> Linear -> emissions
        (batch, seq_len, num_labels)

    emissions + mask -> CRF (pytorch-crf) -> loss (train) / best path (decode)

The char-BiLSTM is what lets the model generalize to unseen Malay/
English-mixed names and to numeric-shape patterns (NRIC/phone digit
runs) it hasn't memorized as whole words — this matters far more here
than in a typical English-only NER task, given the code-mixed corpus.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchcrf import CRF


class CharBiLSTM(nn.Module):
    """Encodes each word's character sequence into a fixed-size vector
    (concatenation of the final forward and backward hidden states)."""

    def __init__(self, char_vocab_size: int, char_emb_dim: int, char_hidden_dim: int):
        super().__init__()
        self.char_embedding = nn.Embedding(
            char_vocab_size, char_emb_dim, padding_idx=0
        )
        self.char_lstm = nn.LSTM(
            input_size=char_emb_dim,
            hidden_size=char_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # char_ids: (batch, seq_len, max_word_len) -> flatten words into
        # the batch dim so nn.LSTM processes (batch*seq_len, max_word_len).
        batch_size, seq_len, max_word_len = char_ids.shape
        flat = char_ids.view(batch_size * seq_len, max_word_len)

        embedded = self.char_embedding(flat)  # (B*S, W, char_emb_dim)
        _, (h_n, _) = self.char_lstm(embedded)
        # h_n: (2, B*S, char_hidden_dim) -> forward/backward final states
        char_repr = torch.cat([h_n[0], h_n[1]], dim=-1)  # (B*S, 2*char_hidden_dim)

        return char_repr.view(batch_size, seq_len, -1)


class BiLSTMCRF(nn.Module):
    def __init__(
        self,
        word_vocab_size: int,
        char_vocab_size: int,
        num_labels: int,
        word_emb_dim: int = 100,
        char_emb_dim: int = 25,
        char_hidden_dim: int = 25,
        word_hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.word_embedding = nn.Embedding(
            word_vocab_size, word_emb_dim, padding_idx=0
        )
        self.char_bilstm = CharBiLSTM(char_vocab_size, char_emb_dim, char_hidden_dim)

        combined_dim = word_emb_dim + 2 * char_hidden_dim
        self.word_bilstm = nn.LSTM(
            input_size=combined_dim,
            hidden_size=word_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden2label = nn.Linear(2 * word_hidden_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def _emissions(self, word_ids: torch.Tensor, char_ids: torch.Tensor) -> torch.Tensor:
        word_repr = self.word_embedding(word_ids)         # (B, S, word_emb_dim)
        char_repr = self.char_bilstm(char_ids)             # (B, S, 2*char_hidden_dim)
        combined = torch.cat([word_repr, char_repr], dim=-1)
        combined = self.dropout(combined)

        lstm_out, _ = self.word_bilstm(combined)            # (B, S, 2*word_hidden_dim)
        lstm_out = self.dropout(lstm_out)
        return self.hidden2label(lstm_out)                  # (B, S, num_labels)

    def loss(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        label_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood over the batch (pytorch-crf returns
        log-likelihood; we negate + reduce by mean for a stable, batch-
        size-independent training signal)."""
        emissions = self._emissions(word_ids, char_ids)
        log_likelihood = self.crf(emissions, label_ids, mask=mask, reduction="mean")
        return -log_likelihood

    def decode(
        self, word_ids: torch.Tensor, char_ids: torch.Tensor, mask: torch.Tensor
    ) -> list[list[int]]:
        """Viterbi-decoded best label-index sequence per example (each
        inner list's length equals that example's true, unpadded
        seq_len, since CRF.decode respects the mask)."""
        emissions = self._emissions(word_ids, char_ids)
        return self.crf.decode(emissions, mask=mask)
