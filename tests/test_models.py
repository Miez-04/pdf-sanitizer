import torch
import pytest

from models.dataset import parse_iob2_file, PIIDataset, collate_fn, _split_compound_line
from models.vocab import WordVocab, CharVocab, LabelVocab
from models.bilstm_crf import BiLSTMCRF

REAL_DATA_PATH = "data/raw/latest_train_dataset.txt"


def _has_real_data() -> bool:
    from pathlib import Path

    return Path(REAL_DATA_PATH).exists()


pytestmark = pytest.mark.skipif(
    not _has_real_data(),
    reason="real training corpus not present in this environment",
)


# ---------------------------------------------------------------------
# Compound-token splitting (the "Tan Sri" quirk)
# ---------------------------------------------------------------------

def test_split_compound_line_single_word():
    assert _split_compound_line("Ahmad", "B-PERSON") == [("Ahmad", "B-PERSON")]


def test_split_compound_line_tan_sri_b_tag():
    result = _split_compound_line("Tan Sri", "B-PERSON")
    assert result == [("Tan", "B-PERSON"), ("Sri", "I-PERSON")]


def test_split_compound_line_tan_sri_o_tag():
    result = _split_compound_line("Tan Sri", "O")
    assert result == [("Tan", "O"), ("Sri", "O")]


# ---------------------------------------------------------------------
# Real-corpus parsing
# ---------------------------------------------------------------------

def test_parse_real_corpus_sentence_count():
    sentences = parse_iob2_file(REAL_DATA_PATH)
    # 4400 blank-line-separated blocks, verified via `grep -c '^$'`.
    assert len(sentences) == 4400


def test_parse_real_corpus_every_sentence_nonempty():
    sentences = parse_iob2_file(REAL_DATA_PATH)
    assert all(len(s) > 0 for s in sentences)
    assert all(len(s.tokens) == len(s.labels) for s in sentences)


def test_parse_real_corpus_tan_sri_actually_split():
    sentences = parse_iob2_file(REAL_DATA_PATH)
    assert not any(" " in tok for s in sentences for tok in s.tokens), (
        "a compound token slipped through un-split; inference-time "
        "tokenization (whitespace-split by pdf_ingestion) would mismatch"
    )


def test_parse_real_corpus_no_inline_nric_or_phone_continuation():
    """Dataset never tags I-NRIC/I-PHONE (confirmed via corpus audit) —
    both entity types are always single-token. If this ever fires, the
    label space / conflict-resolution assumptions need revisiting."""
    sentences = parse_iob2_file(REAL_DATA_PATH)
    labels = {l for s in sentences for l in s.labels}
    assert "I-NRIC" not in labels
    assert "I-PHONE" not in labels


# ---------------------------------------------------------------------
# Vocab + dataset + model, wired together on a small real slice
# ---------------------------------------------------------------------

@pytest.fixture
def small_setup():
    sentences = parse_iob2_file(REAL_DATA_PATH)[:20]
    tokens = [t for s in sentences for t in s.tokens]
    labels = [l for s in sentences for l in s.labels]

    word_vocab = WordVocab.build(tokens)
    char_vocab = CharVocab.build(tokens)
    label_vocab = LabelVocab.build(labels)

    ds = PIIDataset(sentences, word_vocab, char_vocab, label_vocab)
    return sentences, ds, word_vocab, char_vocab, label_vocab


def test_vocab_roundtrip_no_unk_on_seen_tokens(small_setup):
    sentences, ds, word_vocab, char_vocab, label_vocab = small_setup
    unk_id = word_vocab.token2idx["<UNK>"]
    for s in sentences:
        for tok in s.tokens:
            assert word_vocab.encode(tok) != unk_id


def test_dataset_item_shapes(small_setup):
    _, ds, word_vocab, char_vocab, label_vocab = small_setup
    word_ids, char_ids, label_ids = ds[0]
    seq_len = len(ds.sentences[0])
    assert word_ids.shape == (seq_len,)
    assert char_ids.shape == (seq_len, ds.max_word_len)
    assert label_ids.shape == (seq_len,)


def test_collate_fn_padding_and_mask(small_setup):
    _, ds, word_vocab, char_vocab, label_vocab = small_setup
    batch = [ds[i] for i in range(5)]
    word_ids, char_ids, label_ids, mask = collate_fn(
        batch, word_vocab.pad_id, char_vocab.pad_id, label_pad_id=0
    )
    max_len = max(len(ds.sentences[i]) for i in range(5))
    assert word_ids.shape == (5, max_len)
    assert mask.dtype == torch.bool
    for i in range(5):
        assert mask[i].sum().item() == len(ds.sentences[i])


def test_model_forward_backward_and_decode(small_setup):
    _, ds, word_vocab, char_vocab, label_vocab = small_setup
    batch = [ds[i] for i in range(5)]
    word_ids, char_ids, label_ids, mask = collate_fn(
        batch, word_vocab.pad_id, char_vocab.pad_id, label_pad_id=0
    )

    model = BiLSTMCRF(
        word_vocab_size=len(word_vocab),
        char_vocab_size=len(char_vocab),
        num_labels=len(label_vocab),
        word_emb_dim=16,
        char_emb_dim=8,
        char_hidden_dim=8,
        word_hidden_dim=16,
    )

    loss = model.loss(word_ids, char_ids, label_ids, mask)
    assert loss.item() > 0
    loss.backward()
    assert model.word_embedding.weight.grad is not None

    decoded = model.decode(word_ids, char_ids, mask)
    assert len(decoded) == 5
    assert [len(d) for d in decoded] == mask.sum(dim=1).tolist()
    for seq in decoded:
        assert all(0 <= idx < len(label_vocab) for idx in seq)


def test_loss_decreases_over_a_few_steps(small_setup):
    """Not a convergence test — just confirms gradients actually move
    the loss down on a handful of real sentences, catching wiring bugs
    (e.g. a detached tensor or wrong reduction) that a single forward
    pass wouldn't."""
    _, ds, word_vocab, char_vocab, label_vocab = small_setup
    batch = [ds[i] for i in range(10)]
    word_ids, char_ids, label_ids, mask = collate_fn(
        batch, word_vocab.pad_id, char_vocab.pad_id, label_pad_id=0
    )

    model = BiLSTMCRF(
        word_vocab_size=len(word_vocab),
        char_vocab_size=len(char_vocab),
        num_labels=len(label_vocab),
        word_emb_dim=16,
        char_emb_dim=8,
        char_hidden_dim=8,
        word_hidden_dim=16,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for _ in range(15):
        optimizer.zero_grad()
        loss = model.loss(word_ids, char_ids, label_ids, mask)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
