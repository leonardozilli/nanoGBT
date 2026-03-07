import json
from pathlib import Path

import pytest

from common import tokenizer as tokenizer_module
from common.tokenizer import CharTokenizer, UnigramTokenizer, load_tokenizer


def test_char_tokenizer_encode_decode_with_special_tokens(tmp_path):
    vocab = {
        "special_tokens": {"BOS": "<SONNET>", "EOS": "<END>"},
        "itos": {
            "0": "<SONNET>",
            "1": "a",
            "2": "b",
            "3": "<END>",
        },
        "stoi": {
            "a": 1,
            "b": 2,
        },
    }

    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

    tokenizer = CharTokenizer(str(vocab_path))

    assert tokenizer.encode("ab") == [1, 2]
    assert (
        tokenizer.decode([0, 1, 2, 3], skip_special_tokens=False) == "<SONNET>ab<END>"
    )
    assert tokenizer.decode([0, 1, 2, 3], skip_special_tokens=True) == "ab"


def test_unigram_flatten_ids_handles_nested_and_flat_lists():
    tokenizer = UnigramTokenizer.__new__(UnigramTokenizer)

    assert tokenizer._flatten_ids([[1, 2], [3]]) == [1, 2, 3]
    assert tokenizer._flatten_ids([4, 5, 6]) == [4, 5, 6]


class _DummyCharTokenizer:
    def __init__(self, path):
        self.kind = "char"
        self.path = path


class _DummyBPETokenizer:
    def __init__(self, path):
        self.kind = "bpe"
        self.path = path


class _DummyUnigramTokenizer:
    def __init__(self, path):
        self.kind = "unigram"
        self.path = path


def test_load_tokenizer_dispatches_by_explicit_type(monkeypatch):
    monkeypatch.setattr(tokenizer_module, "CharTokenizer", _DummyCharTokenizer)
    monkeypatch.setattr(tokenizer_module, "BPETokenizer", _DummyBPETokenizer)
    monkeypatch.setattr(tokenizer_module, "UnigramTokenizer", _DummyUnigramTokenizer)

    char_tok = load_tokenizer(Path("foo/vocab.json"), "char")
    bpe_tok = load_tokenizer(Path("foo/tokenizer.json"), "bpe")
    unigram_tok = load_tokenizer(Path("foo/model.model"), "unigram")

    assert char_tok.kind == "char"
    assert bpe_tok.kind == "bpe"
    assert unigram_tok.kind == "unigram"
    assert char_tok.path == "foo/vocab.json"
    assert bpe_tok.path == "foo/tokenizer.json"
    assert unigram_tok.path == "foo/model.model"


def test_load_tokenizer_infers_from_path_when_type_unknown(monkeypatch):
    monkeypatch.setattr(tokenizer_module, "CharTokenizer", _DummyCharTokenizer)
    monkeypatch.setattr(tokenizer_module, "BPETokenizer", _DummyBPETokenizer)
    monkeypatch.setattr(tokenizer_module, "UnigramTokenizer", _DummyUnigramTokenizer)

    inferred_char = load_tokenizer(Path("foo/vocab.json"), "auto")
    inferred_unigram = load_tokenizer(Path("foo/tokenizer.model"), "auto")
    inferred_bpe = load_tokenizer(Path("foo/tokenizer.json"), "auto")

    assert inferred_char.kind == "char"
    assert inferred_unigram.kind == "unigram"
    assert inferred_bpe.kind == "bpe"


def test_load_tokenizer_invalid_type():
    with pytest.raises(ValueError, match="Invalid tokenizer_type"):
        load_tokenizer(Path("foo/tokenizer.json"), "wordpiece")
