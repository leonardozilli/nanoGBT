import json
from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer

SPECIAL_TOKENS = {
    "BOS": "<SONNET>",
    "SEP": "<STANZA>",
    "EOS": "<END>",
}


class CharTokenizer:
    def __init__(self, path: str | None = None):
        self.kind = "char"
        self.path = path
        self.special_tokens = {}
        self.itos = {}
        self.stoi = {}
        self.vocab_size = 0

        if path:
            with open(path, "r") as f:
                vocab = json.load(f)

            self.special_tokens = vocab["special_tokens"]
            self.itos = {int(k): v for k, v in vocab["itos"].items()}
            self.stoi = vocab["stoi"]
            self.vocab_size = len(self.itos)

    def encode(self, text: str):
        return [self.stoi[c] for c in text]

    def decode(self, batch: list, skip_special_tokens: bool = True):
        if isinstance(batch, int):
            batch = [batch]

        tokens = (self.itos[i] for i in batch)

        if skip_special_tokens:
            special = set(self.special_tokens.values())
            tokens = (tok for tok in tokens if tok not in special)

        return "".join(tokens)


class BPETokenizer:
    def __init__(self, path: str):
        self.kind = "bpe"
        self.path = path
        self.tokenizer = Tokenizer.from_file(path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.special_tokens = SPECIAL_TOKENS.copy()

    def encode(self, text: str):
        return self.tokenizer.encode(text).ids

    def decode(self, batch: list, skip_special_tokens: bool = True):
        return self.tokenizer.decode(batch, skip_special_tokens=skip_special_tokens)


class UnigramTokenizer:
    def __init__(self, path: str):
        self.kind = "unigram"
        self.path = path

        self.special_tokens = {
            "UNK": "[UNK]",
            "BOS": "<SONNET>",
            "SEP": "<STANZA>",
            "EOS": "<END>",
        }

        if path.endswith(".model"):
            import sentencepiece as spm

            spm.set_min_log_level(2)  # suppress warnings

            self.backend = "sentencepiece"
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(path)
            self.vocab_size = self.sp.vocab_size()
        else:
            self.backend = "hf"
            print(path)
            self.tokenizer = Tokenizer.from_file(path)
            self.vocab_size = self.tokenizer.get_vocab_size()

    def _flatten_ids(self, batch: Iterable[int] | Iterable[Iterable[int]]):
        ids = list(batch)
        if ids and isinstance(ids[0], list):
            return [item for sublist in ids for item in sublist]
        return ids

    def encode(self, text: str):
        if self.backend == "sentencepiece":
            return self.sp.EncodeAsIds(text)
        return self.tokenizer.encode(text).ids

    def decode(self, batch: list, skip_special_tokens: bool = True):
        ids = self._flatten_ids(batch)

        if self.backend == "sentencepiece":
            text = self.sp.Decode(ids)
            if skip_special_tokens:
                for token in self.special_tokens.values():
                    text = text.replace(token, "")
            return text

        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


def load_tokenizer(
    tokenizer_path: Path, tokenizer_type: str
) -> CharTokenizer | BPETokenizer | UnigramTokenizer:
    tokenizer_path_str = str(tokenizer_path)

    valid_tokenizer_types = {"char", "bpe", "unigram", "auto"}
    if tokenizer_type not in valid_tokenizer_types:
        raise ValueError(
            "Invalid tokenizer_type "
            f"{tokenizer_type!r}. Expected one of: {sorted(valid_tokenizer_types)}"
        )

    if tokenizer_type == "char":
        return CharTokenizer(tokenizer_path_str)
    if tokenizer_type == "bpe":
        return BPETokenizer(tokenizer_path_str)
    if tokenizer_type == "unigram":
        return UnigramTokenizer(tokenizer_path_str)

    if tokenizer_path.name == "vocab.json":
        return CharTokenizer(tokenizer_path_str)
    if tokenizer_path.suffix == ".model":
        return UnigramTokenizer(tokenizer_path_str)
    return BPETokenizer(tokenizer_path_str)
