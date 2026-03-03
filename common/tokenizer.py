import json
from tokenizers import Tokenizer


class CharTokenizer:
    def __init__(self, vocab_path: str | None = None):
        self.special_tokens = {}
        self.itos = {}
        self.stoi = {}
        self.vocab_size = 0

        if vocab_path:
            with open(vocab_path, "r") as f:
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


class SubwordTokenizer:
    def __init__(self, tokenizer_path: str):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.special_tokens = {
            "BOS": "<SONNET>",
            "SEP": "<STANZA>",
            "EOS": "<END>",
        }

    def encode(self, text: str):
        return self.tokenizer.encode(text).ids

    def decode(self, batch: list, skip_special_tokens: bool = True):
        return self.tokenizer.decode(batch, skip_special_tokens=skip_special_tokens)
