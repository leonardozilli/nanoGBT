import json
import os
import random

import click
import numpy as np
from common.tokenizer import CharTokenizer

SPECIAL_TOKEN_MAP = {
    "<SONNET>": "¶",  # sos
    "\n\n<STANZA>\n\n": "\n\n",  # sep
    "<END>": "§",  # eos
}


@click.command()
@click.option(
    "--data_dir",
    default="data/processed/",
    help="Directory containing processed text files",
)
@click.option(
    "--out_dir", default="data/encoded/char/", help="Directory to save encoded files"
)
def main(data_dir, out_dir):
    chars = set()
    stoi = {}
    itos = {}

    text_content = ""
    sonnets = []
    lengths = []

    for filename in os.listdir(data_dir):
        if not filename.endswith(".txt"):
            continue

        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            text = f.read()
            for token, char in SPECIAL_TOKEN_MAP.items():
                text = text.replace(token, char)
            text_content += text + "\n"
            sonnets.append(text)
            lengths.append(len(text))

    print(f"length of dataset in characters: {len(text_content):,}")
    chars = sorted(list(set(text_content)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    print("all unique characters:", repr("".join(chars)))
    print(f"vocab size: {vocab_size:,}")
    print(
        f"number of sonnets: {len(sonnets):,}, average length: {np.mean(lengths):.2f} chars"
    )

    random.shuffle(sonnets)
    n_train = int(len(sonnets) * 0.9)
    train_text = "\n".join(sonnets[:n_train])
    val_text = "\n".join(sonnets[n_train:])

    tokenizer = CharTokenizer()
    tokenizer.itos = itos
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(out_dir, "train.bin"))
    val_ids.tofile(os.path.join(out_dir, "val.bin"))

    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
        "special_tokens": {
            "SOS": SPECIAL_TOKEN_MAP["<SONNET>"],
            "SEP": SPECIAL_TOKEN_MAP["\n\n<STANZA>\n\n"],
            "EOS": SPECIAL_TOKEN_MAP["<END>"],
        },
    }
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


if __name__ == "__main__":
    main()


# length of dataset in characters: 1,341,250
# all unique characters: '\n !(),-.236:;<>?ABCDEFGHIJLMNOPQRSTUVYZabcdefghijlmnopqrstuvwxz«»ÀÈÉÊàáèéìíïòóôöùúăēěŌ—’'
# vocab size: 90
# train has 1,205,060 tokens
# val has 136,188 tokens
