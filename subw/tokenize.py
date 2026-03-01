import os
import click
import random

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


@click.command()
@click.option(
    "--data_dir",
    default="data/processed/",
    help="Directory containing processed text files",
)
@click.option(
    "--out_dir", default="data/encoded/subw/", help="Directory to save encoded files"
)
@click.option("--vocab_size", default=1000, help="Vocabulary size")
def main(data_dir, out_dir, vocab_size):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    sonnets = []

    for filename in os.listdir(data_dir):
        if not filename.endswith(".txt"):
            continue

        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            text = f.read()
            sonnets.append(text)

    print(f"Loaded {len(sonnets)} sonnets.")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()

    special_tokens = ["[UNK]", "<SONNET>", "<STANZA>", "<END>"]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
    )

    tokenizer.train_from_iterator(sonnets, trainer)

    filtered_sonnets = []
    token_lengths = []

    for text in sonnets:
        text += "\n"
        ids = tokenizer.encode(text).ids
        if len(ids) < 10:
            print(f"Short sonnet ({len(ids)} tokens) found, skipping.")
            continue
        if len(ids) > 1000:
            print(f"Long sonnet ({len(ids)} tokens) found.")
        filtered_sonnets.append(text)
        token_lengths.append(len(ids))

    sonnets = filtered_sonnets

    token_lengths = np.array(token_lengths)

    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Average tokens per file: {token_lengths.mean():.2f}")
    print(f"Median tokens: {np.median(token_lengths)}")
    print(f"Min tokens: {token_lengths.min()}")
    print(f"Max tokens: {token_lengths.max()}")
    print(f"Std dev: {token_lengths.std():.2f}")

    print("---special tokens---")
    for token in special_tokens:
        print(f"{token}: {tokenizer.token_to_id(token)}")
    print("--------------------")

    random.shuffle(sonnets)
    n_train = int(len(sonnets) * 0.9)
    train_text = "".join(sonnets[:n_train])
    val_text = "".join(sonnets[n_train:])

    train_ids = tokenizer.encode(train_text).ids
    val_ids = tokenizer.encode(val_text).ids

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(out_dir, "train.bin"))
    val_ids.tofile(os.path.join(out_dir, "val.bin"))

    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))


if __name__ == "__main__":
    main()
