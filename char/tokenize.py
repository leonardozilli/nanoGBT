import json
import os
import random
import sys
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)

# ruff: noqa: E402
import click
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from common.tokenizer import CharTokenizer


SPECIAL_TOKEN_MAP = {
    "<SONNET>": "¶",  # bos
    "\n\n<STANZA>\n\n": "\n\n",  # sep
    "<END>": "§",  # eos
}


def create_sonnets_dataset(data_dir) -> tuple[list[str], str]:
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
            lengths.append(len(text))
            sonnets.append(text)
    print(
        f"number of sonnets: {len(sonnets):,}",
        f"average length: {np.mean(lengths):.2f} chars",
        f"length of dataset in characters: {len(text_content):,}",
    )

    return sonnets, text_content


def load_pretrain_data(parquet_file):
    """Load pretrain texts from parquet file"""
    table = pq.read_table(parquet_file, columns=["text"])
    texts = table["text"].to_pylist()
    processed_texts = []
    for text in texts:
        for token, char in SPECIAL_TOKEN_MAP.items():
            text = text.replace(token, char)
        processed_texts.append(text)
    return processed_texts


def build_tokenizer(texts):
    """Build char tokenizer"""
    all_text = "\n".join(texts)
    chars = sorted(list(set(all_text)))
    tokenizer = CharTokenizer()
    tokenizer.itos = {i: ch for i, ch in enumerate(chars)}
    tokenizer.stoi = {ch: i for i, ch in enumerate(chars)}

    return tokenizer


def encode_and_save(tokenizer, texts, out_file):
    """Encode texts streaming to .bin"""
    total_tokens = 0
    with open(out_file, "wb") as f:
        for i, text in tqdm(
            enumerate(texts),
            total=len(texts),
            desc=f"Encoding to {os.path.join(os.path.dirname(out_file), os.path.basename(out_file))}",
        ):
            chunk_text = text if i == 0 else "\n" + text

            ids = tokenizer.encode(chunk_text)

            ids_arr = np.array(ids, dtype=np.uint16)
            f.write(ids_arr.tobytes())

            total_tokens += len(ids_arr)

    return total_tokens


def save_dataset_metadata(out_dir, dataset_name, vocab_size, split_sizes):
    metadata = {
        "dataset": dataset_name,
        "vocab_size": vocab_size,
        "splits": split_sizes,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)


@click.command()
@click.option(
    "--sonnet-dir", default="data/processed/", help="Directory with sonnet .txt files"
)
@click.option(
    "--pretrain-data", default=None, help="Path to pretrain data parquet file"
)
@click.option(
    "--out-dir",
    default="data/encoded/sonnets",
    help="Directory to save encoded files",
)
def main(sonnet_dir, pretrain_data, out_dir):
    np.random.seed(42)
    random.seed(42)
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Load sonnets
    sonnets, _ = create_sonnets_dataset(sonnet_dir)
    print(f"Loaded {len(sonnets):,} sonnets")

    if pretrain_data:
        # --- 2. Load pretrain data if provided
        pretrain_texts = load_pretrain_data(pretrain_data)
        print(f"Loaded {len(pretrain_texts):,} pretrain texts")
        tokenizer_texts = pretrain_texts + sonnets
    else:
        tokenizer_texts = sonnets

    # --- 3. Build tokenizer
    tokenizer = build_tokenizer(tokenizer_texts)
    vocab_size = len(tokenizer.itos)
    print(f"Vocabulary size: {vocab_size:,} chars")
    print("Characters:", repr("".join(tokenizer.itos.values())))

    random.shuffle(sonnets)
    n_train = int(len(sonnets) * 0.9)
    sonnet_train = sonnets[:n_train]
    sonnet_val = sonnets[n_train:]

    # --- 4. Encode sonnets
    sonnets_out_dir = os.path.join(out_dir, "sonnets") if pretrain_data else out_dir
    os.makedirs(sonnets_out_dir, exist_ok=True)
    train_chars = encode_and_save(
        tokenizer, sonnet_train, os.path.join(sonnets_out_dir, "train.bin")
    )
    val_chars = encode_and_save(
        tokenizer, sonnet_val, os.path.join(sonnets_out_dir, "val.bin")
    )
    print(f"Sonnet train: {train_chars:,} tokens, val: {val_chars:,} tokens")
    save_dataset_metadata(
        sonnets_out_dir,
        "sonnets",
        vocab_size,
        {"train_tokens": train_chars, "val_tokens": val_chars},
    )

    if pretrain_data:
        # --- 5. Encode pretrain data if provided
        random.shuffle(pretrain_texts)
        n_train = int(len(pretrain_texts) * 0.9)
        pretrain_train = pretrain_texts[:n_train]
        pretrain_val = pretrain_texts[n_train:]

        pretrain_out_dir = os.path.join(out_dir, "pretrain")
        os.makedirs(pretrain_out_dir, exist_ok=True)

        train_chars = encode_and_save(
            tokenizer, pretrain_train, os.path.join(pretrain_out_dir, "train.bin")
        )
        val_chars = encode_and_save(
            tokenizer, pretrain_val, os.path.join(pretrain_out_dir, "val.bin")
        )
        print(f"Pretrain train: {train_chars:,} tokens, val: {val_chars:,} tokens")
        save_dataset_metadata(
            pretrain_out_dir,
            "pretrain",
            vocab_size,
            {"train_tokens": train_chars, "val_tokens": val_chars},
        )

    # --- 6. Save common vocab
    vocab = {
        "vocab_size": vocab_size,
        "itos": tokenizer.itos,
        "stoi": tokenizer.stoi,
        "special_tokens": {
            "BOS": SPECIAL_TOKEN_MAP["<SONNET>"],
            "SEP": SPECIAL_TOKEN_MAP["\n\n<STANZA>\n\n"],
            "EOS": SPECIAL_TOKEN_MAP["<END>"],
        },
    }

    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    print(f"Tokenizer vocab saved to {os.path.join(out_dir, 'vocab.json')}")


if __name__ == "__main__":
    main()
