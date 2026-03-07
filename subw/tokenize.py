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
import sentencepiece as spm
import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE, Unigram
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer, UnigramTrainer

spm.set_min_log_level(2)  # suppress warnings


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_sonnets(data_dir):
    sonnets = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                sonnets.append(f.read())
    print(f"Loaded {len(sonnets)} sonnets.")
    return sonnets


def split_data(data, split_ratio=0.9):
    random.shuffle(data)
    n_train = int(len(data) * split_ratio)
    return data[:n_train], data[n_train:]


def train_sentencepiece(
    texts, out_dir, vocab_size, special_tokens, model_name="spm_model"
):
    temp_input_path = os.path.join(out_dir, "spm_temp_input.txt")
    with open(temp_input_path, "w", encoding="utf-8") as f:
        for text in texts:
            safe_text = text.replace("\r\n", "\n").replace("\n", "<NEWLINE>")
            f.write(safe_text + "\n\n")

    model_prefix = os.path.join(out_dir, model_name)
    spm.SentencePieceTrainer.Train(
        input=temp_input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        user_defined_symbols=special_tokens + ["<NEWLINE>"],
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
    )
    os.remove(temp_input_path)

    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")
    return sp


def augment_train(sp, train_list, n_times=10, nbest_size=64, alpha=0.1):
    train_ids = []
    for _ in range(n_times):
        for text in train_list:
            safe_text = (
                text.replace("\r\n", "\n").replace("\n", "<NEWLINE>") + "<NEWLINE>"
            )
            ids = sp.SampleEncodeAsIds(safe_text, nbest_size=nbest_size, alpha=alpha)
            train_ids.extend(ids)
    return np.array(train_ids, dtype=np.uint16)


def train_hf_tokenizer(sonnets, tokenizer_type, vocab_size, special_tokens):
    if tokenizer_type == "unigram":
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(
            vocab_size=vocab_size, special_tokens=special_tokens, unk_token="[UNK]"
        )
    else:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            vocab_size=vocab_size, min_frequency=2, special_tokens=special_tokens
        )

    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.train_from_iterator(sonnets, trainer)
    return tokenizer


def encode_and_save_hf(tokenizer, train_list, val_list, out_dir):
    train_text = "".join(train_list)
    val_text = "".join(val_list)
    train_ids = np.array(tokenizer.encode(train_text).ids, dtype=np.uint16)
    val_ids = np.array(tokenizer.encode(val_text).ids, dtype=np.uint16)
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))

    train_ids.tofile(os.path.join(out_dir, "train.bin"))
    val_ids.tofile(os.path.join(out_dir, "val.bin"))

    print(f"Static Train tokens: {len(train_ids):,}")
    print(f"Static Val tokens: {len(val_ids):,}")


@click.command()
@click.option(
    "--data-dir",
    default="data/processed/",
    help="Directory containing processed text files",
)
@click.option(
    "--out-dir", default="data/encoded/subw/", help="Directory to save encoded files"
)
@click.option("--vocab-size", default=2000, help="Vocabulary size")
@click.option(
    "--tokenizer-type",
    type=click.Choice(["unigram", "bpe"], case_sensitive=False),
    default="unigram",
    show_default=True,
)
@click.option(
    "--regularization", is_flag=True, help="Enable SentencePiece subword regularization"
)
@click.option("--nbest-size", default=64, help="N-best size for subword regularization")
@click.option(
    "--alpha", default=0.2, help="Sampling temperature for subword regularization"
)
@click.option("--seed", default=42, help="Random seed")
def main(
    data_dir,
    out_dir,
    vocab_size,
    tokenizer_type,
    regularization,
    nbest_size,
    alpha,
    seed,
):
    set_seed(seed)
    tokenizer_type = tokenizer_type.lower()
    if tokenizer_type == "bpe" and regularization:
        raise click.ClickException(
            "--regularization is only supported with --tokenizer-type unigram"
        )

    os.makedirs(out_dir, exist_ok=True)
    special_tokens = ["[UNK]", "<SONNET>", "<STANZA>", "<END>"]

    sonnets = load_sonnets(data_dir)
    train_list, val_list = split_data(sonnets)
    if tokenizer_type == "unigram" and regularization:
        sp = train_sentencepiece(sonnets, out_dir, vocab_size, special_tokens)

        train_ids = augment_train(
            sp, train_list, n_times=10, nbest_size=nbest_size, alpha=alpha
        )

        val_ids = []
        for text in val_list:
            safe_text = (
                text.replace("\r\n", "\n").replace("\n", "<NEWLINE>") + "<NEWLINE>"
            )
            val_ids.extend(sp.EncodeAsIds(safe_text))
        val_ids = np.array(val_ids, dtype=np.uint16)

        train_ids.tofile(os.path.join(out_dir, "train.bin"))
        val_ids.tofile(os.path.join(out_dir, "val.bin"))
        print(f"Augmented Train tokens: {len(train_ids):,}")
        print(f"Static Val tokens: {len(val_ids):,}")

        token_lengths = [
            len(
                sp.EncodeAsIds(
                    text.replace("\r\n", "\n").replace("\n", "<NEWLINE>") + "<NEWLINE>"
                )
            )
            for text in sonnets
        ]

    else:
        tokenizer = train_hf_tokenizer(
            sonnets, tokenizer_type, vocab_size, special_tokens
        )
        encode_and_save_hf(tokenizer, train_list, val_list, out_dir)
        token_lengths = [len(tokenizer.encode(text).ids) for text in sonnets]

    word_lengths = [len(text.split()) for text in sonnets]

    print("---- Stats")
    print(f"Average words per file: {np.mean(word_lengths):.2f}")
    print(f"Median words: {np.median(word_lengths)}")
    print(f"Min words: {np.min(word_lengths)}")
    print(f"Max words: {np.max(word_lengths)}")
    print(f"Std dev: {np.std(word_lengths):.2f}")
    print("----")
    print(f"Average tokens per file: {np.mean(token_lengths):.2f}")
    print(f"Median tokens: {np.median(token_lengths)}")
    print(f"Min tokens: {np.min(token_lengths)}")
    print(f"Max tokens: {np.max(token_lengths)}")
    print(f"Std dev: {np.std(token_lengths):.2f}")


if __name__ == "__main__":
    main()
