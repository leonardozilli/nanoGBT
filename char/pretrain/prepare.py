import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)

# ruff: noqa: E402
import click
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features, Value, load_dataset
from datasets.exceptions import CastError
from tqdm.auto import tqdm


@click.command()
@click.option(
    "--dataset-name",
    default="PleIAs/Italian-PD",
    show_default=True,
    help="Hugging Face dataset name.",
)
@click.option(
    "--target-words",
    default=200_000_000,
    type=int,
    show_default=True,
    help="Target number of words.",
)
@click.option(
    "--batch-size",
    default=10_000,
    type=int,
    show_default=True,
    help="Rows per parquet flush.",
)
@click.option(
    "--out-dir",
    default="data/italian-pd",
    show_default=True,
    help="Output directory for parquet and metadata.",
)
def main(
    dataset_name: str,
    target_words: int,
    batch_size: int,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "data.parquet")

    print("Loading dataset...")
    features = Features(
        {
            "identifier": Value("string"),
            "creator": Value("string"),
            "title": Value("string"),
            "publication_date": Value("int32"),
            "text": Value("string"),
            "word_count": Value("int32"),
        }
    )

    ds = load_dataset(dataset_name, split="train", streaming=True, features=features)

    ds_filtered = ds.filter(
        lambda x: x["publication_date"] is not None and 1791 <= x["publication_date"]
    )

    print(f"Extracting ~{target_words:,} words...")

    years_counter = Counter()
    buffer = []
    current_words = 0
    writer = None

    with tqdm(
        total=target_words,
        initial=current_words,
        unit="words",
        unit_scale=True,
        desc="Collecting samples",
    ) as pbar:
        try:
            for idx, row in enumerate(ds_filtered):
                if current_words >= target_words:
                    break

                text = row.get("text")
                publication_date = row.get("publication_date")
                words_in_row = row.get("word_count") or 0

                if not text or words_in_row == 0:
                    continue

                remaining = target_words - current_words
                increment = min(words_in_row, remaining)

                years_counter[publication_date] += 1
                current_words += increment
                pbar.update(increment)

                buffer.append(
                    {
                        "identifier": row.get("identifier"),
                        "creator": row.get("creator"),
                        "title": row.get("title"),
                        "publication_date": publication_date,
                        "word_count": words_in_row,
                        "text": text,
                    }
                )

                if len(buffer) >= batch_size:
                    table = pa.Table.from_pylist(buffer)

                    if writer is None:
                        writer = pq.ParquetWriter(out_file, table.schema)

                    writer.write_table(table)
                    buffer = []

        except CastError:
            print("error")

    if buffer:
        table = pa.Table.from_pylist(buffer)
        if writer is None:
            writer = pq.ParquetWriter(out_file, table.schema)
        writer.write_table(table)

    if writer:
        writer.close()

    print(f"\nSuccess! Saved {current_words:,} words to {out_file}")

    metadata = {
        "dataset_name": dataset_name,
        "total_words": current_words,
        "publication_years": dict(years_counter),
        "created_at": datetime.now().isoformat(),
    }

    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    main()
