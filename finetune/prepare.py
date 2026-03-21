import json
import os
import re
from pathlib import Path

import click


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    default=Path("data/processed/sonnets/"),
    show_default=True,
    help="Directory containing source sonnet .txt files.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("data/processed/finetune"),
    show_default=True,
    help="Directory where processed output is written.",
)
def main(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    filenames = [fn for fn in os.listdir(data_dir) if fn.endswith(".txt")]

    jsonl_data = []

    for filename in filenames:
        with open(data_dir / filename, "r", encoding="utf-8") as f:
            text = f.read().strip()
            title = re.search(r"\d{1,}\.\s(.*)\.", filename).group(1).strip("][»")  # type: ignore

            jsonl_data.append({"text": "<TITLE>" + title + "\n\n" + text})

    with open(output_dir / "finetune.jsonl", "w", encoding="utf-8") as f:
        for item in jsonl_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"Processed {len(jsonl_data)} sonnets and saved to {output_dir / 'finetune.jsonl'}."
    )


if __name__ == "__main__":
    main()
