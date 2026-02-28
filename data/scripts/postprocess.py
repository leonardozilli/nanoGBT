"""
Clean and process raw sonnets, adding special tokens:
- <SONNET> at the beginning of each sonnet
- <STANZA> between stanzas
- <END> at the end of each sonnet
"""

import os
import re
from pathlib import Path

import click

replace_table = {
    "ä": "a",
    "ā": "a",
    "â": "a",
    "ō": "o",
    "ö": "o",
    "ū": "u",
    "ī": "ì",
    "î": "i",
    "Ô": "O",
    "ë": "e",
    "ê": "e",
    "ě": "e",
    "ü": "u",
    "[": "",
    "]": "",
    "…": "...",
    "‘": "’",
    "\xa0": " ",
    "“": "«",
    "”": "»",
}


def clean_text(text):
    for old, new in replace_table.items():
        text = text.replace(old, new)

    text = re.sub(r"\(\d+\)", "", text)
    return text


def check_structure(text: str) -> bool:
    """Checks if text matches standard sonnet stanza structure."""
    stanzas = re.split(r"\n\s*\n", text.strip())
    counts = [len(s.split("\n")) for s in stanzas if s.strip()]
    if len(counts) < 4:
        return False
    if counts[0] != 4 or counts[1] != 4:
        return False
    if not all(c == 3 for c in counts[2:]):
        return False
    return True


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    default=Path("data/raw/"),
    show_default=True,
    help="Directory containing raw sonnet .txt files.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("data/processed/"),
    show_default=True,
    help="Directory where processed sonnets are written.",
)
@click.option(
    "--include-title",
    is_flag=True,
    help="Include the title in the processed output.",
)
def main(data_dir: Path, out_dir: Path, include_title: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    filenames = sorted(fn for fn in os.listdir(data_dir) if fn.endswith(".txt"))

    for filename in filenames:
        with open(data_dir / filename, "r", encoding="utf-8") as f:
            text = f.read().strip()
            text = clean_text(text)
            if not check_structure(text):
                print(f"Unusual structure in {filename}")
            text = "\n\n<STANZA>\n\n".join(text.split("\n\n"))
            if include_title:
                title = filename.replace(".txt", "")
                text = f"<TITLE>{title}</TITLE>\n\n{text}"
            text = "<SONNET>\n" + text + "\n<END>"

        with open(out_dir / filename, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
