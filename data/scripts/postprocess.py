"""
Clean and process raw sonnets, adding special tokens:
- <SONNET> at the beginning of each sonnet
- <STANZA> between stanzas
- <END> at the end of each sonnet
"""

import os
import re
import unicodedata
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
    stanzas = re.split(r"\n\s*\n", text.strip())
    counts = [len(s.split("\n")) for s in stanzas if s.strip()]
    if len(counts) < 4:
        return False
    if counts[0] != 4 or counts[1] != 4:
        return False
    if not all(c == 3 for c in counts[2:]):
        return False
    return True


def _normalize_word(word: str):
    clean_word = re.sub(r"[^\w\s\']", "", word).lower()
    clean_word = unicodedata.normalize("NFD", clean_word)
    return "".join(c for c in clean_word if unicodedata.category(c) != "Mn")


def extract_rhyme_suffix(word: str, max_rhyme_length: int = 2):
    clean_word = _normalize_word(word)
    if not clean_word:
        return ""

    vowels = "aeiou"
    for i in range(len(clean_word) - 1, -1, -1):
        if clean_word[i] in vowels:
            rhyme = clean_word[i:]
            return rhyme[-max_rhyme_length:] if len(rhyme) > max_rhyme_length else rhyme

    return (
        clean_word[-max_rhyme_length:]
        if len(clean_word) >= max_rhyme_length
        else clean_word
    )


def tag_sonnet_rhymes(text, max_rhyme_length=2):
    lines = text.split("\n")
    rhyme_map = {}
    current_char = 65
    tagged_lines = []

    for line in lines:
        if not line.strip() or line.strip().startswith("<"):
            tagged_lines.append(line)
            continue

        words = line.split()
        last_word = words[-1]
        suffix = extract_rhyme_suffix(last_word, max_rhyme_length=max_rhyme_length)

        if suffix not in rhyme_map:
            rhyme_map[suffix] = chr(current_char)
            current_char += 1

        rhyme_letter = rhyme_map[suffix]

        tagged_line = f"{line} <RHYME_{rhyme_letter}>"
        tagged_lines.append(tagged_line)

    return "\n".join(tagged_lines)


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
@click.option(
    "--mark-rhymes", is_flag=True, help="Mark rhyming words with special tokens"
)
@click.option(
    "--rhyme-length",
    default=2,
    show_default=True,
    help="Number of characters to consider for rhyme detection",
)
def main(
    data_dir: Path,
    out_dir: Path,
    include_title: bool,
    mark_rhymes: bool,
    rhyme_length: int,
):
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
            if mark_rhymes:
                text = tag_sonnet_rhymes(text, max_rhyme_length=rhyme_length)

        with open(out_dir / filename, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
