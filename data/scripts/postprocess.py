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

from common.rhyme_utils import extract_rhyme_suffix

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


def tag_sonnet_rhymes(text, max_rhyme_length=2, include_last_word=False):
    lines = text.split("\n")
    octave_rhyme_map = {}
    sestet_rhyme_map = {}
    global_rhyme_map = {}
    next_octave_char = 65  # A
    next_sestet_char = 67  # C
    next_global_char = 65
    tagged_lines = []

    line_count = len(
        [line for line in lines if line.strip() and not line.strip().startswith("<")]
    )
    split_octave_sestet_labels = line_count == 14
    line_index = 0

    for line in lines:
        if not line.strip() or line.strip().startswith("<"):
            tagged_lines.append(line)
            continue

        words = line.split()
        last_word = re.sub(r"[^\w]+$", "", words[-1]) or words[-1]
        suffix = extract_rhyme_suffix(last_word, max_rhyme_length=max_rhyme_length)

        if split_octave_sestet_labels:
            if line_index < 8:
                if suffix not in octave_rhyme_map:
                    octave_rhyme_map[suffix] = chr(next_octave_char)
                    next_octave_char += 1
                rhyme_letter = octave_rhyme_map[suffix]
            else:
                if suffix not in sestet_rhyme_map:
                    sestet_rhyme_map[suffix] = chr(next_sestet_char)
                    next_sestet_char += 1
                rhyme_letter = sestet_rhyme_map[suffix]
        else:
            if suffix not in global_rhyme_map:
                global_rhyme_map[suffix] = chr(next_global_char)
                next_global_char += 1
            rhyme_letter = global_rhyme_map[suffix]

        if include_last_word:
            tagged_line = f"<RHYME_{rhyme_letter}> {last_word} | {line}"
        else:
            tagged_line = f"<RHYME_{rhyme_letter}> {line}"

        tagged_lines.append(tagged_line)
        line_index += 1

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
    "--include-last-word",
    is_flag=True,
    help="Include the line's last word in the rhyme markings",
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
    include_last_word: bool,
    rhyme_length: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    filenames = sorted(fn for fn in os.listdir(data_dir) if fn.endswith(".txt"))

    c = 0
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
                text = tag_sonnet_rhymes(
                    text,
                    max_rhyme_length=rhyme_length,
                    include_last_word=include_last_word,
                )

        with open(out_dir / filename, "w", encoding="utf-8") as f:
            f.write(text)
            c += 1

    print(f"Processed {c} sonnets and saved to {out_dir}")


if __name__ == "__main__":
    main()
