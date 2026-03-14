"""
Clean and process raw sonnets, adding special tokens:
- <SONNET> at the beginning of each sonnet
- <STANZA> between stanzas
- <END> at the end of each sonnet
"""

import re
from pathlib import Path

import click

from common.rhyme_utils import extract_rhyme_suffix, normalize_word

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
    "%": "",
    "&": "e",
    "*": "",
    "=": "",
    "_": "",
    "|": "",
    "~": "",
    "©": "",
    "\xad": "",
    "¯": "",
    "°": "'",
    "³": "",
    "·": "",
    "€": "",
    "„": "",
    "ར": "",
    "о": "o",
    "п": "n",
    "ə": "e",
    "ɔ": "o",
    "ı": "i",
    "―": "—",
    "ą": "o",
    "ˇ": "",
    "ç": "c",
    "å": "a",
    "E`": "È",
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


def _assign_rhyme_letter(rhyme_map: dict[str, str], suffix_key: str, next_char: int):
    max_rhyme_char = ord("G")
    first_wrap_char = ord("C")

    if suffix_key not in rhyme_map:
        if next_char > max_rhyme_char:
            next_char = first_wrap_char
        rhyme_map[suffix_key] = chr(next_char)
        next_char += 1

    return rhyme_map[suffix_key], next_char


def tag_sonnet_rhymes(text, include_rhyme_suffix=False):
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

        clean_line = re.sub(r"[^\w\']+$", "", line)

        words = clean_line.split()
        if not words:
            tagged_lines.append(line)
            continue
        last_word = words[-1]

        suffix = extract_rhyme_suffix(last_word)
        suffix_key = normalize_word(suffix)

        if split_octave_sestet_labels:
            if line_index < 8:
                rhyme_letter, next_octave_char = _assign_rhyme_letter(
                    octave_rhyme_map, suffix_key, next_octave_char
                )
            else:
                rhyme_letter, next_sestet_char = _assign_rhyme_letter(
                    sestet_rhyme_map, suffix_key, next_sestet_char
                )
        else:
            rhyme_letter, next_global_char = _assign_rhyme_letter(
                global_rhyme_map, suffix_key, next_global_char
            )

        if include_rhyme_suffix:
            tagged_line = f"<RHYME_{rhyme_letter}> {suffix} | {line}"
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
    "--include-rhyme-suffix",
    is_flag=True,
    help="Include the suffix of the last word in each line along with the rhyme token",
)
def main(
    data_dir: Path,
    out_dir: Path,
    include_title: bool,
    mark_rhymes: bool,
    include_rhyme_suffix: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(p for p in data_dir.rglob("*.txt") if p.is_file())

    c = 0
    for input_file in input_files:
        relative_path = input_file.relative_to(data_dir)
        output_file = out_dir / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
            text = clean_text(text)
            if not check_structure(text):
                print(f"Unusual structure in {relative_path}")
            text = "\n\n<STANZA>\n\n".join(text.split("\n\n"))
            if include_title:
                title = input_file.stem
                text = f"<TITLE>{title}</TITLE>\n\n{text}"
            text = "<SONNET>\n" + text + "\n<END>"
            if mark_rhymes:
                text = tag_sonnet_rhymes(
                    text,
                    include_rhyme_suffix=include_rhyme_suffix,
                )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
            c += 1

    print(f"Processed {c} sonnets and saved to {out_dir}")


if __name__ == "__main__":
    main()
