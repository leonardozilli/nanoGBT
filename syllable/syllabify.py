import re

VOWELS = r"aeiouàáèéìíïîüòóôùúăēōyAEIOUÀÁÈÉÌÍÏÎÜÒÓÔÙÚĂĒŌY’'"
ACCENTED_VOWELS = r"àáèéìíïòóôùúÀÁÈÉÌÍÏÒÓÔÙÚ"
STRONG_VOWELS = r"aeoàáèéòóăēōAEOÀÁÈÉÒÓĂĒŌ"
CONSONANTS = r"bcdfghjlmnpqrstvwxzBCDFGHJLMNPQRSTVWXZ"
RHYME_LINE_RE = re.compile(r"^(<RHYME_[A-Z]>)\s+([^|]+?)\s+\|\s+(.*)$")

INDIVISIBLE_CLUSTERS = r"ch|gh|gn|gl|sc|[pbcftvdg][rl]|s[bcdfghjlmnpqrtvwxz]+"

SYLLABIFICATION_RULES = [
    # 1. split double consonants
    (
        re.compile(
            rf"([{VOWELS}])(?!(?:{INDIVISIBLE_CLUSTERS}))([{CONSONANTS}])([{CONSONANTS}])",
            re.IGNORECASE,
        ),
        r"\1\2-\3",
    ),
    # 2. isolate indivisible clusters
    (
        re.compile(
            rf"([{VOWELS}])((?:{INDIVISIBLE_CLUSTERS}))(?=[{VOWELS}])", re.IGNORECASE
        ),
        r"\1-\2",
    ),
    # 3. split strong vowels
    (re.compile(rf"([{STRONG_VOWELS}])([{STRONG_VOWELS}])", re.IGNORECASE), r"\1-\2"),
    # 4. split vowel + single consonant + vowel
    (
        re.compile(rf"([{VOWELS}])([{CONSONANTS}])(?=[{VOWELS}])", re.IGNORECASE),
        r"\1-\2",
    ),
]


def syllabify_word(word: str) -> list[str]:
    """Split a single word into syllables."""
    for pattern, replacement in SYLLABIFICATION_RULES:
        word = pattern.sub(replacement, word)
    return [s for s in word.split("-") if s]


def _build_syllables(line: str) -> list[str]:
    """Split words and attach separators to the previous syllable block."""
    syllables = []
    tokens = re.findall(r"[A-Za-zÀ-Ōà-ō\’\']+|[^A-Za-zÀ-Ōà-ō\’\']+", line)

    for token in tokens:
        if re.match(r"^[A-Za-zÀ-Ōà-ō\’\']+$", token):
            syllables.extend(syllabify_word(token))
            continue

        if syllables:
            syllables[-1] += token
        else:
            syllables.append(token)

    return syllables


def _should_merge_syllables(current_syl: str, next_syl: str) -> bool:
    """Return True when elision or sinalefe allows merging two adjacent syllables."""
    curr_clean = re.sub(r"[^A-Za-zÀ-Ōà-ō\’\']", "", current_syl)
    next_clean = re.sub(r"[^A-Za-zÀ-Ōà-ō\’\']", "", next_syl)

    if not curr_clean or not next_clean:
        return False

    # 1. check if the token ends in an apostrophe
    curr_ends_apostrophe = bool(re.search(r"[\’\']$", curr_clean))

    # 2. check for a vowel at the end of the current token or start of the next.
    # treat 'h' as a vowel here.
    curr_ends_vowel = bool(
        re.search(rf"[{VOWELS}\’\'][hH]?$", curr_clean, re.IGNORECASE)
    )
    next_starts_vowel = bool(re.search(rf"^[hH]?[{VOWELS}]", next_clean, re.IGNORECASE))

    # 3. check if there's a space or punctuation before the next word
    is_word_boundary = bool(re.search(r"[^A-Za-zÀ-Ōà-ō\’\']$", current_syl))

    is_elision = curr_ends_apostrophe and next_starts_vowel

    is_sinalefe = curr_ends_vowel and next_starts_vowel and is_word_boundary

    return is_elision or is_sinalefe


def _merge_syllables(syllables: list[str]) -> list[str]:
    """Merge syllables to respect elision/sinalefe rules."""
    merged_syllables = []
    i = 0

    while i < len(syllables):
        current_syl = syllables[i]

        while i < len(syllables) - 1:
            next_syl = syllables[i + 1]
            # merge if elision or sinalefe
            if _should_merge_syllables(current_syl, next_syl):
                current_syl += next_syl
                i += 1
            else:
                break

        merged_syllables.append(current_syl)
        i += 1

    return merged_syllables


def _split_and_move_whitespace(syllables: list[str]) -> list[str]:
    """Split punctuation and move whitespace to the beginning of the next token."""
    final_tokens = []

    for syl in syllables:
        # keep merged blocks intact
        if " " in syl.strip() or "'" in syl or "’" in syl:
            final_tokens.append(syl)
            continue

        for part in re.split(r"(\s+|[^A-Za-zÀ-Ōà-ō\’\'\s])", syl):
            if part:
                final_tokens.append(part)

    prefixed_tokens = []
    pending_space = ""

    for token in final_tokens:
        if token.isspace():
            pending_space += token
            continue

        stripped_token = token.rstrip(" \t")
        trailing_spaces = token[len(stripped_token) :]
        prefixed_tokens.append(pending_space + stripped_token)
        pending_space = trailing_spaces

    if pending_space:
        prefixed_tokens.append(pending_space)

    return prefixed_tokens


def syllabify_line(line: str) -> list[str]:
    """Split a line into syllables."""
    syllables = _build_syllables(line)
    syllables = _merge_syllables(syllables)
    return _split_and_move_whitespace(syllables)


def _extract_rhyme_metadata(line: str) -> tuple[list[str] | None, str | None]:
    """Parse a line for prepended rhyme metadata (special tokens, rhyme tags, rhyme suffixes).
    Returns a tuple of (metadata_tokens, verse).
    """
    match = RHYME_LINE_RE.match(line)
    if not match:
        return None, None

    special_token = match.group(1)
    rhyme_suffix = match.group(2).strip()
    verse = match.group(3)
    metadata_tokens = [special_token, rhyme_suffix, "|"]
    return metadata_tokens, verse


def syllabify_text(text: str) -> list[str]:
    """Split a block of text into syllables."""
    lines = text.split("\n")
    tokens = []

    for line in lines:
        stripped_line = line.strip()

        # don't split special tokens
        if stripped_line in ["<SONNET>", "<STANZA>", "<END>"]:
            tokens.append(stripped_line)
            tokens.append("\n")
            continue

        if stripped_line == "":
            tokens.append("\n")
            continue

        metadata_tokens, verse = _extract_rhyme_metadata(stripped_line)

        if metadata_tokens is not None:
            tokens.extend(metadata_tokens)
            tokens.extend(syllabify_line(verse))
        else:
            tokens.extend(syllabify_line(line))

        tokens.append("\n")

    # cleanup trailing newline
    if tokens and tokens[-1] == "\n":
        tokens.pop()

    return tokens
