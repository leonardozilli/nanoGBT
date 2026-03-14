import re
import unicodedata

ACCENTED_VOWELS = "àèéìíòóùú"
PLAIN_VOWELS = "aeiou"


def normalize_word(word: str):
    word = unicodedata.normalize("NFD", word)
    word = re.sub(r"[^\w\s']", "", word)
    word = "".join(c for c in word if unicodedata.category(c) != "Mn")
    return word.lower()


def extract_rhyme_suffix(word: str) -> str:
    original = re.sub(rf"[^\w\s'{ACCENTED_VOWELS}]", "", word.lower())
    normalized = normalize_word(word)

    if not normalized:
        return ""

    # if there's an accented vowel, use that
    for i, c in enumerate(original):
        if c in ACCENTED_VOWELS:
            return original[i:]

    # if not, use the last vowel
    vowel_indices = [i for i, c in enumerate(normalized) if c in PLAIN_VOWELS]

    # if no vowels, return last 3 chars
    if not vowel_indices:
        return normalized[-3:]

    start = vowel_indices[-2] if len(vowel_indices) >= 2 else vowel_indices[0]

    return normalized[start:]
