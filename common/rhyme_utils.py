import re
import unicodedata


def normalize_word(word: str):
    clean_word = re.sub(r"[^\w\s\']", "", word).lower()
    clean_word = unicodedata.normalize("NFD", clean_word)
    return "".join(c for c in clean_word if unicodedata.category(c) != "Mn")


def extract_rhyme_suffix(word: str, max_rhyme_length: int = 2):
    clean_word = normalize_word(word)
    if not clean_word:
        return ""

    for i in range(len(clean_word) - 1, -1, -1):
        if clean_word[i] in "aeiou":
            rhyme = clean_word[i:]
            return rhyme[-max_rhyme_length:] if len(rhyme) > max_rhyme_length else rhyme

    return (
        clean_word[-max_rhyme_length:]
        if len(clean_word) >= max_rhyme_length
        else clean_word
    )


if __name__ == "__main__":
    print(extract_rhyme_suffix("ccredote"))
