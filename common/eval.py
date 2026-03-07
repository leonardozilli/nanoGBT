from collections import Counter

from common.rhyme_utils import extract_rhyme_suffix

ALLOWED_OCTAVE_PATTERNS = {"ABBAABBA", "ABABABAB", "ABBABAAB", "ABABBABA"}
ALLOWED_SESTET_PATTERNS = {"ABABAB", "ABAABA", "ABCABC", "ABCBAC", "ABCACB"}
ALLOWED_QUATRAIN_PATTERNS = {"ABBA", "ABAB", "BAAB", "BABA"}
ALLOWED_TERCET_PATTERNS = {"ABA", "BAB", "ABC", "BAC", "ACB"}


def _pattern_signature(sequence: str):
    mapping = {}
    next_code = ord("A")
    signature = []

    for symbol in sequence:
        if symbol not in mapping:
            mapping[symbol] = chr(next_code)
            next_code += 1
        signature.append(mapping[symbol])

    return "".join(signature)


def evaluate_structure(text):
    text = text.replace("<SONNET>", "").replace("<END>", "").strip()
    text = text.replace("<NEWLINE>", "\n")

    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip() and not line.startswith("<")
    ]

    line_count = len(lines)
    is_14_lines = line_count == 14

    stanzas = [stanza.strip() for stanza in text.split("\n\n") if stanza.strip()]
    stanza_lengths = [
        len([ln for ln in stanza.split("\n") if ln.strip() and not ln.startswith("<")])
        for stanza in stanzas
    ]
    stanza_lengths = [
        stanza_length for stanza_length in stanza_lengths if stanza_length > 0
    ]
    is_correct_structure = stanza_lengths == [4, 4, 3, 3]

    rhyme_map = {}
    current_char = ord("A")
    scheme = []
    for line in lines:
        words = line.split()
        if not words:
            continue
        suffix = extract_rhyme_suffix(words[-1])
        if suffix not in rhyme_map:
            rhyme_map[suffix] = chr(current_char)
            current_char += 1
        scheme.append(rhyme_map[suffix])

    rhyme_counts = Counter(scheme)
    rhyme_lines = sum(v for v in rhyme_counts.values() if v > 1)

    total_stanzas = len(stanza_lengths)
    valid_stanzas = 0
    cursor = 0

    for i, stanza_length in enumerate(stanza_lengths):
        stanza_scheme = "".join(scheme[cursor : cursor + stanza_length])
        cursor += stanza_length
        stanza_pattern = _pattern_signature(stanza_scheme)
        if i < 2 and stanza_length == 4 and stanza_pattern in ALLOWED_QUATRAIN_PATTERNS:
            valid_stanzas += 1
        elif (
            i >= 2 and stanza_length == 3 and stanza_pattern in ALLOWED_TERCET_PATTERNS
        ):
            valid_stanzas += 1

    is_valid_stanzas = is_14_lines and total_stanzas == 4 and valid_stanzas == 4

    is_valid_sonnet = False
    if is_14_lines and len(scheme) == 14:
        octave_pattern = _pattern_signature("".join(scheme[:8]))
        sestet_pattern = _pattern_signature("".join(scheme[8:]))
        if (
            octave_pattern in ALLOWED_OCTAVE_PATTERNS
            and sestet_pattern in ALLOWED_SESTET_PATTERNS
        ):
            is_valid_sonnet = True

    return {
        "is_14_lines": is_14_lines,
        "is_correct_structure": is_correct_structure,
        "is_valid_stanzas": is_valid_stanzas,
        "valid_stanzas": valid_stanzas,
        "total_stanzas": total_stanzas,
        "is_valid_sonnet": is_valid_sonnet,
        "rhyme_lines": rhyme_lines,
        "line_count": line_count,
    }
