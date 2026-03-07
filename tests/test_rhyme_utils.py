from common.rhyme_utils import extract_rhyme_suffix


def test_extract_rhyme_suffix():
    assert extract_rhyme_suffix("Supplisce!!!") == "e"
    assert extract_rhyme_suffix("Sgraffignar!!") == "ar"
    assert extract_rhyme_suffix("A") == "a"
    assert extract_rhyme_suffix("Pietà!", max_rhyme_length=2) == "a"
    assert extract_rhyme_suffix("amor", max_rhyme_length=2) == "or"
    assert extract_rhyme_suffix("amore...", max_rhyme_length=1) == "e"
    assert extract_rhyme_suffix("rhythms", max_rhyme_length=3) == "hms"
    assert extract_rhyme_suffix("sciampanella", max_rhyme_length=3) == "a"
