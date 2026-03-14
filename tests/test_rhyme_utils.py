from common.rhyme_utils import extract_rhyme_suffix


def test_extract_rhyme_suffix():
    assert extract_rhyme_suffix("Supplisce!!!") == "isce"
    assert extract_rhyme_suffix("Sgraffignar!!") == "ignar"
    assert extract_rhyme_suffix("A") == "a"
    assert extract_rhyme_suffix("Pietà!") == "à"
    assert extract_rhyme_suffix("amor") == "amor"
    assert extract_rhyme_suffix("amore...") == "ore"
    assert extract_rhyme_suffix("rhythms") == "hms"
    assert extract_rhyme_suffix("sciampanella") == "ella"
