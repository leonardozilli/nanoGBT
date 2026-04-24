from common.eval import evaluate_structure


def _build_sonnet_from_scheme(scheme: str) -> str:
    words = {
        "A": "casa",
        "B": "mare",
        "C": "luna",
        "D": "sole",
        "E": "vento",
    }

    lines = [
        f"Linea {index + 1} termina con {words[letter]}"
        for index, letter in enumerate(scheme)
    ]

    return "\n\n".join(
        [
            "\n".join(lines[0:4]),
            "\n".join(lines[4:8]),
            "\n".join(lines[8:11]),
            "\n".join(lines[11:14]),
        ]
    )


def test_evaluate_structure_valid_sonnet():
    text = """<SONNET>
    Te lo saressi creso, eh Gurgumella,
    ch’er zor paìno, er zor dorce-me-frega,
    che mmanco ha ffiato per annà a bbottega,
    potessi slargà er buscio a ’na zitella?

    Tu nu lo sai ch’edè sta marachella;
    tutta farina de quell’antra strega.
    Mo che nun trova lei chi jje la sega,
    fa la ruffiana de la su’ sorella.

    Io sarebbe omo, corpo de l’abbrei,
    senza mettécce né ssale né ojjo,
    de dàjjene tre vorte trentasei:

    ma nun vojo piú affríggeme nun vojjo;
    che de donne pe ddio come che llei
    ’ggni monnezzaro me ne dà un pricojjo.
    """

    metrics = evaluate_structure(text)

    assert metrics["is_14_lines"] is True
    assert metrics["is_correct_structure"] is True
    assert metrics["valid_stanzas"] == 4
    assert metrics["total_stanzas"] == 4
    assert metrics["line_count"] == 14
    assert metrics["rhyme_lines"] == 14
    assert metrics["is_valid_sonnet"] is True


def test_evaluate_structure_valid_sonnet_not_strict():
    text = """<SONNET>
    Jeri, all’orloggio de la Cchiesa Nova,
    fra Luca incontrò Agnesa co la brocca.
    Dice: «Beato lui», dice, «a chi tocca»,
    dice, «e nun sa ch’edè chi nu lo prova».

    Risponne lei, dice: «Chi cerca, trova;
    ma a me», dice, «puliteve la bocca».
    «Aùh», dicéee... «e perché nun te fai biocca?»
    «Eh», dice, «e chi me mette sotto l’ova?»

    «Ce n’ho io», dice, «un paro fresche vive»,
    dice, «e ttamante, e tutt’e ddua ’ngallate:
    le vôi sperà si ssò bbone o ccattive?»

    Checco, te pensi che nun l’ha pijjate?
    Ah llei pe nnun sapé legge né scrive,
    ha vorzuto assaggià l’ova der frate.
    """

    metrics = evaluate_structure(text, strict=False)

    assert metrics["is_14_lines"] is True
    assert metrics["is_correct_structure"] is True
    assert metrics["valid_stanzas"] == 4
    assert metrics["total_stanzas"] == 4
    assert metrics["line_count"] == 14
    assert metrics["rhyme_lines"] == 14
    assert metrics["is_valid_sonnet"] is True


def test_evaluate_structure_invalid_sonnet():
    text = """
    Ciarivò ir Papa, ch’er Papa io me sce vorze
    la fasse scusa der Papa Palazzo,
    si cce s’abbi er monno doppo er gran birba-de-staggione a ppoco la pupazzone
    """

    metrics = evaluate_structure(text)

    assert metrics["is_14_lines"] is False
    assert metrics["is_correct_structure"] is False
    assert metrics["valid_stanzas"] == 0
    assert metrics["total_stanzas"] == 1
    assert metrics["is_valid_sonnet"] is False
    assert metrics["line_count"] == 3


def test_evaluate_structure_rejects_valid_octave_with_invalid_sestet():
    text = _build_sonnet_from_scheme("ABBAABBAEEEEEE")

    metrics = evaluate_structure(text)

    assert metrics["is_14_lines"] is True
    assert metrics["is_correct_structure"] is True
    assert metrics["valid_stanzas"] == 2
    assert metrics["total_stanzas"] == 4
    assert metrics["is_valid_sonnet"] is False
