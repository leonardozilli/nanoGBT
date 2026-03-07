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
    Che ffior de Papa creeno! Accidenti!
    Co rrispetto de lui pare er Cacamme.
    Bbella galanteria da tate e mmamme
    pe ffà bbobo a li fijji impertinenti!

    Ha un erpeto pe ttutto, nun tiè ddenti,
    è gguercio, je strascineno le gamme,
    spènnola da una parte, e bbuggiaramme
    si arriva a ffà la pacchia a li parenti.

    Guarda llí cche ffigura da vienicce
    a ffà da Crist’in terra! Cazzo matto
    imbottito de carne de sarcicce!

    Disse bbene la serva de l’Oreficce
    quanno lo vedde in chiesa: cianno fatto matto
    un gran brutto strucchione de Ponteficce.
    """

    metrics = evaluate_structure(text)

    assert metrics["is_14_lines"] is True
    assert metrics["is_correct_structure"] is True
    assert metrics["valid_stanzas"] == 4
    assert metrics["total_stanzas"] == 4
    assert metrics["is_valid_sonnet"] is True
    assert metrics["line_count"] == 14
    assert metrics["rhyme_lines"] == 14


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
