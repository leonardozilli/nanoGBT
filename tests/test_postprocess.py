import importlib.util
from pathlib import Path

from click.testing import CliRunner

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "scripts" / "postprocess.py"
)
SPEC = importlib.util.spec_from_file_location("postprocess", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
postprocess = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(postprocess)


def test_clean_text():
    text = "Cäsa… [prova] (12) “ciao”\xa0fine"

    cleaned = postprocess.clean_text(text)

    assert "ä" not in cleaned
    assert "[" not in cleaned and "]" not in cleaned
    assert "(12)" not in cleaned
    assert "..." in cleaned
    assert "«ciao»" in cleaned
    assert "\xa0" not in cleaned


def test_check_structure_valid():
    valid = "\n\n".join(
        [
            "a\nb\nc\nd",
            "e\nf\ng\nh",
            "i\nj\nk",
            "l\nm\nn",
        ]
    )

    assert postprocess.check_structure(valid) is True


def test_check_structure_invalid():
    invalid = "\n\n".join(
        [
            "a\nb\nc",
            "e\nf\ng\nh",
            "i\nj\nk",
            "l\nm\nn",
        ]
    )

    assert postprocess.check_structure(invalid) is False


def test_tag_sonnet_rhymes():
    text = "\n".join(
        [
            "<SONNET>",
            "Prima linea casa",
            "Seconda linea sole",
            "Terza linea sole",
            "Quarta linea casa",
            "",
            "<STANZA>",
            "Quinta linea luna",
            "<END>",
        ]
    )

    tagged = postprocess.tag_sonnet_rhymes(text, max_rhyme_length=2)

    assert "Prima linea casa <RHYME_A>" in tagged
    assert "Seconda linea sole <RHYME_B>" in tagged
    assert "Terza linea sole <RHYME_B>" in tagged
    assert "Quarta linea casa <RHYME_A>" in tagged
    assert "<SONNET>" in tagged
    assert "<STANZA>" in tagged
    assert "<END>" in tagged


def test_tag_sonnet_rhymes_uses_separate_tercet_labels_for_14_lines():
    text = "\n\n".join(
        [
            "uno casa\ndue sole\ntre sole\nquattro casa",
            "cinque casa\nsei sole\nsette sole\notto casa",
            "nove luna\ndieci lume\nundici luna",
            "dodici lume\ntredici luna\nquattordici lume",
        ]
    )

    tagged = postprocess.tag_sonnet_rhymes(text, max_rhyme_length=2)

    assert "uno casa <RHYME_A>" in tagged
    assert "due sole <RHYME_B>" in tagged
    assert "nove luna <RHYME_C>" in tagged
    assert "dieci lume <RHYME_D>" in tagged


def test_main_cli(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()

    sonnet_text = "\n\n".join(
        [
            "uno casa\ndue sole\ntre sole\nquattro casa",
            "cinque casa\nsei sole\nsette sole\notto casa",
            "nove luna\ndieci mare\nundici luna",
            "dodici luna\ntredici mare\nquattordici luna",
        ]
    )
    input_file = raw_dir / "01. test.txt"
    input_file.write_text(sonnet_text, encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        postprocess.main,
        [
            "--data-dir",
            str(raw_dir),
            "--out-dir",
            str(out_dir),
            "--include-title",
            "--mark-rhymes",
            "--rhyme-length",
            "2",
        ],
    )

    assert result.exit_code == 0

    output_file = out_dir / "01. test.txt"
    assert output_file.exists()

    output_text = output_file.read_text(encoding="utf-8")
    assert output_text.startswith("<SONNET>")
    assert "<TITLE>01. test</TITLE>" in output_text
    assert "<STANZA>" in output_text
    assert output_text.rstrip().endswith("<END>")
    assert output_text.count("<RHYME_") == 14
