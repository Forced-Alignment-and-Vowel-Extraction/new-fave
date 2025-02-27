from new_fave.extract import fave_extract
from pathlib import Path
from click.testing import CliRunner
import pytest
import yaml
import tempfile
import logging

def test_audio_textgrid():
    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)


    audio_path = Path("tests", "test_data", "corpus", "josef-fruehwald_speaker.wav")
    textgrid_path = Path("tests", "test_data", "corpus", "josef-fruehwald_speaker.TextGrid")
    ft_config = Path("tests", "test_patterns", "test_ft_config.yml")

    runner = CliRunner()

    result = runner.invoke(
        fave_extract,
        [
            "audio-textgrid",
            str(audio_path),
            str(textgrid_path),
            "--destination", tmp.name,
            "--ft-config", str(ft_config)
        ]
    )

    assert result.exit_code == 0, result.output
    csvs = list(tmp_path.glob("*.csv"))
    assert len(csvs) > 0

    result = runner.invoke(
        fave_extract,
        [
            "audio-textgrid",
            str(audio_path),
            str(textgrid_path),
            "--destination", tmp.name,
            "--ft-config", str(ft_config)
        ],
        input='n\nn\n'
    )

    assert result.exit_code == 0, result.output
    tmp.cleanup()

def test_corpus():
    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)


    corpus_path = Path("tests", "test_data", "corpus")
    speaker_path = Path("tests", "test_data", "corpus", "speakers.csv")
    ft_config = Path("tests", "test_patterns", "test_ft_config.yml")

    runner = CliRunner()

    result = runner.invoke(
        fave_extract,
        [
            "corpus",
            str(corpus_path),
            "--destination", tmp.name,
            "--ft-config", str(ft_config),
            "--speakers", str(speaker_path)
        ]
    )

    assert result.exit_code == 0, result.output
    csvs = list(tmp_path.glob("*.csv"))
    assert len(csvs) > 0


    result = runner.invoke(
        fave_extract,
        [
            "corpus",
            str(corpus_path),
            "--destination", tmp.name,
            "--ft-config", str(ft_config)
        ],
        input='n\nn\n'
    )

    assert result.exit_code == 0, result.output
    tmp.cleanup()

def test_bad_demographics():
    pass
    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)


    corpus_path = Path("tests", "test_data", "corpus")
    speaker_path = Path("tests", "test_data", "corpus", "speakers.csv")
    ft_config = Path("tests", "test_patterns", "test_ft_config.yml")

    runner = CliRunner()
    

    result = runner.invoke(
        fave_extract,
        [
            "corpus",
            str(corpus_path),
            "--destination", tmp.name,
            "--ft-config", str(ft_config),
            "--speakers", str(speaker_path)
        ]
    )

    assert result.exit_code == 0, result.output
    csvs = list(tmp_path.glob("*.csv"))
    assert len(csvs) > 0    
    tmp.cleanup()

def test_subcorpora():
    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)


    corpus_path1 = Path("tests", "test_data", "subcorpora", "josef-fruehwald"),
    corpus_path2 = Path("tests", "test_data", "subcorpora", "KY25A")
    speaker = Path("tests", "test_data", "subcorpora", "demographics.csv")
    ft_config = Path("tests", "test_patterns", "test_ft_config.yml")

    runner = CliRunner()

    result = runner.invoke(
        fave_extract,
        [
            "subcorpora",
            str(corpus_path1),
            str(corpus_path2),
            "--destination", tmp.name,
            "--ft-config", str(ft_config),
            "--speakers", str(speaker)
        ]
    )

    assert result.exit_code == 0, result.output
    csvs = list(tmp_path.glob("*.csv"))
    assert len(csvs) > 0


    result = runner.invoke(
        fave_extract,
        [
            "subcorpora",
            str(corpus_path1),
            str(corpus_path2),
            "--destination", tmp.name,
            "--ft-config", str(ft_config)
        ],
        input='n\nn\n'
    )

    assert result.exit_code == 0, result.output
    tmp.cleanup()
