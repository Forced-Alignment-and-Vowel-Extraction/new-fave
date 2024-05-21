from new_fave.speaker.speaker import Speaker
import polars as pl
from pathlib import Path


speaker_path = Path("tests", "test_data", "speaker")

def test_yaml():
    speaker1 = Speaker(speaker_path.joinpath("speaker.yml"))
    speaker2 = Speaker(speaker_path.joinpath("speaker2.yml"))

    assert isinstance(speaker1, Speaker)
    assert isinstance(speaker2, Speaker)

    assert isinstance(speaker1.df, pl.DataFrame)
    assert isinstance(speaker2.df, pl.DataFrame)

def test_csv():
    speaker = Speaker(speaker_path.joinpath("speaker.csv"))
    assert isinstance(speaker, Speaker)
    assert isinstance(speaker.df, pl.DataFrame)

def test_xlsx():
    speaker = Speaker(speaker_path.joinpath("speaker.xlsx"))
    assert isinstance(speaker, Speaker)
    assert isinstance(speaker.df, pl.DataFrame)

def test_oldfave():
    speaker = Speaker(speaker_path.joinpath("josef-fruehwald_speaker.speaker"))
    assert isinstance(speaker, Speaker)
    assert isinstance(speaker.df, pl.DataFrame)

def test_multi_speaker():
    speaker1 = Speaker(speaker_path.joinpath("speaker.yml"))
    speaker2 = Speaker(speaker_path.joinpath("speaker2.yml"))

    multi_speaker = Speaker([speaker1, speaker2])

    assert isinstance(multi_speaker, Speaker)
    assert isinstance(multi_speaker.df, pl.DataFrame)
    assert multi_speaker.df.shape[0] == speaker1.df.shape[0] + speaker2.df.shape[0]
