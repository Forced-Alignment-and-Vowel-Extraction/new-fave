from new_fave import fave_corpus
from new_fave.patterns.writers import write_data, pickle_speakers, unpickle_speakers
from new_fave import SpeakerCollection
from pathlib import Path
import tempfile
import logging
logging.basicConfig(level = logging.INFO)

corpus_path = Path("tests", "test_data", "corpus")
ft_config_path = Path("tests", "test_patterns", "test_ft_config.yml")

speakers = fave_corpus(
    corpus_path=corpus_path,
    speakers=0,
    ft_config=ft_config_path
)

def test_write_all_data():
    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)

    write_data(
        speakers,
        destination=tmp_path
    )

    all_files = [p.name for p in tmp_path.glob("*")]

    assert any(["_tracks.csv" in f for f in all_files])
    assert any(["_param.csv" in f for f in all_files])
    assert any(["_logparam.csv" in f for f in all_files])
    assert any(["_points.csv" in f for f in all_files])
    assert any(["_recoded.TextGrid" in f for f in all_files])

    n_input = len({x.file_name for x in speakers.values()})
    
    assert len(all_files) == n_input * 5

    tmp.cleanup()


def test_write_all_data_sep():
    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)

    write_data(
        speakers,
        destination=tmp_path,
        separate=True
    )

    all_files = [p.name for p in tmp_path.glob("*")]

    assert any(["_tracks.csv" in f for f in all_files])
    assert any(["_param.csv" in f for f in all_files])
    assert any(["_logparam.csv" in f for f in all_files])
    assert any(["_points.csv" in f for f in all_files])
    assert any(["_recoded.TextGrid" in f for f in all_files])

    n_input = len(speakers)
    n_tg = len({x.file_name for x in speakers.values()})
    
    assert len(all_files) == (n_input * 4) + n_tg

    tmp.cleanup()


def test_pickling():
    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)
    tmp_file = tmp_path.joinpath("speaker.pickle")

    pickle_speakers(speakers, tmp_file)

    assert tmp_file.exists()

    re_read = unpickle_speakers(tmp_file)

    assert isinstance(re_read, SpeakerCollection)

    for k in re_read:
        assert k in speakers

    tmp.cleanup()