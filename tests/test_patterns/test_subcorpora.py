from new_fave import fave_subcorpora
from new_fave import SpeakerCollection
from new_fave.speaker.speaker import Speaker
from pathlib import Path

SUBCORPORA = Path("tests", "test_data", "subcorpora", "*")
SPEAKER_GLOB =  Path("tests", "test_data", "subcorpora", "*", "*.yml")
SPFILE = Path("tests", "test_data", "speaker", "speaker3.yml")

SPEAKERS = fave_subcorpora(
    subcorpora_glob=SUBCORPORA,
    speakers=0,
    recode_rules="cmu2labov",
    labelset_parser="cmu_parser",
    point_heuristic="fave",
    ft_config=Path("tests", "test_patterns", "test_ft_config.yml")
)

SPEAKERS_spglob = fave_subcorpora(
    subcorpora_glob=SUBCORPORA,
    speakers_glob=SPEAKER_GLOB,
    recode_rules="cmu2labov",
    labelset_parser="cmu_parser",
    point_heuristic="fave",
    ft_config=Path("tests", "test_patterns", "test_ft_config.yml")
)

SPEAKERS_spfile = fave_subcorpora(
    subcorpora_glob=SUBCORPORA,
    speakers=SPFILE,
    recode_rules="cmu2labov",
    labelset_parser="cmu_parser",
    point_heuristic="fave",
    ft_config=Path("tests", "test_patterns", "test_ft_config.yml")
)

def test_fave_subcorpora():
    assert isinstance(SPEAKERS, SpeakerCollection)
    assert isinstance(SPEAKERS_spglob, SpeakerCollection)
    assert isinstance(SPEAKERS_spfile, SpeakerCollection)

    assert isinstance(SPEAKERS_spglob.speaker, Speaker)
    assert isinstance(SPEAKERS_spfile.speaker, Speaker)

    assert len(SPEAKERS) > 0
    assert len(SPEAKERS_spglob) > 0
    assert len(SPEAKERS_spfile) > 0