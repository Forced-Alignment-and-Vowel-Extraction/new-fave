from new_fave import fave_corpus
from new_fave import SpeakerCollection
from new_fave.speaker.speaker import Speaker
from pathlib import Path

CORPUS = Path("tests", "test_data", "corpus")

SPEAKERS = fave_corpus(
    corpus_path=CORPUS,
    speakers=0,
    recode_rules="cmu2labov",
    labelset_parser="cmu_parser",
    point_heuristic="fave",
    ft_config=Path("tests", "test_patterns", "test_ft_config.yml")
)

SPEAKERS_spfile = fave_corpus(
    corpus_path=CORPUS,
    speakers=Path("tests", "test_data", "speaker", "speaker3.yml"),
    recode_rules="cmu2labov",
    labelset_parser="cmu_parser",
    point_heuristic="fave",
    ft_config=Path("tests", "test_patterns", "test_ft_config.yml")
)

def test_fave_corpus():
    assert isinstance(SPEAKERS, SpeakerCollection)
    assert isinstance(SPEAKERS_spfile, SpeakerCollection)

    assert isinstance(SPEAKERS_spfile.speaker, Speaker)


def test_fave_corpus_demo_keys_and_unique():
    expected = {
        ("KY25A_1", "KY25A"),
        ("KY25A_1", "IVR"),
        ("josef-fruehwald_speaker", "group_0"),
    }
    assert set(SPEAKERS_spfile.keys()) == expected

    for key, vcc in SPEAKERS_spfile.items():
        ids = [vm.id for vc in vcc.values() for vm in vc]
        assert len(ids) == len(set(ids)), (
            f"duplicate candidate ids under {key}: "
            f"{len(ids) - len(set(ids))} duplicates"
        )

