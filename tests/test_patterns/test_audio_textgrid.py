from new_fave import fave_audio_textgrid
from new_fave import SpeakerCollection
from new_fave.speaker.speaker import Speaker
from pathlib import Path

WAV = Path("tests", "test_data", "corpus", "KY25A_1.wav")
TG = Path("tests", "test_data", "corpus", "KY25A_1.TextGrid")
SPEAKERS = fave_audio_textgrid(
    audio_path=WAV,
    textgrid_path=TG,
    speakers = 0,
    recode_rules="cmu2labov",
    labelset_parser="cmu_parser",
    ft_config=Path("tests", "test_patterns", "test_ft_config.yml")
)
SPEAKERS_no_overlap = fave_audio_textgrid(
    audio_path=WAV,
    textgrid_path=TG,
    include_overlaps=False,
    speakers = 0,
    recode_rules="cmu2labov",
    labelset_parser="cmu_parser",
    ft_config=Path("tests", "test_patterns", "test_ft_config.yml")
)

SPEAKERS_spfile = fave_audio_textgrid(
    audio_path=WAV,
    textgrid_path=TG,
    speakers=Path("tests", "test_data", "speaker", "speaker2.yml"),
    recode_rules="cmu2labov",
    labelset_parser="cmu_parser",
    ft_config=Path("tests", "test_patterns", "test_ft_config.yml")
)

def test_audio_textgrid():
    """
    Test basic audio_textgrid processing
    """
    assert isinstance(SPEAKERS, SpeakerCollection)

    s_vm = list(SPEAKERS.values())[0].vowel_measurements
    so_vm = list(SPEAKERS_no_overlap.values())[0].vowel_measurements

    assert len(so_vm) < len(s_vm)

    assert isinstance(SPEAKERS_spfile.speaker, Speaker)

