from aligned_textgrid import  AlignedTextGrid, \
    SequenceTier, \
    SequenceInterval, \
    Word, \
    Phone, \
    Top
from aligned_textgrid.sequences.tiers import TierGroup
from fasttrackpy import CandidateTracks
from pathlib import Path
from fasttrackpy import process_corpus

## to test
from new_fave.utils.textgrid import get_top_tier, \
    get_tier_group, \
    get_textgrid, \
    get_all_textgrid
###


tg_path = Path("tests", "test_data", "corpus", "josef-fruehwald_speaker.TextGrid")
corpus_path = Path("tests", "test_data", "corpus")

def test_get_top_tier():
    atg = AlignedTextGrid(
        textgrid_path=str(tg_path),
        entry_classes=[Word, Phone]
    )

    interval = atg[0].Phone[200]
    assert isinstance(interval, SequenceInterval)
    tier = get_top_tier(interval)
    assert isinstance(tier, SequenceTier)
    assert issubclass(tier.superset_class, Top)


def test_get_tier_group():
    atg = AlignedTextGrid(
        textgrid_path=str(tg_path),
        entry_classes=[Word, Phone]
    )

    interval = atg[0].Phone[200]
    assert isinstance(interval, SequenceInterval)
    tier_group = get_tier_group(interval)
    assert isinstance(tier_group, TierGroup)

def test_get_textgrid():
    atg = AlignedTextGrid(
        textgrid_path=str(tg_path),
        entry_classes=[Word, Phone]
    )

    interval = atg[0].Phone[200]
    assert isinstance(interval, SequenceInterval)
    text_grid = get_textgrid(interval)
    assert text_grid is atg

    candidates = process_corpus(
        str(corpus_path),
        entry_classes=["Word", "Phone"],
        target_labels= "[AEIOU]",
        min_max_formant=4000,
        max_max_formant=7500,
        n_formants = 3,
        nstep=5
        )
    
    atg_paths = list(
        corpus_path.glob("*TextGrid")
    )

    all_tg = get_all_textgrid(candidates)

    assert len(atg_paths) == len(all_tg)
