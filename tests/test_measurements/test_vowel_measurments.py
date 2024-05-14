from aligned_textgrid import AlignedTextGrid, Word, Phone
from fasttrackpy import CandidateTracks, process_corpus
from pathlib import Path
from functools import reduce

import polars as pl
import numpy as np

from new_fave.measurements.vowel_measurement import SpeakerCollection, \
    VowelClassCollection, \
    VowelClass, \
    VowelMeasurement

corpus_path = Path("tests", "test_data", "corpus")

NSTEP = 10
NFORMANT = 3

candidates = process_corpus(
    corpus_path,
    entry_classes=["Word", "Phone"],
    target_labels= "[AEIOU]",
    min_max_formant=4000,
    max_max_formant=7500,
    n_formants = NFORMANT,
    nstep=NSTEP
)
tgs_paths = corpus_path.glob("*TextGrid")

atgs = [
    AlignedTextGrid(textgrid_path=p, entry_classes=[Word, Phone])
    for p in tgs_paths
]

vms = [VowelMeasurement(t) for t in candidates]
speakers = SpeakerCollection(vms)

def test_sepeakers_size():
    keys = speakers.keys()

    n_groups = [len(a) for a in atgs]
    total_groups = reduce(lambda a, b: a+b, n_groups)
    assert len(keys) == total_groups

def test_class_nesting():
    speaker_keys = [s for s in speakers]
    for speaker in speaker_keys:
        vowel_space = speakers[speaker]
        assert isinstance(vowel_space, VowelClassCollection)

        vcs = [vc for vc in vowel_space]
        for vc in vcs:
            vowel_class = vowel_space[vc]
            assert isinstance(vowel_class, VowelClass)

            meases = [m for m in vowel_class.tracks]
            for m in meases:
                assert isinstance(m, VowelMeasurement)

def test_winner_reset():
    """
        Cached values for speaker-level parameters
        related to maximum formant and parameters
        should reset on a new winner.
    """
    speaker_keys = [s for s in speakers]
    first_speaker = speaker_keys[0]

    speaker = speakers[first_speaker]
    speaker_vms = speaker.vowel_measurements

    initial_mean = speaker.maximum_formant_means

    initial_index = speaker_vms[0].winner_index
    speaker_vms[0].winner = initial_index+1

    new_mean = speaker.maximum_formant_means

    assert ~np.isclose(initial_mean, new_mean)

def test_probs():
    """
    Test that the length of log probs
    is equal to the number of steps
    """
    for vm in vms:
        cand_mahal_log_prob = vm.cand_mahal_log_prob
        max_formant_log_prob = vm.max_formant_log_prob
        error_log_prob = vm.error_log_prob

        assert cand_mahal_log_prob.size == NSTEP
        assert max_formant_log_prob.size == NSTEP
        assert error_log_prob.size == NSTEP




def test_track_df():
    df = speakers.to_tracks_df()
    assert isinstance(df, pl.DataFrame)

    col_names = [
        "F1", "F2", "F3", "F1_s", "F2_s", "F3_s",
        'error', 'time', 'max_formant', 'n_formant',
        'smooth_method', 'file_name', 'id', 'group',
        'label', 'speaker_num', 'word', 'stress', 'dur',
        'pre_word', 'fol_word', 'pre_seg', 'fol_seg',
        'abs_pre_seg', 'abs_fol_seg', 'context']
    
    for name in col_names:
        assert name in df.columns

    agg = (
        df
        .group_by(["file_name", "group"])
        .agg(pl.len())
    )

    n_groups = [len(a) for a in atgs]
    total_groups = reduce(lambda a, b: a+b, n_groups)

    assert agg.shape[1] == total_groups

def test_param_df():
    df = speakers.to_param_df()
    assert isinstance(df, pl.DataFrame)

    col_names = ['param', 'F1', 'F2', 'F3', 
         'error', 'max_formant', 'file_name', 'id', 'group', 
         'label', 'speaker_num', 'word', 'stress', 'dur',
         'pre_word', 'fol_word', 'pre_seg', 'fol_seg',
         'abs_pre_seg', 'abs_fol_seg', 'context'
         ]
    
    for name in col_names:
        assert name in df.columns

    agg = (
        df
        .group_by(["file_name", "group"])
        .agg(pl.len())
    )

    n_groups = [len(a) for a in atgs]
    total_groups = reduce(lambda a, b: a+b, n_groups)

    assert agg.shape[1] == total_groups    



def test_point_df():
    df = speakers.to_point_df()
    assert isinstance(df, pl.DataFrame)

    col_names = ['F1', 'F2', 'F3',
                 'max_formant', 'smooth_error', 'time',
                 'rel_time', 'prop_time', 'id', 'label', 'file_name',
                 'group', 'speaker_num', 'word', 'stress',
                 'dur', 'pre_word', 'fol_word', 'pre_seg',
                 'fol_seg', 'abs_pre_seg', 'abs_fol_seg', 'context'
                 ]
    
    for name in col_names:
        assert name in df.columns

    agg = (
        df
        .group_by(["file_name", "group"])
        .agg(pl.len())
    )

    n_groups = [len(a) for a in atgs]
    total_groups = reduce(lambda a, b: a+b, n_groups)

    assert agg.shape[1] == total_groups    


