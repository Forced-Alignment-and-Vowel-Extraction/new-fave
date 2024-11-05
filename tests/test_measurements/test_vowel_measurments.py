from aligned_textgrid import AlignedTextGrid, Word, Phone
from fasttrackpy import CandidateTracks, OneTrack, process_corpus
from pathlib import Path
from functools import reduce
from copy import copy

import polars as pl
import numpy as np

from new_fave.measurements.vowel_measurement import (SpeakerCollection, 
    VowelClassCollection, 
    VowelClass, 
    VowelMeasurement
)

from new_fave.speaker.speaker import Speaker

corpus_path = Path("tests", "test_data", "corpus")
speaker_demo= Speaker(corpus_path.joinpath("speakers.csv"))

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
speakers.speaker = speaker_demo

def test_sepeakers_size():
    keys = speakers.keys()

    n_groups = [len(a) for a in atgs]
    total_groups = reduce(lambda a, b: a+b, n_groups)
    assert len(keys) == total_groups

def test_class_nesting():
    """
    This is the intended nesting of classes
    and default iteration behavior
    """
    
    # SpeakerCollection(defaultdict)
    speaker_keys = [s for s in speakers]
    for speaker in speaker_keys:
        vowel_space = speakers[speaker]
        assert isinstance(vowel_space, VowelClassCollection)

        # VowelClassCollection(defaultdict)
        vc_keys = [vc for vc in vowel_space]
        for vc in vc_keys:
            vowel_class = vowel_space[vc]
            assert isinstance(vowel_class, VowelClass)

            # class VowelClass(Sequence)
            meases = [m for m in vowel_class]
            for m in meases:
                assert isinstance(m, VowelMeasurement)

                # class VowelMeasurement(Sequence)
                tracks = [t for t in m]
                for t in tracks:
                    assert isinstance(t, OneTrack)

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

    initial_mean = copy(speaker.winner_maxformant_mean)
    assert "winner_maxformant_mean" in speaker.__dict__

    initial_index = speaker_vms[0].winner_index
    speaker_vms[0].winner = initial_index+2
    assert not "winner_maxformant_mean" in speaker.__dict__

    new_mean = speaker.winner_maxformant_mean

    assert ~np.isclose(initial_mean, new_mean)

## Vowel Class

def test_vowel_class_setting():
    """
    Every vm inside a vowel class should have
    that vowel class object set
    """

    all_vcs = [
        vc 
        for vsp in speakers.values() 
        for vc in vsp.values()
    ]

    for vc in all_vcs:
        for vm in vc:
            assert vm.vowel_class is vc

def test_winner_getting():
    """
    Test that the right number of winners are gotten
    """
    all_vcs = [
        vc 
        for vsp in speakers.values() 
        for vc in vsp.values()
    ]

    for vc in all_vcs:
        all_tracks = [t for t in vc]
        all_winners = [t.winner for t in all_tracks]
        winners = vc.winners

        winner_gotten = [w in all_winners for w in winners]
        assert all(winner_gotten)

def test_winner_param():
    """
    test that the winner params are the 
    right size
    """
    all_vcs = [
        vc 
        for vsp in speakers.values() 
        for vc in vsp.values()
    ]

    for vc in all_vcs:
        params = vc.winner_param
        expected_shape = (6, NFORMANT, len(vc))
        for s1, s2 in zip(params.shape, expected_shape):
            assert s1 == s2

## Vowel Measurement
def test_probs():
    """
    Test that the length of log probs
    is equal to the number of steps
    And all log probs are finite and <= 0
    """
    target_properties = [
        x 
        for x in VowelMeasurement.__dict__.keys()
        if "logprob" in x
    ]

    for vm in vms:
       for target in target_properties:
           log_prob = getattr(vm, target)
           assert log_prob.size == NSTEP, f"{target}"
           assert np.all(~np.isnan(log_prob))
           assert np.all(log_prob <= 0)

## output tests
def test_vm_context():
    vm0 = vms[0]

    df = vm0.vm_context
    cols =  [
            "optimized",
            "id",
            "word",
            "stress",
            "dur",
            "pre_word",
            "fol_word",
            "pre_seg",
            "fol_seg",
            "abs_pre_seg",
            "abs_fol_seg",
            "context"]
    for c in cols:
        assert c in df.columns

    for c in df.columns:
        assert c in cols

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
                 'fol_seg', 'abs_pre_seg', 'abs_fol_seg', 'context',
                 'gender'
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


