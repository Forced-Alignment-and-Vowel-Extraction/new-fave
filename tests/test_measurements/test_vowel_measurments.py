from aligned_textgrid import AlignedTextGrid, Word, Phone
from fasttrackpy import CandidateTracks, process_corpus
from pathlib import Path
from functools import reduce

import polars as pl

from new_fave.measurements.vowel_measurement import SpeakerCollection, \
    VowelClassCollection, \
    VowelClass, \
    VowelMeasurement

corpus_path = Path("tests", "test_data", "corpus")

candidates = process_corpus(
    corpus_path,
    entry_classes=["Word", "Phone"],
    target_labels= "[AEIOU]",
    min_max_formant=4000,
    max_max_formant=7500,
    n_formants = 3,
    nstep=10
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


