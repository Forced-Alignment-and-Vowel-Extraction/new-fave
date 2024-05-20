from fasttrackpy import process_corpus
from aligned_textgrid import AlignedTextGrid
from fave_recode.fave_recode import run_recode, \
    get_rules, \
    get_parser, \
    RuleSet, \
    LabelSetParser
from fave_measurement_point.heuristic import Heuristic
from new_fave.measurements.vowel_measurement import VowelClassCollection, \
    VowelMeasurement, \
    SpeakerCollection
from new_fave.optimize.optimize import run_optimize
from new_fave.utils.textgrid import get_textgrid, get_all_textgrid, mark_overlaps
from new_fave.utils.local_resources import recodes, \
    parsers,\
    heuristics, \
    fasttrack_config,\
    generic_resolver
from new_fave.utils.fasttrack_config import read_fasttrack
from new_fave.patterns.common_processing import resolve_resources
from new_fave.speaker.speaker import Speaker
import numpy as np

from pathlib import Path
import re
import logging
logger = logging.getLogger("corpus")
logger.setLevel(level=logging.INFO)


def fave_corpus(
    corpus_path: str|Path,
    speakers: int|list[int]|str|Path = 0,
    include_overlaps: bool = True,
    recode_rules: str|None = None,
    labelset_parser: str|None = None,
    point_heuristic: str|None = None,
    ft_config: str|None = "default"
)->SpeakerCollection:
    """
    Process a corpus directory.

    Args:
        corpus_path (str | Path): 
            Path to a corpus directory
        speakers (int, str, Path, optional): 
            Which speaker(s) to produce data for.
            Can be a numeric index, or a path to a 
            speaker file, or "all"
        include_overlaps (bool, optional):
            Whether or not to include vowels that are overlapped
            with speech from other tiers. Defaults to `True`.            
        recode_rules (str | None, optional): 
            Either a string naming built-in set of
            recode rules, or path to a custom  ruleset. 
            Defaults to None.
        labelset_parser (str | None, optional): 
            Either a string naming a built-in labelset
            parser, or a path to a custom parser definition. 
            Defaults to None.
        point_heuristic (str | None, optional): 
            Either a string naming a built in point heuristic,
            or a path to a custom heuristic definition. 
            Defaults to None.
        ft_config (str | None, optional): 
            Either a string naming a built-in fasttrack config file,
            or a path to a custom config file. 
            Defaults to "default".

    Returns:
        (SpeakerCollection): 
            A [](`new_fave.SpeakerCollection`)
    """
    ruleset, parser, heuristic, fasttrack_kwargs,  = resolve_resources(
        recode_rules, labelset_parser, point_heuristic, ft_config
    )
    logger.info('FastTrack Processing')
    candidates = process_corpus(
        corpus_path=corpus_path,
        **fasttrack_kwargs
    )

    atgs = get_all_textgrid(candidates)

    target_candidates = candidates
    
    if type(speakers) is int:
        target_candidates = [
            cand
            for cand in candidates
            if cand.group == get_textgrid(cand.interval)[speakers].name
        ]

    speaker_path = None
    if type(speakers) is str and not speakers == "all":
        speaker_path = Path(speakers)
    if isinstance(speakers, Path):
        speaker_path = speakers

    speaker_demo = None
    if speaker_path:
        speaker_demo = Speaker(speaker_path)
    
    if speaker_demo:
        file_names = speaker_demo.df["file_name"].to_list()
        speaker_nums = speaker_demo.df["speaker_num"].to_list()
        speaker_nums = [s-1 for s in speaker_nums]
        target_speakers = list(zip(file_names, speaker_nums))

        target_candidates = [
            cand for cand in candidates
            for fn, num in zip(file_names, speaker_nums)
            if (
                cand.file_name, 
                int(re.search("^(\d+)-", cand.id).group(1))+1
            ) in target_speakers
        ]


    for atg in atgs:
        run_recode(
            atg, 
            parser=parser, 
            scheme=ruleset, 
            target_tier="Phone"
            )
        if not include_overlaps:
            mark_overlaps(atg)

    if not include_overlaps:
        target_candidates = [
            cand 
            for cand in target_candidates
            if not cand.interval.overlapped
        ]   
        
    for cand in target_candidates:
        cand.label = cand.interval.label
        for track in cand.candidates:
            track.label = cand.label

    vms = [VowelMeasurement(t, heuristic=heuristic) for t in target_candidates]
    vowel_systems = SpeakerCollection(vms)
    if speaker_demo:
        vowel_systems.speaker = speaker_demo

    for vs in vowel_systems:
        logger.info(f'Optimizing {vs}')
        run_optimize(vowel_systems[vs])

    return vowel_systems


    

