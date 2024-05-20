from fasttrackpy import process_audio_textgrid
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
from new_fave.patterns.common_processing import resolve_resources, resolve_speaker
from new_fave.speaker.speaker import Speaker
import numpy as np

from pathlib import Path
import logging
logger = logging.getLogger("audio-textgrid")
logger.setLevel(level=logging.INFO)


def fave_audio_textgrid(
    audio_path: str|Path,
    textgrid_path: str|Path,
    speakers: int|list[int]|str|Path = 0,
    include_overlaps: bool = True,
    recode_rules: str|None = None,
    labelset_parser: str|None = None,
    point_heuristic: str|None = None,
    ft_config: str|None = "default"
)->SpeakerCollection:
    """
    Process a single audio/textgrid pair.

    Args:
        audio_path (str | Path): 
            Path to an audio file
        textgrid_path (str | Path): 
            Path to a textgrid
        speakers (int | list[int] | str | Path | optional): 
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

    speaker_demo, speakers = resolve_speaker(speakers)

    logger.info("FastTrack Processing")
    candidates = process_audio_textgrid(
        audio_path = audio_path,
        textgrid_path = textgrid_path,
        **fasttrack_kwargs
    )

    atg = get_textgrid(candidates[0].interval)

    tg_names = [tg.name for tg in atg]
    if speakers == "all":
        speakers = np.arange(len(atg))
    if len(speakers) > len(atg):
        raise ValueError(
            (
                f"{len(speakers)} speakers were set as targets "
                f"but textgrid has only {len(atg)} speakers."
            )
        )

    target_tgs = [tg_names[i] for i in speakers]
    target_candidates = [
        cand 
        for cand in candidates 
        if cand.group in target_tgs
    ]

    run_recode(
        atg,
        parser = parser,
        scheme=ruleset,
        target_tier="Phone"
    )  

    for cand in target_candidates:
        cand.label = cand.interval.label
        for track in cand.candidates:
            track.label = cand.label

    if not include_overlaps:
        mark_overlaps(atg)
        target_candidates = [
            cand 
            for cand in target_candidates
            if not cand.interval.overlapped
        ]      

    vms = [VowelMeasurement(t, heuristic=heuristic) for t in target_candidates]
    vowel_systems = SpeakerCollection(vms)
    if speaker_demo:
        vowel_systems.speaker = speaker_demo

    for vs in vowel_systems:
        logger.info(f"Optimizing {vs}")
        run_optimize(vowel_systems[vs])

    return vowel_systems


    
