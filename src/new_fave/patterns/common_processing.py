from fave_recode.fave_recode import run_recode, \
    get_rules, \
    get_parser, \
    RuleSet, \
    LabelSetParser

from new_fave.utils.local_resources import recodes, \
    parsers,\
    heuristics, \
    fasttrack_config,\
    generic_resolver

from new_fave.speaker.speaker import Speaker

from fave_measurement_point.heuristic import Heuristic

from new_fave.utils.fasttrack_config import read_fasttrack

from pathlib import Path
import numpy as np

def resolve_resources(
    recode_rules: str|None = None,
    labelset_parser: str|None = None,
    point_heuristic: str|None = None,
    ft_config: str|None = "default"
) -> tuple[RuleSet, LabelSetParser, Heuristic, dict]:
    
    ruleset = generic_resolver(
        resolve_func = get_rules,
        to_resolve = recode_rules,
        resource_dict = recodes,
        default_value=RuleSet()
    )

    parser = generic_resolver(
        resolve_func = get_parser,
        to_resolve = labelset_parser,
        resource_dict = parsers,
        default_value = LabelSetParser()
    )

    heuristic = generic_resolver(
        resolve_func=lambda x: Heuristic(heuristic_path=x),
        to_resolve=point_heuristic,
        resource_dict=heuristics,
        default_value=Heuristic()
    )

    fasttrack_kwargs = generic_resolver(
        resolve_func=read_fasttrack,
        to_resolve=ft_config,
        resource_dict=fasttrack_config,
        default_value=dict()
    )
    return (ruleset, parser, heuristic, fasttrack_kwargs)

def resolve_speaker(
        speakers: int|list[int]|str|Path
    ) -> tuple[Speaker|None, list[int]]:
    if type(speakers) is int:
        speakers = [speakers]
        return (None, speakers)
    
    if speakers == "all":
        return (None, speakers)

    speaker_path = None
    if type(speakers) is str:
        speaker_path = Path(speakers)
    if isinstance(speakers, Path):
        speaker_path = speakers

    if speaker_path:
        speaker_demo = Speaker(speaker_path)
        speakers = speaker_demo.df["speaker_num"].to_list()
        speakers = [s-1 for s in speakers]

        return(speaker_demo, speakers)
    
    return (None, [0])