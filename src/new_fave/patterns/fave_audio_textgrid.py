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
from new_fave.utils.textgrid import get_textgrid, get_all_textgrid
from new_fave.utils.local_resources import recodes, \
    parsers,\
    heuristics, \
    fasttrack_config,\
    generic_resolver
from new_fave.utils.fasttrack_config import read_fasttrack


from pathlib import Path



def fave_audio_textgrid(
    audio_path: str|Path,
    textgrid_path: str|Path,
    speakers: int|list[int] = 0,
    recode_rules: str|None = None,
    labelset_parser: str|None = None,
    point_heuristic: str|None = None,
    ft_config: str|None = "default"
):
    fasttrack_kwargs = generic_resolver(
        resolve_func=read_fasttrack,
        to_resolve=ft_config,
        resource_dict=fasttrack_config,
        default_value=dict()
    )
    
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

    if type(speakers) is int:
        speakers = [speakers]

    candidates = process_audio_textgrid(
        audio_path = audio_path,
        textgrid_path = textgrid_path,
        **fasttrack_kwargs
    )

    atg = get_textgrid(candidates[0].interval)
    tg_names = [tg.name for tg in atg]
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

    vms = [VowelMeasurement(t) for t in target_candidates]
    vowel_systems = SpeakerCollection(vms)

    for vs in vowel_systems.values():
        run_optimize(vs)

    return (vowel_systems, atg)


    

