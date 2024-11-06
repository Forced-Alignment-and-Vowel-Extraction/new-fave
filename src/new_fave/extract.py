# !!! This is NOT the original extractFormants.py file !!!              ##

from new_fave import (fave_audio_textgrid, 
    fave_corpus, 
    fave_subcorpora, 
    write_data
)
from fasttrackpy.patterns.just_audio import create_audio_checker
from fasttrackpy.patterns.corpus import get_audio_files, get_corpus, CorpusPair
from fasttrackpy.utils.safely import safely, filter_nones
from new_fave.patterns.writers import check_outputs
from new_fave.patterns.common_processing import resolve_resources, resolve_speaker
from new_fave.measurements.reference import ReferenceValues

import numpy as np

from pathlib import Path
from glob import glob
import click
import cloup
from cloup import (Context, HelpFormatter, HelpTheme, Style,
    option_group, option)
from cloup.constraints import mutually_exclusive

import re

import yaml

import inspect

from typing import Any, Literal

import warnings

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

try:
    import magic
    no_magic = False
except:
    warnings.warn("libmagic not found. "\
                "Some audio file types won't be discovered by fasttrack. "\
                "(mp3, ogg, ...)")
    import sndhdr
    from sndhdr import SndHeaders
    no_magic = True

is_audio = create_audio_checker(no_magic=no_magic)

def ask(message: str) -> bool:
    response = click.confirm(
        f"{message}",
        default=True
    )
    return response

formatter_settings = HelpFormatter.settings(
    theme=HelpTheme(
        invoked_command=Style(fg='bright_yellow'),
        heading=Style(fg='bright_white', bold=True),
        constraint=Style(fg='magenta'),
        col1=Style(fg='green'),
    )
)

configs = cloup.option_group(
    "Configuration Options",
    cloup.option(
        "--recode-rules",
        type=click.STRING,
        default="cmu2labov",
        show_default=True,
        help=(
            "Recoding rules to adjust vowel interval labels. "
            "Values can be a string naming one of the built-in "
            "recode rules ('cmu2labov','cmu2phila', 'norecode'), or a path "
            "to a custom recoding yml file."
        )
    ),
    cloup.option(
        "--labelset-parser",
        type = click.STRING,
        default="cmu_parser",
        show_default=True,
        help = (
            "A labeleset parser. Values can be a string naming a "
            "built-in parser ('cmu_parser') "
            "or a path to a custom parser yml file. "
        )
    ),
    cloup.option(
        "--point-heuristic",
        type = click.STRING,
        default="fave",
        show_default=True,
        help=(
            "The point measurement heuristic to use. "
            "Values can be a built in heuristic ('fave') "
            "or a path to a custom heuristic file. "
        )
    ),
    cloup.option(
        "--vowel-place",
        type = click.STRING,
        default="default",
        show_default=True,
        help = (
            "A vowel place definition file. "
            "Values can be the name of a built in config ('defailt) "
            "or a path to a custom config file."
        )
    ), 
    cloup.option(
        "--f1-cutoff",
        type = click.FLOAT,
        default=1500,
        show_default=True,
        help = (
            "The maximum considerable F1 value."
        )
    ), 
    cloup.option(
        "--f2-cutoff",
        type = click.FLOAT,
        default=3500,
        show_default=True,
        help = (
            "The maximum considerable F2 value."
        )
    ),     
    cloup.option(
        "--ft-config",
        type = click.STRING,
        default="default",
        show_default=True,
        help = (
            "A fasttrack config file. "
            "Values can be the name of a built in config ('default') "
            "or a path to a custom config file."
        )
    ),
    cloup.option(
        "--fave-aligned",
        type=click.BOOL,
        is_flag=True,
        show_default=True,
        default=False,
        help = (
            "Include this flag if the textgrid was aligned with "
            "FAVE align."
        )
    ),
    cloup.option(
        "--exclude-overlaps",
        type=click.BOOL,
        is_flag=True,
        default=False,
        help=(
            "Include this flag if you want to "
            "exclude overlapping speech."
        )
    ),
    cloup.option(
        "--no-optimize",
        type=click.BOOL,
        is_flag=True,
        default=False,
        help=(
            "Include this flag if you want to "
            "skip fave optimization"
        )
    )
)

reference_values = cloup.option_group(
        "Reference Values",
        cloup.option(
        "--logparam-reference",
        type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
        default=None,
        show_default=False,
        help = (
            "A path to a collection of reference *_logparam.csv files."
        )
    ),
    cloup.option(
        "--param-reference",
        type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
        default=None,
        show_default=False,
        help = (
            "A path to a collection of reference *_param.csv files."
        )
    ),
    cloup.option(
        "--points-reference",
        type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
        default=None,
        show_default=False,
        help = (
            "A path to a collection of reference *_points.csv files."
        )
    ),  
    constraint=mutually_exclusive
)

speaker_opt = cloup.option(
    "--speakers",
    default=1,
    show_default=True,
    type=click.UNPROCESSED,
    help=("Which speakers to analyze. " 
          "Values can be: a numeric value (1 = first speaker), "
          "the string 'all', for all speakers, or "
          "a path to a speaker demographics file."
    )
)
outputs = cloup.option_group(
    "Output options",
    cloup.option(
        "--destination",
        type = click.Path(file_okay=False, dir_okay=True),
        default=Path("fave_results/"),
        show_default=True,
        help=(
            "Destination directory for resulting data "
            "files. If the directory doesn't exist, it will be "
            "created."
        )
    ),
    cloup.option(
        "--which",
        default=["all"],
        multiple=True,
        type=click.Choice(["all", "tracks", "points", "param", "log_param", "textgrid"]),
        show_default=True,
        help = (
            "Which output files to write. Default is 'all'. "
            "This option can be included multiple times to write "
            "just some of the options (e.g. 'tracks' and 'points'"
            )
    ),
    cloup.option(
        "--separate",
        is_flag=True,
        default=False,
        type=click.BOOL,
        help = (
            "Should each individual speaker be written to separate data files?"
        )
    ),
    help = "Options for writing output data.",
)

@cloup.group(name="fave-extract", show_subcommand_aliases=True)
def fave_extract():
    """Run new fave-extract"""
    pass

@fave_extract.command(
    aliases = ["audio-textgrid"],
    formatter_settings=formatter_settings,
    help = "Run fave-extract on a single audio+textgrid pair."
)
@cloup.argument(
    "audio_path",
    type = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to the audio file."
)
@cloup.argument(
    "textgrid_path",
    type = click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Path to the TextGrid file."
)
@speaker_opt
@configs
@reference_values
@outputs
def audio_textgrid(
    audio_path: str|Path,
    textgrid_path: str|Path,
    speakers: int|list[int]|str|Path,
    exclude_overlaps: bool,
    no_optimize:bool,
    recode_rules: str|None,
    labelset_parser: str|None,
    point_heuristic: str|None,
    vowel_place: str|None,
    f1_cutoff: float | np.float64,
    f2_cutoff: float | np.float64,    
    logparam_reference: str|None,
    param_reference: str|None,    
    points_reference: str|None,
    ft_config: str|None,
    fave_aligned: bool,
    destination: Path,
    which: list[Literal[
            "tracks", "points", "param", "log_param", "textgrid"
        ]],
    separate: bool
):
    audio_path = Path(audio_path)
    textgrid_path = Path(textgrid_path)
    destination = Path(destination)
    if "all" in which:
        which = [
            "tracks", "points", "param", "log_param", "textgrid"
        ]
    matched_which = check_outputs(audio_path, destination, which)

    overwrite = True
    if len(matched_which) > 0:
        overwrite = ask(
            (
            f"Some output files already exist for {audio_path.stem} at {destination}. \n"
            "Should they be overwritten? (y = overwrite, n = don't overwrite.)"
            )
        )
    if not overwrite:
        which = [x for x in which if x not in matched_which]
    
    if len(which) == 0:
        return 
    
    if logparam_reference or param_reference or points_reference:
        logging.info("Processing Reference Values")
    reference_values = ReferenceValues(
        logparam_corpus=logparam_reference,
        param_corpus=param_reference,
        points_corpus=points_reference
    )
    
    include_overlaps = not exclude_overlaps
    if type(speakers) is int:
        speakers = speakers - 1
    SpeakerData = fave_audio_textgrid(
        audio_path=audio_path,
        textgrid_path=textgrid_path,
        speakers=speakers,
        include_overlaps=include_overlaps,
        no_optimize=no_optimize,
        recode_rules=recode_rules,
        labelset_parser=labelset_parser,
        point_heuristic=point_heuristic,
        vowel_place_config=vowel_place,
        f1_cutoff = f1_cutoff,
        f2_cutoff = f2_cutoff,        
        ft_config=ft_config,
        reference_values = reference_values,
        fave_aligned=fave_aligned
    )
    
    if SpeakerData is not None:
        write_data(
            SpeakerData,
            destination=destination,
            which=which,
            separate=separate
        )

@fave_extract.command(
    aliases = ["corpus"],
    formatter_settings=formatter_settings,
    help = "Run fave-extract on a directory of audio+textgrid pairs."
)
@cloup.argument(
    "corpus_path",
    type = click.Path(file_okay=False, dir_okay=True),
    help="Path to a corpus directory."
)
@speaker_opt
@configs
@reference_values
@outputs
def corpus(
    corpus_path: str|Path,
    speakers: int|list[int]|str|Path,
    exclude_overlaps: bool,
    no_optimize:bool,    
    recode_rules: str|None,
    labelset_parser: str|None,
    point_heuristic: str|None,
    vowel_place: str|None,
    f1_cutoff: float | np.float64,
    f2_cutoff: float | np.float64,    
    ft_config: str|None,
    logparam_reference: str|None,
    param_reference: str|None,    
    points_reference: str|None,
    fave_aligned: bool,
    destination: Path,
    which: list[Literal[
            "tracks", "points", "param", "log_param", "textgrid"
        ]],
    separate: bool    
):
    if "all" in which:
        which = [
            "tracks", "points", "param", "log_param", "textgrid"
        ]
    all_audio = get_audio_files(corpus_path = corpus_path)
    all_which = [which for a in all_audio]
    result_which = []
    for a,w in zip(all_audio, all_which):
        overwrite = True
        matched_which = check_outputs(a, destination, which)

        if len(matched_which) > 0:
            overwrite = ask(
                (
                f"Some output files already exist for {a.stem} at {destination}. \n"
                "Should they be overwritten? (y = overwrite, n = don't overwrite.)"
                )
            )
        new_which = w
        if not overwrite:
            new_which = [x for x in w if x not in matched_which]
        result_which.append(new_which)
    
    audio_to_process = [a for a,w in zip(all_audio, result_which) if len(w) > 0]

    result_which,audio_to_process =  filter_nones(result_which, [result_which, audio_to_process])

    corpus = get_corpus(audio_to_process)

    include_overlaps = not exclude_overlaps
    if type(speakers) is int:
        speakers = speakers - 1

    if logparam_reference or param_reference or points_reference:
        logging.info("Processing Reference Values")

    reference_values = ReferenceValues(
        logparam_corpus=logparam_reference,
        param_corpus=param_reference,
        points_corpus=points_reference
    )
    for pair, w in zip(corpus, result_which):
        SpeakerData = fave_audio_textgrid(
            audio_path=pair.wav,
            textgrid_path=pair.tg,
            speakers = speakers,
            include_overlaps=include_overlaps,
            no_optimize=no_optimize,
            recode_rules=recode_rules,
            labelset_parser=labelset_parser,
            point_heuristic=point_heuristic,
            vowel_place_config=vowel_place,
            f1_cutoff = f1_cutoff,
            f2_cutoff = f2_cutoff,            
            ft_config=ft_config,
            reference_values = reference_values,        
            fave_aligned=fave_aligned
        )
        if SpeakerData is not None:
            write_data(
                SpeakerData,
                destination=destination,
                which = w,
                separate=separate
            )

@fave_extract.command(
    aliases = ["subcorpora"],
    formatter_settings=formatter_settings,
    help = "Run fave-extract on multiple subdirectories."
)
@cloup.argument(
    "subcorpora",
    type = click.UNPROCESSED,
    nargs=-1,
    help="A glob that resolves to subcorpora directories"
)
@speaker_opt
@configs
@reference_values
@outputs
def subcorpora(
    subcorpora: list[str|Path],
    speakers: int|list[int]|str|Path,
    exclude_overlaps: bool,
    no_optimize:bool,    
    recode_rules: str|None,
    labelset_parser: str|None,
    point_heuristic: str|None,
    vowel_place: str|None,
    f1_cutoff: float | np.float64,
    f2_cutoff: float | np.float64,    
    ft_config: str|None,
    logparam_reference: str|None,
    param_reference: str|None,    
    points_reference: str|None,    
    fave_aligned: bool,
    destination: Path,
    which: list[Literal[
            "tracks", "points", "param", "log_param", "textgrid"
        ]],
    separate: bool 
):
    corpora = [Path(p) for p in subcorpora]
    if "all" in which:
        which = [
            "tracks", "points", "param", "log_param", "textgrid"
        ]
    all_audio = [a for c in corpora for a in get_audio_files(corpus_path = c)]
    all_which = [which for a in all_audio]
    result_which = []
    for a,w in zip(all_audio, all_which):
        overwrite = True
        matched_which = check_outputs(a, destination, which)

        if len(matched_which) > 0:
            overwrite = ask(
                (
                f"Some output files already exist for {a.stem} at {destination}. \n"
                "Should they be overwritten? (y = overwrite, n = don't overwrite.)"
                )
            )
        new_which = w
        if not overwrite:
            new_which = [x for x in w if x not in matched_which]
        result_which.append(new_which)
    
    audio_to_process = [a for a,w in zip(all_audio, result_which) if len(w) > 0]

    result_which,audio_to_process =  filter_nones(result_which, [result_which, audio_to_process])

    corpus = get_corpus(audio_to_process)    

    include_overlaps = not exclude_overlaps
    if type(speakers) is int:
        speakers = speakers - 1

    if logparam_reference or param_reference or points_reference:
        logging.info("Processing Reference Values")
        
    reference_values = ReferenceValues(
        logparam_corpus=logparam_reference,
        param_corpus=param_reference,
        points_corpus=points_reference
    )        
    for pair, w in zip(corpus, result_which):
        SpeakerData = fave_audio_textgrid(
            audio_path=pair.wav,
            textgrid_path=pair.tg,
            speakers = speakers,
            include_overlaps=include_overlaps,
            no_optimize=no_optimize,
            recode_rules=recode_rules,
            labelset_parser=labelset_parser,
            point_heuristic=point_heuristic,
            vowel_place_config=vowel_place,
            f1_cutoff = f1_cutoff,
            f2_cutoff = f2_cutoff,            
            ft_config=ft_config,
            reference_values = reference_values,         
            fave_aligned=fave_aligned
        )
        if SpeakerData is not None:
            write_data(
                SpeakerData,
                destination=destination,
                which = w,
                separate=separate
            )
        else:
            logging.info("Problem writing data")
    pass

# @fave_extract.command(
#     aliases = ["show"],
#     formatter_settings=formatter_settings,
#     help = "Show fave-extract configs."
# )
# @speaker_opt
# @configs
# @outputs
# def show(
#     recode_rules: str|None,
#     labelset_parser: str|None,
#     point_heuristic: str|None,
#     vowel_place: str|None,
#     ft_config: str|None,
#     vowel_place_config: str|None,
#     fave_aligned: bool,
#     destination: Path,
#     which: list[Literal[
#             "tracks", "points", "param", "log_param", "textgrid"
#         ]],
#     speakers: str|None,
#     separate: bool 
# ):
#     ruleset, parser, heuristic, fasttrack_kwargs, vowel_place_dict = resolve_resources(
#         recode_rules, labelset_parser, point_heuristic, ft_config, vowel_place_config
#     )
#     pass


if __name__ == "__main__":
    fave_extract()
