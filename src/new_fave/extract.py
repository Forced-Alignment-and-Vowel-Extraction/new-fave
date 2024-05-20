from new_fave import fave_audio_textgrid,\
    fave_corpus,\
    fave_subcorpora,\
    write_data

from pathlib import Path
import click
import cloup
from cloup import Context, HelpFormatter, HelpTheme, Style,\
    option_group, option

import yaml

import inspect

from typing import Any, Literal

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


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
            "recode rules ('cmu2labov' and 'cmu2phila'), or a path "
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
@cloup.option(
    "--speakers",
    default=1,
    show_default=True,
    help=("Which speakers to analyze. " 
          "Values can be: a numeric value (1 = first speaker), "
          "the string 'all', for all speakers, or "
          "a path to a speaker demographics file."
    )
)
@configs
@outputs
def audio_textgrid(
    audio_path: str|Path,
    textgrid_path: str|Path,
    speakers: int|list[int]|str|Path,
    exclude_overlaps: bool,
    recode_rules: str|None,
    labelset_parser: str|None,
    point_heuristic: str|None,
    ft_config: str|None,
    fave_aligned: bool,
    destination: Path,
    which: list[Literal[
            "tracks", "points", "param", "log_param", "textgrid"
        ]],
    separate: bool
):
    include_overlaps = not exclude_overlaps
    if type(speakers) is int:
        speakers = speakers - 1
    SpeakerData = fave_audio_textgrid(
        audio_path=audio_path,
        textgrid_path=textgrid_path,
        speakers=speakers,
        include_overlaps=include_overlaps,
        recode_rules=recode_rules,
        labelset_parser=labelset_parser,
        point_heuristic=point_heuristic,
        ft_config=ft_config,
        fave_aligned=fave_aligned
    )
    
    write_data(
        SpeakerData,
        destination=destination,
        which=which,
        separate=separate
    )

if __name__ == "__main__":
    fave_extract()
