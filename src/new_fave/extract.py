from fasttrackpy import process_corpus

from aligned_textgrid import AlignedTextGrid,\
    custom_classes

from fave_recode.fave_recode import run_recode, \
    get_rules, \
    get_parser
from fave_recode.labelset_parser import LabelSetParser
from fave_recode.rule_classes import RuleSet

from fave_measurement_point.formants import FormantArray
from fave_measurement_point.heuristic import Heuristic

from new_fave.measurements.vowel_measurement import VowelMeasurement,\
    VowelClassCollection, \
    VowelClass

from new_fave.optimize.optimize import optimize_vowel_measures

from new_fave.utils.textgrid import get_textgrid, \
    get_tier_group, \
    get_top_tier

from new_fave.utils.local_resources import recodes,\
    parsers, \
    measurements,\
    fasttrack_config

from new_fave.speaker.speaker import Speaker

from pathlib import Path
import click
import cloup
from cloup import Context, HelpFormatter, HelpTheme, Style,\
    option_group, option

import yaml

import inspect


@cloup.command()
@cloup.argument(
    "corpus_path",
    help="Path to a corpus",
    type = click.Path(
        exists=True,
        dir_okay=True,
        file_okay=False,
        readable=True
    )
)
@option_group(
    "Configs",
    "Config files",
    option(
        "--fasttrack_config",
        default=fasttrack_config["default"],
        type=click.File(mode='r')
    ),
    option(
        "--labelset_parser",
        default=parsers["cmu_parser"],
        type=click.Path(exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True)
    ),
    option(
        "--recode_rules",
        default=recodes["cmu2labov"],
        type=click.Path(exists=True)
    ),
    option(
        "--measurement_points",
        default = measurements["fave"],
        type=click.Path(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True
        )
    )
)

def fave_extract(
    corpus_path: Path,
    fasttrack_config: Path,
    labelset_parser: Path,
    recode_rules: Path,
    measurement_points: Path,
    **kwargs
):
    """_summary_
    """

    
    fast_track_args = yaml.safe_load(fasttrack_config)
    corpus_args = {
        x:fast_track_args[x]
        for x in inspect.signature(process_corpus).parameters
        if x in fast_track_args
    }

    vms = process_corpus(
        corpus_path=corpus_path,
        **corpus_args
    )

    click.echo(f"{len(vms)}")

if __name__ == "__main__":
    fave_extract()
