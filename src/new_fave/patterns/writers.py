from new_fave.measurements.vowel_measurement import SpeakerCollection, VowelClassCollection
from new_fave.utils.textgrid import get_textgrid
from aligned_textgrid import AlignedTextGrid
from pathlib import Path
from typing import Literal
import polars as pl
import logging
from copy import copy, deepcopy
import cloudpickle
import sys
import warnings

logger = logging.getLogger("write-data")

def write_df(
    df: pl.DataFrame,
    destination: Path, 
    appendix: str, 
    separate: bool =False
):
    """
    Write the data frame, with the given appdendix.

    #### Intended Usage
    This is not intended to be used on its own. Rather
    it is a convenience function for [](`~new_fave.patterns.writers.write_data`).

    Args:
        df (pl.DataFrame): A polars dataframe.
        destination (Path): The destination directory
        appendix (str): Appendix to add
        separate (bool, optional): Split data by filename and group. Defaults to False.
    """
    file_group = (df
            .select("file_name", "group")
            .unique()
            .with_columns(
                pl.concat_str(
                    [pl.col("file_name"),
                     pl.col("group")],
                     separator="_"
                ).alias("newname")
            ) 
    )

    if separate:
        unique_entries = file_group.rows_by_key("newname", named = True)
    
        for entry in unique_entries:
            file = unique_entries[entry][0]["file_name"]
            group = unique_entries[entry][0]["group"]
            entry_stem = Path(str(entry) + "_" + appendix)
            entry_path = destination.joinpath(entry_stem).with_suffix(".csv")

            out_df = (
                df
                .filter(
                    (pl.col("file_name") == file) &
                    (pl.col("group") == group)
                )
            )

            out_df.write_csv(file = entry_path)
        
        return
    
    unique_entries = (
        file_group
        .select("file_name")
        .unique()
    )["file_name"].to_list()
    for file in unique_entries:
        entry_stem = Path(str(file) + "_" + appendix)
        entry_path = destination.joinpath(entry_stem).with_suffix(".csv")
        out_df = (
            df
            .filter(pl.col("file_name") == file)
        )

        out_df.write_csv(file = entry_path)


            

def write_data(
    vowel_spaces: SpeakerCollection,
    destination:str|Path = Path("."),
    which: Literal["all"] | 
        list[Literal[
            "tracks", "points", "param", "log_param", "textgrid"
        ]] = "all",
    separate: bool = False
):
    """
    Save data. 
    
    #### Intended usage 

    There are multiple data output types, including
    
    - `tracks`: Vowel formant tracks
    - `points`: Point measurements
    - `param`: DCT parameters on Hz
    - `log_param`: DCT parameters on log(Hz)
    - `textgrid`: The recoded textgrid
    
    By default, they will all be saved.

    Args:
        vowel_spaces (SpeakerCollection): 
            An entire `SpeakerCollection`
        destination (str | Path, optional): 
            Destination directory. Defaults to `Path(".")`.
        which (Literal["all"] | list[Literal[ "tracks", "points", "param", "log_param", "textgrid" ]], optional): 
            Which data to save. The values are described above. Defaults to "all".
        separate (bool, optional): 
            Whether or not to write separate `.csv`s for each individual speaker.
            Defaults to False.

    """
    if "all" in which:
        which = [
            "tracks", 
            "points", 
            "param", 
            "log_param", 
            "textgrid"
        ]

    if destination and not isinstance(destination, Path):
        destination = Path(destination)

    if destination.exists() and destination.is_file():
        raise ValueError(
            (
                f"The provided destination, {str(destination)}, "
                "is a file  not a directory"
            )
        )

    
    if not destination.exists():
        destination.mkdir()
    
    if "tracks" in which:
        logger.info("Writing track data.")
        write_df(vowel_spaces.to_tracks_df(), destination, "tracks", separate)

    if "points" in which:
        logger.info("Writing point data.")
        write_df(vowel_spaces.to_point_df(), destination, "points", separate)
    
    if "param" in which:
        logger.info("Writing DCT(Hz) data.")
        write_df(vowel_spaces.to_param_df(output="param"), destination, "param", separate)

    if "log_param" in which:
        logger.info("Writing DCT(log(Hz)) data.")
        write_df(vowel_spaces.to_param_df(output="log_param"), destination, "logparam", separate)
        
    if "textgrid" in which:
        logger.info("Writing recoded textgrid.")
        tg_name = set(
            [(vs.textgrid, vs.file_name) for vs in vowel_spaces.values()]
        )

        for pair in tg_name:
            out_name = Path(pair[1] + "_recoded").with_suffix(".TextGrid")
            out_path = destination.joinpath(out_name)
            pair[0].save_textgrid(out_path)

def check_outputs(
    stem: Path|str,
    destination: Path|str,
    which: Literal["all"] | 
        list[Literal[
            "tracks", "points", "param", "log_param", "textgrid"
        ]] = "all"
)->list[Literal["tracks", "points", "param", "log_param", "textgrid"]]:
    """
    Check to see if outputs already exist
    for a given file stem.

    Args:
        stem (Path | str): 
            The filestem
        destination (Path | str):
            The destination where some output files may exist.
        which (Literal["all"] | list[Literal[ "tracks", "points", "param", "log_param", "textgrid" ]], optional): 
            Which data to save. The values are described above. Defaults to "all".

    Returns:
        (list[Literal["tracks", "points", "param", "log_param", "textgrid"]]):
            The output types for which data exists.
    """
    stem = Path(stem).stem
    destination = Path(destination)
    if "all" in which:
        which = [
            "tracks", 
            "points", 
            "param", 
            "log_param", 
            "textgrid"
        ]
    affixes = copy(which)
    for i,a in enumerate(affixes):
        if a == "log_param":
            affixes[i] = "logparam"
        if a == "textgrid":
            affixes[i] = "recoded"
    to_glob = [
        str(stem)+"*_"+a+"*"
        for a in affixes
    ]

    matched_which = []
    for tg, w in zip(to_glob, which):
        globbed = list(destination.glob(tg))
        if len(globbed)>0:
            matched_which.append(w)

    return matched_which


def pickle_speakers(
        speakers: SpeakerCollection|VowelClassCollection,
        path: str | Path
):
    """
    This will serialize a SpeakerCollection to a pickle
    file, that can be re-read in a new python session.

    **Note**: new-fave uses the cloudpickle library, 
    rather than the standard pickle library, which comes
    with the following limitations, according to the
    cloudpickle documentation:

    > Cloudpickle can only be used to send objects between the exact same version of Python.
    > 
    > Using cloudpickle for long-term object storage is not supported and strongly discouraged.

    Args:
        speakers (SpeakerCollection): 
            A SpeakerCollection to serialize
        path (str | Path):
            The destination file to save the
            pickle file.
    """
    path = Path(path)

    if isinstance(speakers, VowelClassCollection):
        speakers_to_write = SpeakerCollection()
        speakers_to_write[
            (speakers.file_name, speakers.group)
        ]
    if not isinstance(speakers, SpeakerCollection):
        raise ValueError("pickle_speakers can only pickle a SpeakerCollection")
    
    with path.open('wb') as f:
        sys.setrecursionlimit(30000)
        cloudpickle.dump(speakers, f)


def unpickle_speakers(
        path: str | Path
) -> SpeakerCollection:
    """
    Unpickle a pickled SpeakerCollection

    Args:
        path (str | Path):
            Path to a pickled speaker collection

    Returns:
        (SpeakerCollection):
            The unpickled SpeakerCollection
    """
    path = Path(path)
    with path.open('rb') as f:
        sys.setrecursionlimit(30000)
        speakers = cloudpickle.load(f)
    
    if not isinstance(speakers, SpeakerCollection):
        warnings.warn("An unexpected object type was returned.")

    return speakers