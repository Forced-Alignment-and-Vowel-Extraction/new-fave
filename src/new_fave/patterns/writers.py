from new_fave.measurements.vowel_measurement import SpeakerCollection
from new_fave.utils.textgrid import get_textgrid
from aligned_textgrid import AlignedTextGrid
from pathlib import Path
from typing import Literal
import polars as pl


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
    if separate:
        unique_entries = (df
            .select("file_name", "group")
            .unique()
            .with_columns(
                pl.concat_str(
                    [pl.col("file_name"),
                     pl.col("group")],
                     separator="_"
                ).alias("newname")
            ) 
            .rows_by_key("newname", named = True)
        )

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
    
    file = Path(df["file_name"][0] + "_" + appendix).with_suffix(".csv")
    out_path = destination.joinpath(file)
    df.write_csv(out_path)
            

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
    if which == "all":
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
        write_df(vowel_spaces.to_tracks_df(), destination, "tracks", separate)

    if "points" in which:
        write_df(vowel_spaces.to_point_df(), destination, "points", separate)
    
    if "param" in which:
        write_df(vowel_spaces.to_param_df(output="param"), destination, "param", separate)

    if "log_param" in which:
        write_df(vowel_spaces.to_param_df(output="log_param"), destination, "logparam", separate)
        
    if "textgrid" in which:
        tg_name = set(
            [(vs.textgrid, vs.file_name) for vs in vowel_spaces.values()]
        )

        for pair in tg_name:
            out_name = Path(pair[1] + "_recoded").with_suffix(".TextGrid")
            out_path = destination.joinpath(out_name)
            pair[0].save_textgrid(out_path)
