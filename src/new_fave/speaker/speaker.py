import polars as pl
from pathlib import Path
import yaml
import csv
import warnings
from glob import glob
import argparse

def yaml2speaker(path: Path) -> pl.DataFrame:
    with path.open() as f:
        obj = yaml.safe_load(f)

    return pl.DataFrame(obj)

def csv2speaker(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path)
    return df

def excel2speaker(path: Path) -> pl.DataFrame:
    df = pl.read_excel(path)
    return df

def oldfave2speaker(path: Path) -> pl.DataFrame:
    speaker_parser = argparse.ArgumentParser(description="parses a .speaker file",
                                     fromfile_prefix_chars="+")
    speaker_parser.add_argument("--name")
    speaker_parser.add_argument("--first_name")
    speaker_parser.add_argument("--last_name")
    speaker_parser.add_argument("--age")
    speaker_parser.add_argument("--sex")
    speaker_parser.add_argument("--ethnicity")
    speaker_parser.add_argument("--years_of_schooling")
    speaker_parser.add_argument("--location")
    speaker_parser.add_argument("--city")
    speaker_parser.add_argument("--state")
    speaker_parser.add_argument("--year")
    speaker_parser.add_argument("--speakernum")
    speaker_parser.add_argument("--tiernum")
    speaker_parser.add_argument("--vowelSystem",
        choices = ['phila', 'Phila', 'PHILA', 'NorthAmerican', 'simplifiedARPABET'])

    speaker_opts = speaker_parser.parse_args(["+"+str(path)])
    speaker_opts_dict = speaker_opts.__dict__
    speaker_opts_dict = {
        k:speaker_opts_dict[k]
        for k in speaker_opts_dict
        if speaker_opts_dict[k]
    }


    df = pl.DataFrame(speaker_opts_dict)
    df = (
        df
        .with_columns(
            file_name = pl.lit(path.stem),
            speaker_num = pl.col("tiernum").str.to_integer()
        )
    )
    return df

read_method = {
    ".yml": yaml2speaker,
    ".csv": csv2speaker,
    ".xlsx": excel2speaker,
    ".speaker": oldfave2speaker
}
class Speaker():
    """
    This is a class to represent speaker information.
    The argument to `Speaker()` can be one of

    - A .yaml file
    - A .csv file
    - A .xlsx file
    - An old fave .speaker file

    With the exception of the old .speaker files, to
    work well with new-fave, these speaker files should contain
    the following fields

    - `file_name`: The file stem of the wav and textgrid files
    - `speaker_num`: The speaker to be analyzed in a file. the first speaker is `1`.

    Attributes:
        df (pl.DataFrame): 
            A polars data frame of speaker information
    """
    def __init__(self, arg: dict|list|pl.DataFrame|Path = None):
        self.df = None
        if type(arg) is str:
            arg = Path(arg)
        if isinstance(arg, dict):
            self.df = pl.DataFrame(arg)
        if isinstance(arg, list) and not isinstance(arg[0], self.__class__):
            self.df = pl.DataFrame(arg)
        if isinstance(arg, list) and isinstance(arg[0], self.__class__):
            all_df = [s.df for s in arg]
            self.df = pl.concat(all_df, how = "diagonal")
        if isinstance(arg, pl.DataFrame):
            self.df = arg
        if isinstance(arg, Path):
            self.df = self.read_speaker(arg)
        
        if not "file_name" in self.df.columns:
            warnings.warn(
                ( 
                    "The provided speaker file does not contain "
                    "a file_name field. "
                    "Metadata won't be correctly merged."
                )
            )
        
        if not "speaker_num" in self.df.columns:
            warnings.warn(
                ( 
                    "The provided speaker file does not contain "
                    "a speaker_num field. "
                    "Metadata won't be correctly merged."
                )
            )        
        
        if not self.df is None and "file_name" in self.df.columns:
            self.df = (
                self.df
                .with_columns(
                    pl.col("file_name").map_elements(self._get_stem, return_dtype=pl.String)
                )
            )
       
    
    def __repr__(self) -> str:
        out = "Speaker data \n" + self.df.__repr__()
        return out
    
    def _get_stem(self, x):
        out = Path(x).stem
        return out


    def read_speaker(self, path:Path):
        extension = path.suffix
        reader = read_method[extension]
        return reader(path)
