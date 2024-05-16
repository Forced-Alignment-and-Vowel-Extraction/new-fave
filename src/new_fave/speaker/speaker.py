import polars as pl
from pathlib import Path
import yaml
import csv


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

read_method = {
    ".yml": yaml2speaker,
    ".csv": csv2speaker,
    ".xlsx": excel2speaker
}
class Speaker():
    def __init__(self, arg: dict|list|pl.DataFrame|Path = None):
        self.df = None
        if type(arg) is str:
            arg = Path(arg)
        if isinstance(arg, dict) or isinstance(arg, list):
            self.df = pl.DataFrame(arg)
        if isinstance(arg, pl.DataFrame):
            self.df = arg
        if isinstance(arg, Path):
            self.df = self.read_speaker(arg)
        
        if not self.df is None:
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
    
