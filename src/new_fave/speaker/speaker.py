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
        if isinstance(arg, dict) or isinstance(arg, list):
            self.df = pl.DataFrame(arg)
            return
        if isinstance(arg, pl.DataFrame):
            self.df = arg
            return
        if isinstance(arg, Path):
            self.df = self.read_speaker(arg)
            return
        
        self.df = None
    
    def read_speaker(self, path:Path):
        extension = path.suffix
        reader = read_method[extension]
        return reader(path)
    
