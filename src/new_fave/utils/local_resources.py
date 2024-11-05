from importlib.resources import files
from pathlib import Path
from typing import Callable, Any


fave_fasttrack = str(files("new_fave").joinpath("resources", "fasttrack_config.yml"))

fave_cmu2phila = str(files("new_fave").joinpath("resources", "cmu2phila.yml"))
fave_cmu2labov = str(files("new_fave").joinpath("resources", "cmu2labov.yml"))
fave_norecode = str(files("new_fave").joinpath("resources", "norecode.yml"))

cmu_parser = str(files("new_fave").joinpath("resources", "cmu_parser.yml"))

fave_measurement = str(files("new_fave").joinpath("resources", "fave_measurement.yml"))

default_vowel_place = str(files("new_fave").joinpath("resources", "vowel_place.yml"))

recodes = {
    "cmu2phila": fave_cmu2phila,
    "cmu2labov": fave_cmu2labov,
    "norecode": fave_norecode
}

parsers = {
    "cmu_parser": cmu_parser
}

heuristics = {
    "fave": fave_measurement
}

fasttrack_config = {
    "default": fave_fasttrack
}

vowel_place = {
    "default" : default_vowel_place
}

def local_resources():
    """
    Attributes:
        recodes (dict): 
            Recode options. Contains `"cmu2phila"` and `"cmu2labov"`
        parsers (dict):
            Labelset parsers. Contains `"cmu_parser"`
        heursitics (dict):
            Measurement point heuristics. Contains `"fave"`
        vowel_place (dict):
            Vowel place definitions
        fasttrack_config (dict):
            FastTrack config. Contains `"default"`
    """
    return

def generic_resolver(
        resolve_func: Callable, 
        to_resolve: str|Path = None,
        resource_dict: dict = None,
        default_value:Any =  None):
    """
    Resolve a passed string or path, and 
    return the desired object

    Args:
        resolve_func (Callable):
            The function to apply to the file.
        to_resolve (str | Path, optional): 
            The value to resolve. Either a name of 
            a built-in resource, or a path to a config file.
            Defaults to None.
        resource_dict (dict, optional): 
            A dictionary defining providing paths to 
            a built-in resource. 
            Defaults to None.
        default_value (Any, optional):
            A default value to return.
            Defaults to None.

    Returns:
        (Any): The value returned by resolve_func
    """
    if not to_resolve:
        return default_value

    if to_resolve in resource_dict:
        return resolve_func(resource_dict[to_resolve])
    
    if type(to_resolve) is str:
        to_resolve = Path(to_resolve)
    
    if to_resolve.exists() and to_resolve.is_file():
        return resolve_func(to_resolve)
    
    raise ValueError(
        (
            f"The provided value {to_resolve} "
            "is not a built-in resource "
            "and does not appear to be a file."
        )
    )


