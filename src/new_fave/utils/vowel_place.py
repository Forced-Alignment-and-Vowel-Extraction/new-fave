import yaml
from pathlib import Path
import re

def read_vowel_place(
        config: str|Path
) -> dict:
    """Read in a vowel place config

    The yaml config file should be formatted like

    ```yaml
    front:
        - REGEX
    back:
        - REGEX
    ```

    The list of regex for front and back will
    be concatenated with `|`.

    Args:
        config (str | Path): 
            path to a yaml file

    Returns:
        (dict): 
            a dictionary of place: regex
    """
    config = Path(config)
    with config.open('r') as c:
        config_dict = yaml.safe_load(c) 
    
    if len(config_dict) < 1:
        return(dict())

    re_dict = {}
    for k in config_dict:
        re_dict[k] = re.compile(
            "|".join(config_dict[k])
        )
    
    return re_dict