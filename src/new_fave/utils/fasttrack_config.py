import yaml
from pathlib import Path
from fasttrackpy import Smoother
from new_fave.utils.local_resources import fave_fasttrack

def read_fasttrack(config:str|Path)->dict:
    with Path(fave_fasttrack).open('r') as c:
        ft_dict = yaml.safe_load(c)

    config_dict = dict()
    if config:
        if type(config) is str:
            config = Path(config)
        
        with config.open('r') as c:
            config_dict = yaml.safe_load(c)

    for key in config_dict:
        if key in ft_dict:
            ft_dict[key] = config_dict[key]

    smoother = Smoother()
    if "smoother" in ft_dict:
        smoother = Smoother(**ft_dict["smoother"])
    
    ft_dict["smoother"] = smoother

    return ft_dict
