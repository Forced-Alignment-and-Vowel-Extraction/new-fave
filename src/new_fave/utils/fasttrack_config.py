import yaml
from pathlib import Path
from fasttrackpy import Smoother


def read_fasttrack(config = str|Path)->dict:
    if type(config) is str:
        config = Path(config)
    
    with config.open('r') as c:
        config_dict = yaml.safe_load(c)
    
    smoother = Smoother()
    if "smoother" in config_dict:
        smoother = Smoother(**config_dict["smoother"])
    
    config_dict["smoother"] = smoother

    return config_dict
