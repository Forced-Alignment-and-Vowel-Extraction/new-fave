from fasttrackpy import Smoother
from new_fave.utils.local_resources import fasttrack_config
from new_fave.utils.fasttrack_config import read_fasttrack

def test_read_fastrack():
    config = read_fasttrack(fasttrack_config["default"])
    assert isinstance(config["smoother"], Smoother)