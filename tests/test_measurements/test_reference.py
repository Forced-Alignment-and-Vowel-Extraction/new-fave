from pathlib import Path
import polars as pl

from new_fave.measurements.reference import ReferenceValues

REFERENCE_PATH = Path("tests", "test_data", "fave_results")

def test_logparam_reference():
    try:
        reference = ReferenceValues(logparam_corpus=REFERENCE_PATH)
    except:
        assert False, "Problem with processing logparam references"

    assert reference.reference_type == "logparam"
    for k,v in reference.icov_dict.items():
        assert v.shape == (15, 15), f"Wrong shape for {k}."

def test_param_reference():
    try:
        reference = ReferenceValues(param_corpus=REFERENCE_PATH)
    except:
        assert False, "Problem with processing param references"

    assert reference.reference_type == "param"
    for k,v in reference.icov_dict.items():
        assert v.shape == (15, 15), f"Wrong shape for {k}."    

def test_point_reference():
    try:
        reference = ReferenceValues(points_corpus=REFERENCE_PATH)
    except:
        assert False, "Problem with processing point references"

    assert reference.reference_type == "points"
    for k,v in reference.icov_dict.items():
        assert v.shape == (6, 6), f"Wrong shape for {k}."           