from new_fave.utils.local_resources import recodes, \
    parsers, \
    heuristics, \
    fasttrack_config

from fave_recode.labelset_parser import LabelSetParser
from fave_recode.rule_classes import RuleSet
from fave_measurement_point.heuristic import Heuristic

from pathlib import Path


def test_paths():
    recodes_paths = [
        Path(x)
        for x in recodes.values()
    ]

    for p in recodes_paths:
        assert p.is_file()

    for p in recodes_paths:
        try:
            RuleSet(rule_path=p)
            assert True
        except:
            assert False

    parsers_paths = [
        Path(x)
        for x in parsers.values()
    ]

    for p in parsers_paths:
        assert p.is_file()
    
    for p in parsers_paths:
        try:
            LabelSetParser(parser_path=p)
            assert True
        except:
            assert False

    heuristic_paths = [
        Path(p) 
        for p in heuristics.values()
    ]

    for p in heuristic_paths:
        assert p.is_file()

    for p in heuristic_paths:
        try:
            Heuristic(heuristic_path=p)
            assert True
        except:
            assert False


    for p in fasttrack_config.values():
        assert Path(p).is_file()
