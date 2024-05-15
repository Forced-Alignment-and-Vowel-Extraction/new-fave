from new_fave.utils.local_resources import recodes, \
    parsers, \
    heuristics, \
    fasttrack_config,\
    generic_resolver

from fave_recode.labelset_parser import LabelSetParser
from fave_recode.rule_classes import RuleSet
from fave_measurement_point.heuristic import Heuristic
from fave_recode.fave_recode import get_rules, get_parser
from new_fave.utils.fasttrack_config import read_fasttrack

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


def test_generic_resolver():
    ft_config = generic_resolver(
        resolve_func=read_fasttrack,
        to_resolve="default",
        resource_dict=fasttrack_config,
        default_value=dict()
    )

    assert type(ft_config) is dict

    rules = generic_resolver(
        resolve_func=get_rules,
        to_resolve="cmu2labov",
        resource_dict=recodes
    )

    assert isinstance(rules, RuleSet)

    parser = generic_resolver(
        resolve_func = get_parser,
        to_resolve="cmu_parser",
        resource_dict=parsers
    )

    assert isinstance(parser, LabelSetParser)

    heuristic = generic_resolver(
        resolve_func=lambda x: Heuristic(heuristic_path=x),
        to_resolve="fave",
        resource_dict=heuristics
    )

    assert isinstance(heuristic, Heuristic)