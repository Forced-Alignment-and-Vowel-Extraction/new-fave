from new_fave.optimize.optimize import optimize_one_measure,\
    optimize_vowel_measures, \
    run_optimize
import numpy as np

class MockVowelMeasure:
    def __init__(self, len, winner_idx):
        log_prob = -np.ones(len)*2
        log_prob[winner_idx] = -0.5

        self.cand_param_logprob_speaker_global = log_prob
        self.cand_param_logprob_speaker_byvclass = log_prob
        self.cand_maxformant_logprob_speaker_global = log_prob
        self.cand_error_logprob_vm  = log_prob
        self.winner = None

class MockVowelClassCollection:
    def __init__(self, vowel_measurements):
        self.winner_expanded_formants = np.ones((len(vowel_measurements), 100))
        self.vowel_measurements = vowel_measurements

def test_optimize_one_measure():
    target_idx = 4
    vm = MockVowelMeasure(10, winner_idx=target_idx)
    result_idx = optimize_one_measure(vm)
    assert target_idx == result_idx


def test_optimize_measurements():
    target_idces = [3, 4, 5]
    vms = [
        MockVowelMeasure(10, i)
        for i in target_idces
    ]
    optimize_vowel_measures(vms)
    result_idces = [x.winner for x in vms]
    for t, r in zip(target_idces, result_idces):
        assert t == r


def test_run_optimize():
    target_idces = [3, 4, 5]
    vms = [
        MockVowelMeasure(10, i)
        for i in target_idces
    ]
    vcc = MockVowelClassCollection(vms)
    run_optimize(vcc)
    result_idces = [x.winner for x in vms]
    for t, r in zip(target_idces, result_idces):
        assert t == r
