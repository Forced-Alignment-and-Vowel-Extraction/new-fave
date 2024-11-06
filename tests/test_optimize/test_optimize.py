from new_fave.optimize.optimize import optimize_one_measure,\
    optimize_vowel_measures, \
    run_optimize
import numpy as np
import pytest

optim_params = ["param_speaker",
                 "fratio_speaker",
                 "centroid_speaker",
                 "maxformant_speaker"
                ]

class MockVowelMeasure:
    def __init__(self, len, winner_idx):
        log_prob = -np.ones(len)*2
        log_prob[winner_idx] = -0.5

        self.cand_param_logprob_speaker_global = log_prob
        self.cand_param_logprob_speaker_byvclass = log_prob
        self.cand_maxformant_logprob_speaker_global = log_prob 
        self.cand_bparam_logprob_speaker_global = log_prob 
        self.cand_bparam_logprob_speaker_byvclass = log_prob         
        self.cand_error_logprob_vm  = log_prob
        self.reference_logprob  = log_prob
        self.winner = None
        self.place_penalty = np.zeros(len)
        self.count = len

    def __len__(self):
        return self.count

class MockVowelClassCollection:
    def __init__(self, vowel_measurements):
        self.winner_expanded_formants = np.ones((len(vowel_measurements), 100))
        self.vowel_measurements = vowel_measurements

@pytest.mark.skip(reason="Needs rewriting")
def test_optimize_one_measure():
    target_idx = 4
    vm = MockVowelMeasure(10, winner_idx=target_idx)
    result_idx = np.argmax(optimize_one_measure(vm, optim_params))
    assert target_idx == result_idx

@pytest.mark.skip(reason="Needs rewriting")
def test_optimize_measurements():
    target_idces = [3, 4, 5]
    vms = [
        MockVowelMeasure(10, i)
        for i in target_idces
    ]
    optimize_vowel_measures(vms, optim_params=optim_params)
    result_idces = [x.winner for x in vms]
    for t, r in zip(target_idces, result_idces):
        assert t == r

@pytest.mark.skip(reason="Needs rewriting")
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
