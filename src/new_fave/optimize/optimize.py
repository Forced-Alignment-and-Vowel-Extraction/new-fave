from new_fave.measurements.vowel_measurement import VowelMeasurement, \
    VowelClass, \
    VowelClassCollection
from fasttrackpy.utils.safely import safely
import numpy as np
from tqdm import tqdm
from typing import Literal

def run_optimize(
        vowel_system: VowelClassCollection,
        optim_params: list[
             Literal[
                 "param_speaker_global",
                 "param_speaker_byvclass",
                 "bparam_speaker_global",
                 "bparam_speaker_byvclass",
                 "maxformant_speaker_global",
                 "param_corpus_byvowel"
                ]
            ] = [
                 "param_speaker_global",
                 "param_speaker_byvclass",
                 "bparam_speaker_global",
                 "bparam_speaker_byvclass",
                 "maxformant_speaker_global"
                ],
        max_iter = 10
    ):


    """
    Repeatedly run optimization until either `max_iter` is reached,
    or the difference between two iterations becomes small.

    Args:
        vowel_system (VowelClassCollection):
            The vowel space to be optimized
        optim_params (list[Literal["param_speaker_global", "param_speaker_byvclass", "bparam_speaker_global", "bparam_speaker_byvclass", "maxformant_speaker_global", "param_corpus_byvowel"]], optional): 
            The optimization parameters to use. Defaults to [ "param_speaker_global", "param_speaker_byvclass", "bparam_speaker_global", "bparam_speaker_byvclass", "maxformant_speaker_global" ].
        max_iter (int, optional):
            The maximum number of iterations to run.
            Defaults to 10.
    """
    current_formants = vowel_system.winner_expanded_formants
    msqe = [np.inf]
    for i in range(max_iter):
        optimize_vowel_measures(
            vowel_system.vowel_measurements,
            optim_params=optim_params
            )
        new_formants = vowel_system.winner_expanded_formants
        new_msqe = np.sqrt(((current_formants - new_formants)**2).mean())

        if np.isclose(new_msqe, 0.0):
            return

        if msqe[-1]/new_msqe <= 1.1:
            return
        
        current_formants = new_formants
        msqe.append(new_msqe)
    
    return



def optimize_vowel_measures(
        vowel_measurements: list[VowelMeasurement],
        optim_params: list[
             Literal[
                 "param_speaker_global",
                 "param_speaker_byvclass",
                 "bparam_speaker_global",
                 "bparam_speaker_byvclass",
                 "maxformant_speaker_global",
                 "param_corpus_byvowel"
                ]
            ] = [
                 "param_speaker_global",
                 "param_speaker_byvclass",
                 "bparam_speaker_global",
                 "bparam_speaker_byvclass",
                 "maxformant_speaker_global"
                ]
    ):
    """
    Optimize a list of VowelMeasurements.

    Args:
        vowel_measurements (list[VowelMeasurement]): 
            The list of vowel measurements to optimize
        optim_params (list[Literal["param_speaker_global", "param_speaker_byvclass", "bparam_speaker_global", "bparam_speaker_byvclass", "maxformant_speaker_global", "param_corpus_byvowel"]], optional): 
            The optimization parameters to use. Defaults to [ "param_speaker_global", "param_speaker_byvclass", "bparam_speaker_global", "bparam_speaker_byvclass", "maxformant_speaker_global" ].
    """

    new_winners = [
        optimize_one_measure(vm, optim_params=optim_params) 
        for vm in tqdm(vowel_measurements)
    ]

    for idx, new_idx in enumerate(new_winners):
        if new_idx is None:
            new_winners[idx] = vowel_measurements[idx].winner_index
    
    for vm, idx in zip(vowel_measurements, new_winners):
        vm.winner = idx

@safely(message="There was a problem optimizing a vowel.")
def optimize_one_measure(
        vowel_measurement: VowelMeasurement,
         optim_params: list[
             Literal[
                 "param_speaker_global",
                 "param_speaker_byvclass",
                 "bparam_speaker_global",
                 "bparam_speaker_byvclass",
                 "maxformant_speaker_global",
                 "param_corpus_byvowel"
                ]
             ] = [
                 "param_speaker_global",
                 "param_speaker_byvclass",
                 "bparam_speaker_global",
                 "bparam_speaker_byvclass",
                 "maxformant_speaker_global"
                ]
    )->int:
    """
    Optimize a single vowel measurement

    This function optimizes a given vowel measurement based on the 
    specified optimization parameters.

    Args:
        vowel_measurement (VowelMeasurement): 
            The VowelMeasurement to optimize
        optim_params (list[Literal["param_speaker_global", "param_speaker_byvclass", "bparam_speaker_global", "bparam_speaker_byvclass", "maxformant_speaker_global", "param_corpus_byvowel"]], optional): 
            The optimization parameters to use. Defaults to [ "param_speaker_global", "param_speaker_byvclass", "bparam_speaker_global", "bparam_speaker_byvclass", "maxformant_speaker_global" ].

    Returns:
        int: _description_
    """
    prob_dict = dict()

    if "param_speaker_global" in optim_params:
        prob_dict["param_speaker_global"] = vowel_measurement.cand_param_logprob_speaker_global

    if "param_speaker_byvclass" in optim_params:
        prob_dict["param_speaker_byvclass"] = vowel_measurement.cand_param_logprob_speaker_byvclass

    if "bparam_speaker_global" in optim_params:
        prob_dict["bparam_speaker_global"] = vowel_measurement.cand_bparam_logprob_speaker_global

    if "bparam_speaker_byvclass" in optim_params:
        prob_dict["bparam_speaker_byvclass"] = vowel_measurement.cand_bparam_logprob_speaker_byvclass

    if "param_corpus_byvclass" in optim_params:
        prob_dict["param_corpus_byvclass"] = vowel_measurement.cand_param_logprob_corpus_byvowel

    if "maxformant_speaker_global" in optim_params:
        prob_dict["maxformant_speaker_global"] = vowel_measurement.cand_maxformant_logprob_speaker_global
        
    joint_prob = vowel_measurement.cand_error_logprob_vm
    for dim in optim_params:
        joint_prob += prob_dict[dim]
    
    return joint_prob.argmax()