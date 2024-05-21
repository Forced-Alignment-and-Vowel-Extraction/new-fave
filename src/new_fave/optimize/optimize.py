from new_fave.measurements.vowel_measurement import VowelMeasurement, \
    VowelClass, \
    VowelClassCollection
from fasttrackpy.utils.safely import safely
import numpy as np
from tqdm import tqdm
from typing import Literal

def run_optimize(
        vowel_system: VowelClassCollection,
        optim_params = ["cand_mahal", "max_formant"],
        max_iter = 10
    ):
    """
    Repeatedly run optimization until either `max_iter` is reached,
    or the difference between two iterations becomes small.

    Args:
        vowel_system (VowelClassCollection):
            The vowel space to be optimized
        optim_params (list, optional): 
            The parameters to use for optimization. 
            Defaults to ["cand_mahal", "max_formant"].
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
        optim_params: list[Literal["cand_mahal", "max_formant"]] = ["cand_mahal", "max_formant"]
    ):
    """
    Optimize a list of VowelMeasurements.

    Args:
        vowel_measurements (list[VowelMeasurement]): 
            The list of vowel measurements to optimize
        optim_params (list[Literal["cand_mahal", "max_formant"]], optional): 
            The optimization parameters to use. Defaults to ["cand_mahal", "max_formant"].
    """
    
    new_winners = [
        optimize_one_measure(vm, optim_params=optim_params) 
        for vm in tqdm(vowel_measurements)
    ]

    for idx, new_idx in new_winners:
        if new_idx is None:
            new_winners[idx] = vowel_measurements[idx].winner_index
    
    for vm, idx in zip(vowel_measurements, new_winners):
        vm.winner = idx

@safely(message="There was a problem optimizing a vowel.")
def optimize_one_measure(
        vowel_measurement: VowelMeasurement,
         optim_params: list[Literal["cand_mahal", "max_formant"]] = ["cand_mahal", "max_formant"]
    )->int:
    """
    This function optimizes a given vowel measurement based on the 
    specified optimization parameters. The optimization parameters 
    can include 'cand_mahal' and 'max_formant'.

    Args:
        vowel_measurement (VowelMeasurement): 
            The VowelMeasurement to optimize
        optim_params (list[Literal["cand_mahal", "max_formant"]], optional): 
            The optimization parameters to use. Defaults to ["cand_mahal", "max_formant"].

    Returns:
        (int): The index of the winning candidate.
    """
    prob_dict = dict()

    if "cand_mahal" in optim_params:
        prob_dict["cand_mahal"] = vowel_measurement.cand_mahal_log_prob

    if "max_formant" in optim_params:
        prob_dict["max_formant"] = vowel_measurement.max_formant_log_prob
        
    joint_prob = vowel_measurement.error_log_prob 
    for dim in optim_params:
        joint_prob += prob_dict[dim]
    
    return joint_prob.argmax()