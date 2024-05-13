from new_fave.measurements.vowel_measurement import VowelMeasurement, \
    VowelClass, \
    VowelClassCollection
import numpy as np
from tqdm import tqdm

def run_optimize(
        vowel_system: VowelClassCollection,
        optim_params = ["cand_mahal", "max_formant"],
        max_iter = 10
    ):
    current_formants = vowel_system.winner_expanded_formants
    msqe = [np.inf]
    for i in range(max_iter):
        optimize_vowel_measures(
            vowel_system.vowel_measurements,
            optim_params=optim_params
            )
        new_formants = vowel_system.winner_expanded_formants
        new_msqe = np.sqrt(((current_formants - new_formants)**2).mean())

        if msqe[-1]/new_msqe <= 1.1:
            return
        
        current_formants = new_formants
        msqe.append(new_msqe)
    
    return



def optimize_vowel_measures(
        vowel_measurements: list[VowelMeasurement],
        optim_params = ["cand_mahal", "max_formant"]
    ):
    new_winners = [optimize_one_measure(vm, optim_params=optim_params) for vm in tqdm(vowel_measurements)]
    for vm, idx in zip(vowel_measurements, new_winners):
        vm.winner = idx

def optimize_one_measure(
        vowel_measurement: VowelMeasurement,
         optim_params = ["cand_mahal", "max_formant"]
    ):
   
    prob_dict = dict()

    if "cand_mahal" in optim_params:
        prob_dict["cand_mahal"] = vowel_measurement.cand_mahal_log_prob

    if "max_formant" in optim_params:
        prob_dict["max_formant"] = vowel_measurement.max_formant_log_prob
        
    joint_prob = vowel_measurement.error_log_prob 
    for dim in optim_params:
        joint_prob += prob_dict[dim]
    
    return joint_prob.argmax()