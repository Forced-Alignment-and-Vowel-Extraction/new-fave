from new_fave.measurements.vowel_measurement import VowelMeasurement, \
    VowelClass, \
    VowelClassCollection
import numpy as np
from tqdm import tqdm

def optimize_vowel_measures(
        vowel_measurements: list[VowelMeasurement],
        optim_params = ["cand_mahal", "rate", "max_formant", "kde"]
    ):
    #new_winners = Parallel(n_jobs=5)(optimize_one_measure(vm) for vm in vowel_measurements)
    new_winners = [optimize_one_measure(vm, optim_params=optim_params) for vm in tqdm(vowel_measurements)]
    for vm, idx in zip(vowel_measurements, new_winners):
        vm.winner = idx

def optimize_one_measure(
        vowel_measurement: VowelMeasurement,
         optim_params = ["cand_mahal", "max_formant", "rate", "kde"]
    ):
   
    prob_dict = dict()

    if "cand_mahal" in optim_params:
        prob_dict["cand_mahal"] = vowel_measurement.cand_mahal_log_prob
        if np.any(~np.isfinite(prob_dict["cand_mahal"])):
            prob_dict["cand_mahal"] = np.zeros(shape = prob_dict["cand_mahal"].shape[0])

    if "max_formant" in optim_params:
        prob_dict["max_formant"] = vowel_measurement.max_formant_log_prob
        if np.any(~np.isfinite(prob_dict["max_formant"])):
            prob_dict["max_formant"] = np.zeros(shape=prob_dict["max_formant"].shape[0])

    if "kde" in optim_params:
        prob_dict["kde"] =  vowel_measurement.cand_log_kde
    
    if "rate" in optim_params:
        prob_dict["rate"] = vowel_measurement.rate_log_prob
        
    joint_prob = vowel_measurement.error_log_prob 
    for dim in optim_params:
        joint_prob += prob_dict[dim]
    
    return joint_prob.argmax()