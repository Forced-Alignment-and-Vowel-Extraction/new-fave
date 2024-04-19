from new_fave.measurements.vowel_measurement import VowelMeasurement, \
    VowelClass, \
    VowelClassCollection
import numpy as np

def optimize_vowel_measures(
        vowel_measurements: list[VowelMeasurement],
        optim_params = ["cand_mahal", "rate", "kde"]
    ):
    #new_winners = Parallel(n_jobs=5)(optimize_one_measure(vm) for vm in vowel_measurements)
    new_winners = [optimize_one_measure(vm, optim_params=optim_params) for vm in vowel_measurements]
    for vm, idx in zip(vowel_measurements, new_winners):
        vm.winner = idx

def optimize_one_measure(
        vowel_measurement: VowelMeasurement,
         optim_params = ["cand_mahal", "max_formant", "rate", "kde"]
    ):
   
    prob_dict = {
        "cand_mahal": vowel_measurement.cand_mahal_log_prob,
        "max_formant": vowel_measurement.max_formant_log_prob,
        "rate": vowel_measurement.rate_log_prob,
        "kde": vowel_measurement.cand_log_kde


    }
    if np.any(~np.isfinite(prob_dict["cand_mahal"])):
        prob_dict["cand_mahal"] = np.zeros(shape = prob_dict["cand_mahal"].shape[0])
        
    
    if np.any(~np.isfinite(prob_dict["max_formant"])):
        prob_dict["max_formant"] = np.zeros(shape=prob_dict["max_formant"].shape[0])        
    
    joint_prob = vowel_measurement.error_log_prob 
    for dim in optim_params:
        joint_prob += prob_dict[dim]
    
    return joint_prob.argmax()