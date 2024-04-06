from new_fave.measurements.vowel_measurement import VowelMeasurement, \
    VowelClass, \
    VowelClassCollection
import numpy as np

def optimize_vowel_measures(
        vowel_measurements: list[VowelMeasurement]
    ):
    
    new_winners = [optimize_one_measure(vm) for vm in vowel_measurements]
    for vm, idx in zip(vowel_measurements, new_winners):
        vm.winner = idx


def optimize_one_measure(
        vowel_measurement: VowelMeasurement
    ):
    
    cand_mahal_log_prob = vowel_measurement.cand_mahal_log_prob
    max_formant_log_prob = vowel_measurement.max_formant_log_prob

    if np.any(~np.isfinite(cand_mahal_log_prob)):
        cand_mahal_log_prob = np.zeros(shape = cand_mahal_log_prob.shape[0])
        
    
    if np.any(~np.isfinite(max_formant_log_prob)):
        max_formant_log_prob = np.zeros(shape=max_formant_log_prob.shape[0])        
    
    joint_prob = cand_mahal_log_prob +\
                    vowel_measurement.error_log_prob +\
                    max_formant_log_prob
                    
    
    return joint_prob.argmax()