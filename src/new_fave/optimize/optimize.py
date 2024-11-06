from new_fave.measurements.vowel_measurement import (VowelMeasurement,
    VowelClass, 
    VowelClassCollection
)
from fasttrackpy.utils.safely import safely
import numpy as np
from tqdm import tqdm
from typing import Literal

def run_optimize(
        vowel_system: VowelClassCollection,
        optim_params: list = [
                 "param_speaker",
                 "fratio_speaker",
                 "centroid_speaker",
                 "maxformant_speaker"
                ],
        f1_cutoff: float|np.float64 = np.inf,
        f2_cutoff: float|np.float64 = np.inf,        
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
        f1_cutoff (float | np.float64):
            The maximum considerable F1 value
        f2_cutoff (float | np.float64):
            The maximum considerable F2 value
        max_iter (int, optional):
            The maximum number of iterations to run.
            Defaults to 10.
    """
    current_formants = vowel_system.winner_expanded_formants
    msqe = [np.inf]
    for i in range(max_iter):
        optimize_speaker(
            vowel_system,
            optim_params=optim_params,
            f1_cutoff = f1_cutoff,
            f2_cutoff = f2_cutoff
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


def optimize_speaker(
        speaker: VowelClassCollection,
        optim_params: list[str],
        f1_cutoff: float|np.float64 = np.inf,
        f2_cutoff: float|np.float64 = np.inf
):
    keys = speaker.sorted_keys
    total_len = 0
    for k in speaker:
        total_len += len(speaker[k])

    pbar = tqdm(total = total_len)

    for k in keys:
        pbar = optimize_vowel_measures(
            vowel_measurements=speaker[k],
            optim_params = optim_params,
            f1_cutoff = f1_cutoff,
            f2_cutoff = f2_cutoff,            
            pbar = pbar
        )
    
    pbar.close()


def optimize_vowel_measures(
        vowel_measurements: list[VowelMeasurement],
        optim_params: list[str],
        f1_cutoff: float|np.float64 = np.inf,
        f2_cutoff: float|np.float64 = np.inf,       
        pbar: tqdm = None
    ):
    """
    Optimize a list of VowelMeasurements.

    Args:
        vowel_measurements (list[VowelMeasurement]): 
            The list of vowel measurements to optimize
        optim_params (list[Literal["param_speaker_global", "param_speaker_byvclass", "bparam_speaker_global", "bparam_speaker_byvclass", "maxformant_speaker_global", "param_corpus_byvowel"]], optional): 
            The optimization parameters to use. Defaults to [ "param_speaker_global", "param_speaker_byvclass", "bparam_speaker_global", "bparam_speaker_byvclass", "maxformant_speaker_global" ].
        f1_cutoff (float | np.float64):
            The maximum considerable F1 value
        f2_cutoff (float | np.float64):
            The maximum considerable F2 value            
        pbar (tqdm):
            A progress bar.
    """

    scope = "_byvclass"
    if len(vowel_measurements) <= 10:
        scope = "_global"
    
    optim_params = [x+scope for x in optim_params]

    optimized = []
    to_optimize = [vm for vm in vowel_measurements]
    #chunk_size = int(len(to_optimize) * 0.05)
    #if chunk_size < 10:
    chunk_size = 10
    if not pbar:
        pbar = tqdm(total=len(to_optimize))
    while len(to_optimize) > 0:
        to_optim = np.array([
            optimize_one_measure(
                vm, 
                optim_params=optim_params, 
                f1_cutoff=f1_cutoff, 
                f2_cutoff=f2_cutoff
                )[vm.winner_index]
            for vm in to_optimize
        ])

        order = np.argsort(to_optim)
        to_optimize = [
            to_optimize[idx] 
            for idx in order
        ]

        if (len(to_optimize)-1) <= chunk_size :
            chunk = to_optimize[0:]
        else:
            chunk = to_optimize[0:(chunk_size)]

        chunk_winners = []
        for vm in chunk:
            chunk_winners.append(
                np.argmax(optimize_one_measure(vm, optim_params=optim_params))
            )
            pbar.update()
        
        for w, v in zip(chunk_winners, chunk):
            v.winner = w

        for v in chunk:
            to_optimize.pop(to_optimize.index(v))

    return pbar

#@safely(message="There was a problem optimizing a vowel.")
def optimize_one_measure(
        vowel_measurement: VowelMeasurement,
         optim_params: list,
        f1_cutoff: float|np.float64 = np.inf,
        f2_cutoff: float|np.float64 = np.inf
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
        f1_cutoff (float | np.float64):
            The maximum considerable F1 value
        f2_cutoff (float | np.float64):
            The maximum considerable F2 value            

    Returns:
        int: _description_
    """
    prob_dict = dict()

    if "param_speaker_global" in optim_params:
        prob_dict["squareparam_speaker_global"] = vowel_measurement.cand_squareparam_logprob_speaker_global

    if "param_speaker_byvclass" in optim_params:
        prob_dict["squareparam_speaker_byvclass"] = vowel_measurement.cand_squareparam_logprob_speaker_byvclass

    if "bparam_speaker_global" in optim_params:
        prob_dict["bparam_speaker_global"] = vowel_measurement.cand_bparam_logprob_speaker_global

    if "bparam_speaker_byvclass" in optim_params:
        prob_dict["bparam_speaker_byvclass"] = vowel_measurement.cand_bparam_logprob_speaker_byvclass

    if "param_corpus_byvclass" in optim_params:
        prob_dict["param_corpus_byvclass"] = vowel_measurement.cand_param_logprob_corpus_byvowel

    if "maxformant_speaker_global" in optim_params:
        prob_dict["maxformant_speaker_global"] = vowel_measurement.cand_maxformant_logprob_speaker_global
    
    if "maxformant_speaker_byvclass" in optim_params:
        prob_dict["maxformant_speaker_byvclass"] = vowel_measurement.cand_maxformant_logprob_speaker_byvclass
        
    if "fratio_speaker_byvclass" in optim_params:
        prob_dict["fratio_speaker_byvclass"] = vowel_measurement.cand_fratio_logprob_speaker_byvclass

    if "fratio_speaker_global" in optim_params:
        prob_dict["fratio_speaker_global"] = vowel_measurement.cand_fratio_logprob_speaker_global

    if "centroid_speaker_byvclass" in optim_params:
        prob_dict["centroid_speaker_byvclass"] = vowel_measurement.cand_centroid_logprob_speaker_byvclass        

    if "centroid_speaker_global" in optim_params:
        prob_dict["centroid_speaker_global"] = vowel_measurement.cand_centroid_logprob_speaker_global

    
    cutoff = np.zeros(len(vowel_measurement))
    f1_cutoff_prob = np.zeros(len(vowel_measurement))
    f2_cutoff_prob = np.zeros(len(vowel_measurement))

    # cutoff = vowel_measurement.cand_centroid_logprob_speaker_global
    # cutoff[cutoff < -10] = -np.inf
    # cutoff[cutoff > -np.inf] = 0

    # f1_max = np.log(1500)/np.sqrt(2)
    # f2_max = np.log(3500)/np.sqrt(2)

    f1_max = np.log(f1_cutoff)/np.sqrt(2)
    f2_max = np.log(f2_cutoff)/np.sqrt(2)

    f1_cutoff_prob = vowel_measurement.cand_param[0,0,:]
    f1_cutoff_prob[f1_cutoff_prob > f1_max] = -np.inf
    f1_cutoff_prob[f1_cutoff_prob > -np.inf] = 0

    f2_cutoff_prob = vowel_measurement.cand_param[0,1,:]
    f2_cutoff_prob[f2_cutoff_prob > f2_max] = -np.inf
    f2_cutoff_prob[f2_cutoff_prob > -np.inf] = 0

    joint_prob = vowel_measurement.cand_error_logprob_vm + \
        cutoff+\
        f1_cutoff_prob + \
        f2_cutoff_prob +\
        vowel_measurement.reference_logprob + \
        vowel_measurement.place_penalty * 5
       
        
    
    for dim in prob_dict:
        joint_prob += prob_dict[dim]
    
    return joint_prob

