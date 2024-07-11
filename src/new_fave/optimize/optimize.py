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
                 #"param_speaker_global",
                 "param_speaker",
                 #"bparam_speaker",
                 #"bparam_speaker_global",
                 #"bparam_speaker_byvclass",
                 #"maxformant_speaker_global",
                 "maxformant_speaker"
                ],
        max_iter = 5
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
        optimize_speaker(
            vowel_system,
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


def optimize_speaker(
        speaker: VowelClassCollection,
        optim_params = ["param_speaker", "maxformant_speaker"]
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
            pbar = pbar
        )
    
    pbar.close()


def optimize_vowel_measures(
        vowel_measurements: list[VowelMeasurement],
        optim_params: list[
              Literal["param_speaker", "maxformant_speaker"]
            ] = ["param_speaker", "maxformant_speaker"],
        pbar: tqdm = None
    ):
    """
    Optimize a list of VowelMeasurements.

    Args:
        vowel_measurements (list[VowelMeasurement]): 
            The list of vowel measurements to optimize
        optim_params (list[Literal["param_speaker_global", "param_speaker_byvclass", "bparam_speaker_global", "bparam_speaker_byvclass", "maxformant_speaker_global", "param_corpus_byvowel"]], optional): 
            The optimization parameters to use. Defaults to [ "param_speaker_global", "param_speaker_byvclass", "bparam_speaker_global", "bparam_speaker_byvclass", "maxformant_speaker_global" ].
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
            optimize_one_measure(vm, optim_params=optim_params)[vm.winner_index]
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
                 "maxformant_speaker_global",
                 "maxformant_speaker_byvclass",                 
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

    #if "maxformant_speaker_global" in optim_params:
    prob_dict["maxformant_speaker_global"] = vowel_measurement.cand_maxformant_logprob_speaker_global
    
    if "maxformant_speaker_byvclass" in optim_params:
        prob_dict["maxformant_speaker_byvclass"] = vowel_measurement.cand_maxformant_logprob_speaker_byvclass        
        
    joint_prob = vowel_measurement.cand_error_logprob_vm + \
        vowel_measurement.place_penalty #+ \
        #wvowel_measurement.cand_bandwidth_logprob[1,:] 
        
    
    for dim in optim_params:
        joint_prob += prob_dict[dim]
    
    return joint_prob