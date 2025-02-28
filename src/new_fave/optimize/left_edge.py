import numpy as np
from new_fave.measurements.vowel_measurement import VowelMeasurement

def beyond_edge(
        vowel_measurement: VowelMeasurement
    ) -> np.ndarray:
    """
    For a given vowel measurement, return an 
    array of log probabilities indicating whether
    or not a candidate is beyond the desired
    edge of the front of the vowel space.

    Args:
        vowel_measurement (VowelMeasurement):
            A vowel measurement to optimize
        slope (float, optional): 
            The desired slope for the maximum edge
            of front vowel space. Defaults to -1.5.

    Returns:
        np.ndarray: 
            log probabilities of 0 for candidates
            below the threshold, and negative 
            infinity for candidates above it.
    """
    
    slopes = np.linspace(-1.5, -0.75, num = 10)
    penalty = -0.3

    edge_logprob = np.zeros(len(vowel_measurement))
    vowel_system = vowel_measurement.vowel_class.vowel_system
    
    intercepts = vowel_system.edge_intercept(slopes)
    xes = vowel_measurement.cand_centroid[0,1,:]
    ys = vowel_measurement.cand_centroid[0,0,:]
    
    for i, s in zip(intercepts, slopes):
        y_max = i + (s * xes)
        edge_logprob[ys > y_max] += penalty

    y_max = intercepts[-1] + (slopes[-1] * xes)

    edge_logprob[ys > y_max] = -np.inf
    return edge_logprob
