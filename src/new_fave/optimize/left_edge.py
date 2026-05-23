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

    vowel_system = vowel_measurement.vowel_class.vowel_system
    intercepts = vowel_system.edge_intercept(slopes)
    xes = vowel_measurement.cand_centroid[0,1,:]
    ys = vowel_measurement.cand_centroid[0,0,:]

    y_max = intercepts[:, None] + slopes[:, None] * xes[None, :]
    over = ys[None, :] > y_max
    edge_logprob = penalty * over.sum(axis=0)
    edge_logprob[over[-1]] = -np.inf
    return edge_logprob
