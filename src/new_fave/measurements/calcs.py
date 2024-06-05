import numpy as np
import nptyping as npt
from nptyping import NDArray, Shape, Float
from typing import Any

def mahalanobis(
        params:NDArray[Shape['Dim, Cand'], Float], 
        param_means:NDArray[Shape['Dim, 1'], Float], 
        inv_cov:NDArray[Shape['Dim, Dim'], Float]
    )->NDArray[Shape["Cand"], Float]:
    """
    Calculates the Mahalanobis distance.

    Args:
        params (NDArray[Shape['Dim, Cand'], Float]): 
            The parameters for which the Mahalanobis distance is to be calculated.
        param_means (NDArray[Shape['Dim, 1'], Float]): 
            The mean of the distribution.
        inv_cov (NDArray[Shape['Dim, Dim'], Float]): 
            The inverse of the covariance matrix of the distribution.

    Returns:
        (NDArray[Shape["Cand"], Float]): 
            The Mahalanobis distance of each parameter from the distribution.
    """    
    
    x_mu = params - param_means
    left = np.dot(x_mu.T, inv_cov)
    mahal = np.dot(left, x_mu)
    return mahal.diagonal()

def mahal_log_prob(mahals, df):

    pass