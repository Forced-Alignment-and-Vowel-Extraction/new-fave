import numpy as np
import nptyping as npt
from nptyping import NDArray, Shape, Float
from typing import Any
import scipy.stats as stats
import warnings
import functools

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from new_fave.measurements.vowel_measurement import VowelMeasurement

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
    mahal = (
        np.dot(
            np.dot(
                x_mu.T, 
                inv_cov
            ), 
            x_mu)
        .diagonal()
        .copy()
    )
    return mahal

def mahal_log_prob(
        mahals: NDArray[Shape["Cand"], Float], 
        params: NDArray[Shape["*, *, ..."], Float]
    ) -> NDArray[Shape["Cand"], Float]:
    """
    
    Args:
        mahals (NDArray[Shape["Cand"], Float]): 
            The Mahalanobis distances.
        params (NDArray[Shape["*, *, ..."], Float]): 
            The parameters across which the mahalanobis
            distance was calculated

    Returns:
        (NDArray[Shape["Cand"], Float]): 
            The log probability
    """
    df = np.prod(params.shape[0:-1])
    log_prob = stats.chi2.logsf(
            mahals,
            df = df
        )
    if np.isfinite(log_prob).mean() < 0.5:
        log_prob = np.zeros(shape = log_prob.shape)    
    return log_prob


def param_to_cov(
    params:NDArray[Shape["*, *, ..."], Float]
) -> NDArray[Shape["X, X"], Float]:
    """
    Calculates the covariance matrix of the given parameters.

    Args:
        params (NDArray[Shape["*, *, ..."], Float]): 
            The parameters for which the covariance matrix is to be calculated.

    Returns:
        (NDArray[Shape["X, X"], Float]): 
            The covariance matrix of the parameters.
    """    
    N = params.shape[-1]
    square_params = params.reshape(-1, N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        param_cov = np.cov(square_params)

    param_cov = param_cov.reshape((square_params.shape[0],square_params.shape[0]))
    
    return param_cov

def cov_to_icov(
    cov_mat: NDArray[Shape["X, X"], Float]
) -> NDArray[Shape["X, X"], Float]:
    """
    Calculates the inverse covariance matrix of the given covariance matrix.

    Args:
        cov_mat (NDArray[Shape["X, X"], Float]): 
            The covariance matrix for which the inverse is to be calculated.

    Returns:
        (NDArray[Shape["X, X"], Float]): 
            The inverse covariance matrix of the given covariance matrix.
    """    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            params_icov = np.linalg.inv(cov_mat)
        except:
            params_icov = np.array([
                [np.nan] * cov_mat.size
            ]).reshape(
                cov_mat.shape[0],
                cov_mat.shape[1]
            )
    
    return params_icov

def clear_cached_properties(obj:object) -> None:
    """Clear the cache of any property in an object

    Args:
        obj (object): Any object.
    """
    clses = obj.__class__.mro()
    to_clear = []

    to_clear += [
        k 
        for cls in clses
        for k, v in vars(cls).items()
        if isinstance(v, functools.cached_property)
    ]
    for var in to_clear:
        if var in obj.__dict__:
            del obj.__dict__[var]