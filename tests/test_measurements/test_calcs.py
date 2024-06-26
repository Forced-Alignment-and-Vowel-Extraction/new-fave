from new_fave.measurements.calcs import mahalanobis, \
    mahal_log_prob,\
    param_to_cov,\
    cov_to_icov, \
    clear_cached_properties

import numpy as np
from functools import cached_property

NParam = 15
NToken = 100

PARAM = np.random.random(NParam * NToken).reshape(NParam, NToken)


def test_param_to_cov():
    cov_mat = param_to_cov(PARAM)
    assert cov_mat.shape == (NParam, NParam)

def test_cov_to_icov():
    cov_mat = param_to_cov(PARAM)
    icov_mat = cov_to_icov(cov_mat)
    assert icov_mat.shape == (NParam, NParam)

def test_mahalanobis():
    mean = PARAM.mean(axis=1)
    mean = mean[:, np.newaxis]
    cov_mat = param_to_cov(PARAM)
    icov_mat = cov_to_icov(cov_mat)
    distances = mahalanobis(PARAM, mean, icov_mat)
    assert distances.shape == (NToken,)
    
def test_mahalanobis_logprob():
    mean = PARAM.mean(axis=1)
    mean = mean[:, np.newaxis]
    cov_mat = param_to_cov(PARAM)
    icov_mat = cov_to_icov(cov_mat)
    distances = mahalanobis(PARAM, mean, icov_mat)
    log_prob = mahal_log_prob(distances, PARAM)
    assert log_prob.shape == (NToken,)
    assert np.all(log_prob < 0)

def test_clear_cache():
    class MyClass:
        @cached_property
        def name(self):
            return "Cache Test"
        
    foo = MyClass()

    assert not "name" in foo.__dict__

    emitted = foo.name
    
    assert "name" in foo.__dict__

    clear_cached_properties(foo)

    assert not "name" in foo.__dict__
