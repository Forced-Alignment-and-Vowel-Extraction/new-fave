from functools import cached_property, update_wrapper
import numpy as np
from new_fave.measurements.calcs import mahalanobis, \
    mahal_log_prob,\
    param_to_cov,\
    cov_to_icov,\
    clear_cached_properties

from typing import Any


class OptimWrapper:
    def __init__(self, func):
        update_wrapper(self, func)
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class MahalWrap(OptimWrapper):
    prop = property
    def __init__(self, func):
        super().__init__(func)

class MahalCacheWrap(OptimWrapper):
    prop = cached_property
    def __init__(self, func):
        super().__init__(func)


class FlatWrap(OptimWrapper):
    prop = property
    def __init__(self, func):
        super().__init__(func)

class FlatCacheWrap(OptimWrapper):
    prop = cached_property
    def __init__(self, func):
        super().__init__(func)        
    

class PropertyFactory():
    def __init__(self, attr, *args):
        self.attr = attr
    
    @property
    def winner_factory(self):
        def parameterized_func(obj):
            return np.array([
                getattr(x, self.attr)[...,x.winner_index]
                for x in obj.vowel_measurements
            ]).T
        
        return parameterized_func
    
    @property
    def mean_factory(self):

        def parameterized_func(obj):
            winner_array  = getattr(obj, self.attr)
            N = winner_array.shape[-1]
            square_array = winner_array.reshape(-1, N)
            winner_mean =  square_array.mean(axis = 1)
            winner_mean = winner_mean[:, np.newaxis]
            return winner_mean
        
        return parameterized_func
    
    @property
    def icov_factory(self):

        def parameterized_func(obj):
            cov_mat = param_to_cov(getattr(obj, self.attr))
            icov_mat = cov_to_icov(cov_mat)
            return icov_mat

        return parameterized_func

def get_wrapped(cls, wrapper):
    obj_dict = cls.__dict__
    props = [
        attr 
        for attr in obj_dict 
        if isinstance(obj_dict[attr], property)
        if isinstance(obj_dict[attr].fget, wrapper)
    ]

    cprops = [
        attr
        for attr in obj_dict
        if isinstance(obj_dict[attr], cached_property)
        if isinstance(obj_dict[attr].func, wrapper)
    ]

    return props + cprops


def set_prop(self, lhs, rhs, wrapper, factory):
    for l_attr, r_attr in zip(lhs, rhs):

        setattr(
            self.__class__,
            r_attr,
            wrapper.prop(
                getattr(PropertyFactory(l_attr), factory)
            )
        )
        if wrapper.prop is cached_property:
            self.__class__.__dict__[r_attr].__set_name__(self.__class__, r_attr)

    