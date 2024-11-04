from functools import cached_property, update_wrapper
import numpy as np
from new_fave.measurements.calcs import (mahalanobis,
    mahal_log_prob,
    param_to_cov,
    cov_to_icov,
    clear_cached_properties
)

from typing import Any
from nptyping import NDArray, Shape, Float
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from new_fave.measurements.vowel_measurement import (VowelMeasurement,
        VowelClass,
        VowelClassCollection)

class AggWrapper:
    """
    A base class for Optimization parameter wrappers
    """
    def __init__(self, func):
        update_wrapper(self, func)
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    
class OutputWrapper:
    def __init__(self, func):
        update_wrapper(self, func)
        self.func = func
    
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    pass

class MahalWrap(AggWrapper):
    """ Mahalanobis Wrapper

    A wrapper to mark parameters for
    mahalanobis distance estimation,
    to be set as a property
    """
    prop:type[property] = property
    def __init__(self, func):
        super().__init__(func)

class MahalCacheWrap(AggWrapper):
    """Cached Mahalanobis Wrapper

    A wrapper to mark parameters for
    mahalanobis distance estimation,
    to be set as a property
    """
    prop:type[cached_property] = cached_property
    def __init__(self, func):
        super().__init__(func)

class FlatCacheWrap(AggWrapper):
    prop:type[cached_property] = cached_property
    def __init__(self, func):
        super().__init__(func)

class PropertyFactory():
    """ A property factory
    
    A property factory to generate 
    the necessary properties in aggregated classes 
    (VowelClass and VowelClassCollection)
    as well as log probability properties
    in VowelMeasurement

    """
    def __init__(self, attr:str, *args):
        self.attr = attr
    
    @property
    def winner_factory(self):
        """The parameters of the winners
        """
        def parameterized_func(
                obj:'VowelClass|VowelClassCollection'
            ):
            if not hasattr(obj, "vowel_measurements"):
                return None
            return np.array([
                getattr(x, self.attr)[...,x.winner_index].T
                for x in obj.vowel_measurements
            ]).T
        
        return parameterized_func
    
    @property
    def mean_factory(self):
        """The mean parameters of the winners
        """

        def parameterized_func(obj:'VowelClass|VowelClassCollection'):
            if not hasattr(obj, "vowel_measurements"):
                return None

            winner_array  = getattr(obj, self.attr)
            N = winner_array.shape[-1]
            square_array = winner_array.reshape(-1, N)
            winner_mean =  square_array.mean(axis = 1)
            winner_mean = winner_mean[:, np.newaxis]
            return winner_mean
        
        return parameterized_func
    
    @property
    def icov_factory(self):
        """The inverse covariance matrix of the winners.
        """

        def parameterized_func(obj:'VowelClass|VowelClassCollection'):
            if not hasattr(obj, "vowel_measurements"):
                return None            
            cov_mat = param_to_cov(getattr(obj, self.attr))
            icov_mat = cov_to_icov(cov_mat)
            return icov_mat

        return parameterized_func
    
    @property
    def agg_factory(self):
        """Aggregate single numeric values across vowel measurements
        """
        def parameterized_func(obj: 'VowelClass|VowelClassCollection'):
            if not hasattr(obj, "vowel_measurements"):
                return None
            agged = np.array([
                getattr(x, self.attr)
                for x in obj.vowel_measurements
            ])

            agged = agged.flatten()
            return agged
        
        return parameterized_func

    @property
    def speaker_byvclass(self):
        """log probability (based on mahalanobis distance)
        of candidate measurements from the vowel class
        within a speaker
        """
        def parameterized_func(obj:'VowelMeasurement'):
            if not hasattr(obj, "candidates"):
                return None
            if not obj.vowel_class:
                return None
            mean_vec_name = self.attr.replace("cand_", "winner_") + "_mean"
            icov_mat_name = mean_vec_name.replace("_mean", "_icov")

            cand_vals = getattr(obj, self.attr)
            mean_vals = getattr(obj.vowel_class, mean_vec_name)
            icov_vals = getattr(obj.vowel_class, icov_mat_name)

            cand_vals = cand_vals.reshape(-1, cand_vals.shape[-1])

            mahals = mahalanobis(cand_vals, mean_vals, icov_vals)
            logprob = mahal_log_prob(mahals, cand_vals)

            return logprob
        
        return parameterized_func
    

    @property
    def speaker_global(self):
        """log probability (based on mahalanobis distance)
        of candidate measurements from the entire speaker's 
        distribution
        """        
        def parameterized_func(obj:'VowelMeasurement'):
            if not hasattr(obj, "candidates"):
                return None
            if not obj.vowel_class and not obj.vowel_class.vowel_system:
                return None            
            mean_vec_name = self.attr.replace("cand_", "winner_") + "_mean"
            icov_mat_name = mean_vec_name.replace("_mean", "_icov")

            cand_vals = getattr(obj, self.attr)
            mean_vals = getattr(obj.vowel_class.vowel_system, mean_vec_name)
            icov_vals = getattr(obj.vowel_class.vowel_system, icov_mat_name)

            cand_vals = cand_vals.reshape(-1, cand_vals.shape[-1])

            mahals = mahalanobis(cand_vals, mean_vals, icov_vals)
            logprob = mahal_log_prob(mahals, cand_vals)

            return logprob
        
        return parameterized_func
    
    pass

def get_wrapped(cls:type, wrapper:MahalWrap|MahalCacheWrap|FlatCacheWrap|OutputWrapper) -> list[str]:
    """Get the class property names that have been wrapped with `wrapper`

    Args:
        wrapper (MahalWrap|MahalCacheWrap):
            The wrapper to check for 

    Returns:
        (list[str]):
            A list of attribute names
    """
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


def set_prop(
        self:'VowelMeasurement|VowelClass|VowelClassCollection', 
        lhs:str, 
        rhs:str, 
        wrapper:MahalWrap|MahalCacheWrap|FlatCacheWrap, 
        factory:str
    ) -> None:
    """Set a property on a class

    Args:
        self (VowelMeasurement|VowelClass|VowelClassCollection):
            The object to set the property on
        lhs (str): Name of the generating attribute
        rhs (str): Name of the generated attribute
        wrapper (MahalWrap | MahalCacheWrap): The wrapper 
        factory (str): The name of the property factory
    """
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

    