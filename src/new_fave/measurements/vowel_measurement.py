"""
This module contains classes to represent vowel measurements and their
aggregations at different levels.

```{mermaid}
classDiagram
direction LR

class VowelMeasurement~list~{
    .vowel_class
}
class VowelClass~list~{
    .vowel_system
}
class VowelClassCollection~dict~{
    .corpus
}
class SpeakerCollection~dict~

SpeakerCollection --o VowelClassCollection
VowelClassCollection --o VowelClass
VowelClass --o VowelMeasurement
```

When a class has a numpy array for an attribute, its 
type is annotated using [nptyping](https://pypi.org/project/nptyping/)
to provide the expected dimensions. For example:

```
cand_param (NDArray[Shape["Param, Formant, Cand"], Float])
```

This indicates that `cand_param` is a three dimensional array.
The first dimension is `"Param"` (the number of DCT parameters)
long, the second is `"Formant"` (the number of formants) long, 
and the third is `"Cand"` (the number of candidates) long.
"""
import fasttrackpy
from fasttrackpy import CandidateTracks, OneTrack
from aligned_textgrid import AlignedTextGrid, SequenceInterval
from fave_measurement_point.heuristic import Heuristic
from fave_measurement_point.formants import FormantArray

from new_fave.utils.textgrid import get_textgrid
from new_fave.speaker.speaker import Speaker
from new_fave.measurements.calcs import (mahalanobis, 
    mahal_log_prob,
    param_to_cov,
    cov_to_icov,
    clear_cached_properties
)

from new_fave.measurements.reference import ReferenceValues

from new_fave.measurements.decorators import (MahalWrap,
    MahalCacheWrap,
    FlatCacheWrap,
    get_wrapped,
    set_prop
)

from collections import defaultdict
import numpy as np
from typing import Literal, ClassVar
import polars as pl

import scipy.stats as stats
from scipy.fft import idst, idct

import librosa

from joblib import Parallel, delayed, cpu_count

from collections.abc import Sequence, Iterable
from dataclasses import dataclass, field
from nptyping import NDArray, Shape, Float

from functools import lru_cache, cached_property

import re

NCPU = cpu_count()

import warnings

def blank():
    return VowelClass()

def blank_list():
    return []

EMPTY_LIST = blank_list()

class PropertySetter:
    """
    A mixin class to dynamically create properties 
    necessary for calculating log-probabilities
    from properties decorated with either MahalWrap
    or MahalCacheWrap.
    """

    def _make_attrs(self):
        for wrapper in [MahalWrap, MahalCacheWrap]:
            cand_attrs = get_wrapped(VowelMeasurement, wrapper)

            winner_attrs = [
                x.replace("cand_", "winner_")
                for x in cand_attrs
            ]

            set_prop(self, cand_attrs, winner_attrs, wrapper, "winner_factory")
            
            mean_attrs = [
                attr + "_mean" 
                for attr in winner_attrs
            ]
            set_prop(self, winner_attrs, mean_attrs, wrapper, "mean_factory")
            

            icov_attrs = [
                attr + "_icov" 
                for attr in winner_attrs
            ]

            set_prop(self, winner_attrs, icov_attrs, wrapper, "icov_factory")

            speaker_byvclass_attrs = [
                attr+"_logprob_speaker_byvclass"
                for attr in cand_attrs
            ]

            set_prop(self, cand_attrs, speaker_byvclass_attrs, wrapper, "speaker_byvclass")

            speaker_global_attrs = [
                attr+"_logprob_speaker_global"
                for attr in cand_attrs
            ]

            set_prop(self, cand_attrs, speaker_global_attrs, wrapper, "speaker_global")

        for wrapper in [FlatCacheWrap]:
            cand_attrs = get_wrapped(VowelMeasurement, wrapper)

            agg_attrs = [attr + "s" for attr in cand_attrs]

            set_prop(self, cand_attrs, agg_attrs, wrapper, "agg_factory")



@dataclass
class VowelMeasurement(Sequence, PropertySetter):
    """ A class used to represent a vowel measurement.

    ## Intended Usage
    Certain properties of a `VowelMeasurement` instance
    are set by its membership within a [](`~new_fave.VowelClass`)
    and that [](`~new_fave.VowelClass`)'s membership 
    in a [](`~new_fave.VowelClassCollection`). These memberships
    are best managed by passing a list of `VowelMeasurement`s to
    [](`~new_fave.SpeakerCollection`).

    ```{.python}
    vowel_measurements = [VowelMeasurement(t) for t in fasttrack_tracks]
    speakers = SpeakerCollection(vowel_measurements)
    ```

    Args:
        track (fasttrackpy.CandidateTracks): 
            A fasttrackpy.CandidateTrracks object
        heuristic (Heuristic, optional): 
            A point measurement Heuristic to use. 
            Defaults to Heuristic().
        vowel_place_dict (dict[Literal["front","back"], re.Pattern]):
            A dictionary of regexes that match front or back vowels.
    
    Attributes: 
        track (fasttrackpy.CandidateTracks):
            an object of CandidateTracks class
        candidates (list):
            list of candidates for the track
        heuristic (Heuristic, optional):
            an object of Heuristic class (default is Heuristic())
        vowel_class (VowelClass):
            The containing VowelClass object

        formant_array (FormantArray): 
            A FormantArray object            

        file_name (str):
            name of the file of the track
        group (str):
            TierGroup of the track
        id (str):
            id of the track
        interval (aligned_textgrid.SequenceInterval):
            interval of the track
        label (str):
            label of the track
        n_formants (int):
            number of formants in the track
        optimized (int):
            The number of optimization iterations the
            vowel measurement has been through.

        winner: fasttrackpy.OneTrack
            The winning formant track
        winner_index (int):
            The index of the winning formant track
        
        cand_param (NDArray[Shape["Param, Formant, Cand"], Float]):
            A array of the candidate DCT parameters.
        cand_maxformant (NDArray[Shape["1, Cand"], Float]):
            An array of the candidate maximum formants.
        cand_error (NDArray[Shape["Cand"], Float]):
            An array of the candidate smoothing error.

        cand_error_logprob_vm (NDArray[Shape["Cand"], Float]):
            Conversion of the smooth error to log probabilities. The candidate with
            the lowest error = log(1), and the candidate with the largest 
            error = log(0).
        cand_param_(mahal/logprob)_speaker_byvclass (NDArray[Shape["Cand"], Float]):
            The mahalanobis distance (`mahal`) or associated log probability (`logprob`) 
            for each candidate relative to the VowelClass for this speaker.
            These are calculated by drawing the relevant mean and covariance matrix from 
            `vm.vowel_class`
        cand_param_(mahal/logprob)_speaker_global (NDArray[Shape["Cand"], Float]):
            The mahalanobis distance (`mahal`) or associated log probability (`logprob`) 
            for each candidate relative to *all* vowel measurements for this speaker.
            These are calculated by drawing the relevant mean and covariance matrix from 
            `vm.vowel_class.vowel_system`
        cand_param_(mahal/logprob)_corpus_byvclass (NDArray[Shape["Cand"], Float]):
            The mahalanobis distance (`mahal`) or associated log probability (`logprob`) 
            for each candidate relative to this vowel class across all speakers.
            These are calculated by drawing the relevant mean and covariance matrix from 
            `vm.vowel_class.vowel_system.corpus`

        point_measure (pl.DataFrame):
            A polars dataframe of the point measurement for this vowel.
        vm_context (pl.DataFrame):
            A polars dataframe of contextual information for the vowel measurement.
    """
    track: CandidateTracks
    heuristic: Heuristic = field(default = Heuristic())
    vowel_place_dict: dict[Literal["front", "back"], re.Pattern] = field(default_factory=lambda : dict())
    reference_values: ReferenceValues = field(default = ReferenceValues())
    only_fasttrack: bool = field(default=False)
    def __post_init__(
            self
        ):
        super().__init__()
        #self.label = self.track.label
        self.candidates = self.track.candidates
        self.n_formants = self.track.n_formants
        self._label = None
        self.interval = self.track.interval
        self.group = self.track.group
        self.id = self.track.id
        self.file_name = self.track.file_name
        self._expanded_formants = None
        self._optimized = 0
        self._init_winner()
        self._make_attrs()
        


    def __getitem__(self,i):
        return self.candidates[i]
    
    def __len__(self):
        return len(self.candidates)
    
    def __repr__(self):
        out = (
            "VowelMeasurement: {"
            f"label: {self.label}, "
            f"samples: {self.winner.formants.shape[1]}, "
            f"optimized: {self.optimized}"
            "}"
        )
        return out
    
    def _init_winner(self):
        
        joint = self.cand_error_logprob_vm + self.reference_logprob
        # if self.spectral_rolloff < 7:
        #     joint += self.place_penalty/100

        idx = np.nanargmax(joint)

        self._winner = self.track.candidates[idx]
    
    @property
    def label(self) -> str:
        if (not self._label) or (self._label != self.interval.label):
            for cand in self.candidates:
                cand.label = self.interval.label
            self.track.label = self.interval.label
            self._label = self.interval.label

        return self.interval.label

    @label.setter
    def label(self, x:str):
        self.interval._label = x


    @cached_property
    def place(self) -> str:
        for k in self.vowel_place_dict:
            if re.match(self.vowel_place_dict[k], self.label):
                return k
        
        return "unk"
            

    @property
    def formant_array(self) -> FormantArray:
        return FormantArray(
            self.winner.smoothed_formants, 
            self.winner.time_domain,
            offset = self.track.window_length
        )
    
    @property
    def vowel_class(self):
        if self._vclass:
            return self._vclass
    
    @vowel_class.setter
    def vowel_class(self, vclass: 'VowelClass'):
        self._vclass = vclass

    @property
    def winner(self)->OneTrack:
        return self._winner
    
    @winner.setter
    def winner(self, idx):
        self._winner = self.candidates[idx]
        self._reset_winners()
        self.vowel_class.vowel_system._reset_winners()
        self.vowel_class._reset_winners()
        self._expanded_formants = None
        self._optimized += 1

    def _reset_winners(self):
        clear_cached_properties(self)

    @property
    def optimized(self)->int:
        return self._optimized
    
    @property
    def winner_index(self)->int:
        return self.candidates.index(self.winner)
    
    @property
    def expanded_formants(
        self
    )->NDArray[Shape['N, Formant, Cand'], Float]:
        if self._expanded_formants is not None:
            return self._expanded_formants

        self._expanded_formants = np.apply_along_axis(
            lambda x: idct(x.T, n = 20, orthogonalize=True, norm = "forward"),
            0,
            self.cand_param
        )
        return self._expanded_formants    

    # @cached_property
    # @FlatCacheWrap
    # def spectral_rolloff(self) -> float:
    #     n_fft_power = 11
    #     while self.track.samples.size < 2 ** n_fft_power:
    #         n_fft_power -= 1
        
    #     rolloff = librosa.feature.spectral_rolloff(
    #         y = self.track.samples[0] + 0.01,
    #         sr = self.track.sampling_frequency,
    #         n_fft = 2**n_fft_power
    #     ).squeeze()

    #     third = rolloff.size // 3

    #     log_rolloff = np.log(
    #         rolloff[third:-third]
    #     ).mean()

    #     return log_rolloff

    
    @cached_property
    def params(
        self
    ):
        params = np.array(
            [
                x.parameters
                for x in self.candidates
            ]
        ).T
        return params
    
    @cached_property
    def logparams(
        self
    ):
        params = np.array(
            [
                x.log_parameters
                for x in self.candidates
            ]
        ).T
        return params
    
    @cached_property
    def point_values(
        self
    ):
        params = self.params[0]
        bparams = self.cand_bparam[0]

        param_bparam = np.concatenate([params, bparams])
        return param_bparam

    @cached_property
    @MahalCacheWrap
    def cand_param(
        self
    ) -> NDArray[Shape["Param, Formant, Cand"], Float]:
        params = np.array(
            [
                x.log_parameters
                for x in self.candidates
            ]
        ).T
        params = np.concatenate((params, self.cand_bparam))

        return params
    
    @cached_property
    @MahalCacheWrap
    def cand_squareparam(
        self
    ) -> NDArray[Shape["X, Cand"], Float]:
        params = np.array(
            [
                x.log_parameters
                for x in self.candidates
            ]
        ).T
        params = np.concatenate((params, self.cand_bparam))
        square_params = params.reshape((-1, params.shape[-1]))

        return square_params    
    
    @cached_property
    @MahalCacheWrap
    def cand_centroid(
        self
    ) -> NDArray[Shape["Param, Formant, Cand"], Float]:
        params = np.array(
            [
                x.log_parameters
                for x in self.candidates
            ]
        ).T
        params = params[0,:2,:]
        params = np.expand_dims(params, 0)

        return params    
    
    @cached_property
    @MahalCacheWrap
    def cand_fratio(
        self
    ) -> NDArray[Shape["1, Formant"], Float]:
        """The formant ratios

        Returns:
            (np.ndarray):
                A numpy array of the ratios of formants.
                Necessarilly the number of formants minus 1.
        """
        params = np.array(
            [
                x.parameters
                for x in self.candidates
            ]
        ).T

        params = params[0, :, :]
        ratios = np.diff(
                np.log(params),
                axis = 0
            )
        
        ratios = np.expand_dims(ratios, 0)
        return ratios

    @property
    #@MahalCacheWrap
    def cand_bparam(
        self
    ) -> NDArray[Shape["Param, Formant, Cand"], Float]:
        params = np.array([
            x.bandwidth_parameters
            for x in self.candidates
        ]).T

        params = params[0, :, :]
        params = np.expand_dims(params, 0)
    
        return params

    @cached_property
    @MahalCacheWrap
    def cand_maxformant(
        self
    ) -> NDArray[Shape["1, 1, Cand"], Float]:
        mf = np.array([[
            c.maximum_formant
            for c in self.candidates
        ]])

        #mf = mf.reshape((1, np.newaxis, mf.shape[-1]))
        return mf
    
    @cached_property
    def cand_error(
        self
    ) -> NDArray[Shape["Cand"], Float]:
         with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.array([
                c.smooth_error
                for c in self.candidates
            ])
         
    @cached_property
    def place_penalty(
        self
    ) -> NDArray[Shape["Cand"], Float]:
        if not self.place in ["front", "back"]:
            return np.zeros(shape = self.cand_maxformant.shape).squeeze()

        mf_exp =np.power(1.0001, self.cand_maxformant)
        mf_norm = mf_exp - mf_exp.min()
        mf_surv = mf_norm/mf_norm.max()
        
        if self.place == "back":
            mf_surv = 1 - mf_surv

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            penalty = np.log(mf_surv).squeeze()

        return penalty

    @cached_property
    def cand_b2_logprob(self):
        f2_bandwidth = self.cand_bparam[0,1,:]
        f2_norm = f2_bandwidth - np.nanmin(f2_bandwidth)
        f2_surv = 1 - (f2_norm/np.nanmax(f2_norm))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f2_logprob = np.log(f2_surv)

        return f2_logprob

    @property
    def cand_error_logprob_vm(
        self
    ) -> NDArray[Shape["Cand"], Float]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            err_norm = self.cand_error - np.nanmin(self.cand_error)
            err_surv = 1 - (err_norm/np.nanmax(err_norm))
            err_log_prob = np.log(err_surv)
        return err_log_prob

    @cached_property
    def reference_logprob(self):
        if self.reference_values.reference_type is None:
            return np.zeros(len(self))
        
        if not self.label in self.reference_values.mean_dict:
            return np.zeros(len(self))

        if np.isfinite(self.reference_values.icov_dict[self.label]).mean() < 0.5:
            return np.zeros(len(self))
        
        if self.reference_values.reference_type == "logparam":
            params = self.logparams

        if self.reference_values.reference_type == "param":
            params = self.params

        if self.reference_values.reference_type == "points":
            params = self.point_values
        
        params = params.reshape((-1, params.shape[-1]))
        mahals = mahalanobis(
            params = params,
            param_means=self.reference_values.mean_dict[self.label],
            inv_cov = self.reference_values.icov_dict[self.label]
        )
        log_prob = mahal_log_prob(
            mahals=mahals,
            params=params
        )
        
        return log_prob

    @property
    def point_measure(
        self
    ) -> pl.DataFrame:
        winner_slice =  self.heuristic.apply_heuristic(
            self.label,
            formants=self.formant_array
        )

        point_dict = {
            f"F{i+1}": winner_slice.formants[i]
            for i in range(winner_slice.formants.size)
        }
        bandwidth_params = self.cand_bparam[0, :,self.winner_index]
        for idx, param in enumerate(bandwidth_params):
            point_dict[f"B{idx+1}"] = param
        
        point_dict["max_formant"] = self.winner.maximum_formant
        #point_dict["spectral_rolloff"] = self.spectral_rolloff
        point_dict["smooth_error"] = self.winner.smooth_error
        point_dict["time"] = winner_slice.time
        point_dict["rel_time"] = winner_slice.rel_time
        point_dict["prop_time"] = winner_slice.prop_time
        point_dict["id"] = self.winner.id
        point_dict["label"] = self.winner.label
        point_dict["file_name"] = self.winner.file_name
        point_dict["group"] = self.winner.group

        return pl.DataFrame(point_dict)
    
    @cached_property
    def vm_context(
        self
    ) -> pl.DataFrame:   
        optimized = self.optimized
        id = self.winner.id
        word = self.winner.interval.within.label
        dur = self.winner.interval.end - self.winner.interval.start
        pre_word = self.winner.interval.within.prev.label
        fol_word = self.winner.interval.within.fol.label
        pre_seg = self.winner.interval.prev.label
        fol_seg = self.winner.interval.fol.label
        abs_pre_seg = self.winner.interval.get_tierwise(-1).label
        abs_fol_seg = self.winner.interval.get_tierwise(1).label
        stress = ""
        if hasattr(self.track.interval, "stress"):
            stress = self.track.interval.stress

        context = "internal"
        if pre_seg == "#" and fol_seg != "#":
            context = "initial"
        if pre_seg != "#" and fol_seg == "#":
            context = "final"
        if pre_seg == "#" and fol_seg == "#":
            context = "coextensive"

        df = pl.DataFrame({
            "optimized": optimized,
            "id": id,
            "word": word,
            "stress": stress,
            "dur": dur,
            "pre_word": pre_word,
            "fol_word": fol_word,
            "pre_seg": pre_seg,
            "fol_seg": fol_seg,
            "abs_pre_seg": abs_pre_seg,
            "abs_fol_seg": abs_fol_seg,
            "context": context
        })

        return df
    
    def to_tracks_df(self) -> pl.DataFrame:
        """Return a DataFrame of the formant tracks

        Returns:
            (pl.DataFrame): 
                A dataframe with formant track data.
        """
        df = self.winner.to_df()
        df = df.with_columns(
            speaker_num = (
                pl.col("id")
                .str.extract("^(\d+)-")
                .str.to_integer() + 1
            )
        )

        df = df.join(self.vm_context, on = "id")

        return df

    def to_param_df(
            self, 
            output:Literal["param", "log_param"] = "log_param"
        ) -> pl.DataFrame:
        """Return DataFrame of formant DCT parameters.

        Returns:
            (pl.DataFrame):
                A DataFrame of formant DCT parameters
        """
        df = self.winner.to_df(output=output)
        df = df.with_columns(
            max_formant = self.winner.maximum_formant,
            speaker_num = (
                pl.col("id")
                .str.extract("^(\d+)-")
                .str.to_integer() + 1
            )
        )
        
        df = df.join(self.vm_context, on = "id")
        
        return df

    def to_point_df(self) -> pl.DataFrame:
        """Return a DataFrame of point measurements

        Returns:
            (pl.DataFrame): 
                A DataFrame of vowel point measures.
        """
        df = self.point_measure
        df = df.with_columns(
            speaker_num = (
                pl.col("id")
                .str.extract("^(\d+)-")
                .str.to_integer() + 1
            ),
            point_heuristic = pl.lit(self.heuristic.heuristic)
        )

        df = df.join(self.vm_context, on = "id")

        return(df)

@dataclass
class VowelClass(Sequence, PropertySetter):
    """A class used to represent a vowel class.

    ## Intended Usage

    `VowelClass` subclasses [](`collections.abc.Sequence`), so 
    it is indexable. While it can be created on its own, it is 
    best to leave this up to either [](`~new_fave.VowelClassCollection`)
    or [](`~new_fave.SpeakerCollection`).

    ```{.python}
    vowel_measurements = [VowelMeasurement(t) for t in fasttrack_tracks]
    vowel_class = VowelClass("ay", vowel_measurements)
    ```

    Args:
        label (str):
            The vowel class label
        vowel_measurements (list[VowelMeasurement]): 
            A list of VowelMeasurements

    Attributes:
        label (str):
            label of the vowel class
        tracks (list):
            A list of VowelMeasurements
        vowel_system (VowelClassCollection):
            The containing vowel system
        winners (list[OneTrack]):
            A list of winner [](`~fasttrackpy.OneTrack`)s from the vowel class
        winner_param (NDArray[Shape["Param, Formant, N"], Float]):
            An np.array of winner DCT parameters from the vowel class
        winner_param_mean (NDArray[Shape["ParamFormant, 1"], Float]):
            Mean of winner DCT parameters
        winner_param_cov (NDArray[Shape["ParamFormant, ParamFormant"], Float]):
            Covariance of winner DCT parameters
        winner_param_icov (NDArray[Shape["ParamFormant, ParamFormant"], Float]):
            Inverse covariance of winner DCT parameters
    """
    label: str = field(default="")
    vowel_measurements: list[VowelMeasurement] = field(default_factory= lambda : [])
    containing_class: ClassVar[type] = VowelMeasurement
    scope: ClassVar[str] = "speaker_byvclass"

    def __post_init__(self):
        super().__init__()
        self._make_attrs()
        self._winners = [x.winner for x in self.vowel_measurements]
        for t in self.vowel_measurements:
            t.vowel_class = self

    def __getitem__(self, i):
        return self.vowel_measurements[i]
    
    def __len__(self):
        return len(self.vowel_measurements)
    
    def __lt__(self, other):
        return len(self) < len(other)
    
    def __le__(self, other):
        return len(self) <= len(other)
    
    def __repr__(self):
        out = (
            "VowelClass: {"
            f"label: {self.label}, "
            f"len: {len(self)}"
            "}"
        )
        return out
    
    def _reset_winners(self):
        clear_cached_properties(self)

    
    @property
    def vowel_system(self):
        return self._vowel_system
    
    @vowel_system.setter
    def vowel_system(self, vowel_system: 'VowelClassCollection'):
        self._vowel_system = vowel_system

    @cached_property
    def winners(self):
        return [x.winner for x in self.vowel_measurements]
    
    def to_param_df(
            self, 
            output:Literal["param", "log_param"] = "log_param"
        ) -> pl.DataFrame:
        """Return DataFrame of formant DCT parameters.

        Returns:
            (pl.DataFrame):
                A DataFrame of formant DCT parameters
        """
        df = pl.concat(
            [x.to_param_df(output=output) for x in self.vowel_measurements]
        )

        return df

    def to_tracks_df(
            self
        ) -> pl.DataFrame:
        """Return DataFrame of formanttracks.

        Returns:
            (pl.DataFrame):
                A DataFrame of formant tracks
        """
        df = pl.concat(
            [x.to_tracks_df() for x in self.vowel_measurements]
        )

        return df    
    
    def to_point_df(self) -> pl.DataFrame:
        """Return a DataFrame of point measurements

        Returns:
            (pl.DataFrame): 
                A DataFrame of vowel point measures.
        """        
        df = pl.concat(
            [x.to_point_df() for x in self.vowel_measurements]
        )

        return df

class VowelClassCollection(defaultdict, PropertySetter):
    """
    A class for an entire vowel system. 
    
    ## Intended Usage
    It is a subclass of `defaultdict`, so it can be 
    keyed by vowel class label.

    ```{.python}
    vowel_measurements = [VowelMeasurement(t) for t in fasttrack_tracks]
    vowel_system = VowelClassCollection(vowel_measurements)
    ```

    Args:
        track_list (list[VowelMeasurement]):
            A list of `VowelMeasurement`s.

    Attributes:
        winners (list[OneTrack]):
            All winner tracks from the entire vowel system.
        vowel_measurements (list[VowelMeasurement]):
            All `VowelMeasurement` objects within this vowel system
        textgrid (AlignedTextGrid):
            The `AlignedTextGrid` associated with this vowel system.
        winner_expanded_formants (NDArray[Shape["20, FormantN"], Float]):
            A cached property that returns the expanded formants for the winners.

            
        winner_param (NDArray[Shape["Param, Formant, N"], Float]):
            An array of all parameters from all winners across the 
            vowel system.
        winner_maxformant (NDArray[Shape["1, N"], Float]):
            An array of the maximum formants of all winners across
            the vowel system

        winner_param_mean (NDArray[Shape["1, FormantParam"], Float]):
            The mean of all DCT parameters across all formants for the winners
            in this vowel system.
        winner_param_cov (NDArray[Shape["FormantParam, FormantParam"], Float]):
            The covariance of all parameters across all formants for the winners
            in this vowel system
        winner_param_icov (NDArray[Shape["FormantParam, FormantParam"], Float]):
            The inverse of `winner_param_cov`.

        winner_maxformant_mean (float):
            The mean maximum formant across all winners in this vowel system.
        winner_maxformant_cov (NDArray[Shape["1, 1"], Float]):
            The covariance of the maximum formant across all winners
            in this vowel system.
        winner_maxformant_icov (NDArray[Shape["1, 1"], Float]):
            The inverse of `winner_maxformant_cov`
    """
    containing_class = VowelClass
    scope = "speaker_global"
    def __init__(self, track_list:list[VowelMeasurement] = EMPTY_LIST):
        super().__init__(blank)
        self.track_list = track_list
        self.tracks_dict = defaultdict(blank_list)
        if isinstance(self.track_list, Iterable):
            self._make_tracks_dict()
            self._dictify()
        self._vowel_system()
        self._file_name = None
        self._group = None
        self._corpus = None
        self._make_attrs()


    def __setitem__(self, __key, __value) -> None:
        super().__setitem__(__key, __value)

    def _reset_winners(self):
        clear_cached_properties(self)

    def _make_tracks_dict(self):
        for v in self.track_list:
            self.tracks_dict[v.label].append(v)

    def _dictify(self):
        for v in self.tracks_dict:
            self[v] = VowelClass(
                v, 
                self.tracks_dict[v]
            )

    def _vowel_system(self):
        for v in self.tracks_dict:
            self[v].vowel_system = self
    
    def _reset_winners(self):
        clear_cached_properties(self)

    @property
    def sorted_keys(self):
        return sorted(self, key=lambda k: -len(self[k]))
    
    @property
    def corpus(self):
        return self._corpus
    
    @corpus.setter
    def corpus(self, corp):
        self._corpus = corp
    
    @cached_property
    def winners(
        self
    ) -> list[OneTrack]:
        return [
            x
            for vc in self.values()
            for x in vc.winners
        ]
    
    @cached_property
    def vowel_measurements(
        self
    ) -> list[VowelMeasurement]:
        return [
            x  
            for vc in self.values()
            for x in vc.vowel_measurements
        ]
    
    @cached_property
    def textgrid(
        self
    ) -> AlignedTextGrid:
        return get_textgrid(self.vowel_measurements[0].interval)
    
    @cached_property
    def file_name(
        self
    ) -> str:
        if self._file_name:
            return self._file_name
        
        self._file_name = self.vowel_measurements[0].winner.file_name
        return self._file_name
    
    @cached_property
    def group(
        self
    ) -> str:
        if self._group:
            return self._group

        self._group = self.vowel_measurements[0].winner.group
        return self._group
    
    @cached_property
    def corpus_key(
        self
    ) -> tuple[str, str]:
        return (self.file_name, self.group)

    @cached_property
    def winner_expanded_formants(
        self
    ) -> NDArray[Shape["20, FormantN"], Float]:
        formants = np.hstack(
            [
                x.expanded_formants[:, :, x.winner_index]
                for x in self.vowel_measurements
            ]
        )

        return formants
                
    def to_tracks_df(self)->pl.DataFrame:
        """Return a DataFrame of the formant tracks

        Returns:
            (pl.DataFrame): 
                A dataframe with formant track data.
        """        
        df = pl.concat(
            [x.to_tracks_df() for x in self.values()]
        )

        return df
    
    def to_param_df(
            self, 
            output:Literal["param", "log_param"] = "log_param"
        ) -> pl.DataFrame:
        """Return DataFrame of formant DCT parameters.

        Returns:
            (pl.DataFrame):
                A DataFrame of formant DCT parameters
        """        
        df = pl.concat(
            [x.to_param_df(output = output) for x in self.values()]
        )

        return df    
    
    def to_point_df(self) -> pl.DataFrame:
        """Return a DataFrame of point measurements

        Returns:
            (pl.DataFrame): 
                A DataFrame of vowel point measures.
        """                
        df = pl.concat(
            [x.to_point_df() for x in self.values()]
        )

        return df
    
class SpeakerCollection(defaultdict, PropertySetter):
    """
    A class to represent the vowel system of all 
    speakers in a TextGrid. 
    
    ## Intended usage
    It is a subclass of `defaultdict`,
    and can be keyed by the `(file_name, group_name)` tuple.

    ```{.python}
    vowel_measurements = [VowelMeasurement(t) for t in fasttrack_tracks]
    speakers = SpeakerCollection(vowel_measurements)
    ```

    Args:
        track_list (list[VowelMeasurement]):
            A list of `VowelMeasurement`s.
    """
    __hash__ = object.__hash__

    containing_class = VowelClassCollection

    def __init__(self, track_list:list[VowelMeasurement] = []):
        self.track_list = track_list
        self.speakers_dict = defaultdict(blank_list)
        self._make_tracks_dict()
        self._dictify()
        self._speaker = None
        self._associate_corpus()
        self._make_attrs()
    
    def __setitem__(self, __key, __value) -> None:
        super().__setitem__(__key, __value)
        self._associate_corpus()

    def _make_tracks_dict(self):
        for v in self.track_list:
            file_speaker = (v.file_name, v.group)
            self.speakers_dict[file_speaker].append(v)
        
    def _dictify(self):
        for fs in self.speakers_dict:
            self[fs] = VowelClassCollection(
                self.speakers_dict[fs]
            )
    
    def _associate_corpus(self):
        for speaker in self.values():
            speaker.corpus = self

    def _reset_winners(self):
        clear_cached_properties(self)

    @cached_property
    def vowel_dict(
        self
    ) -> defaultdict[str, list[VowelMeasurement]]:
        out = defaultdict(blank_list)
        for speaker in self.values():
            for vowel in speaker:
                out[vowel] += speaker[vowel]
        return out
    
    @cached_property
    def vowel_winners(
        self
    ) -> defaultdict[str, list[OneTrack]]:
        out = defaultdict(blank_list)
        for vowel in self.vowel_dict:
            out[vowel] += [x.winner for x in self.vowel_dict[vowel]]

        return out
    
    @cached_property
    def winner_param(
        self
    ) -> defaultdict[str, NDArray[Shape["Param, Formant, N"], Float]]:
        out = defaultdict(blank_list)
        for vowel in self.vowel_winners:
            params = np.array(
                [
                    x.parameters
                    for x in self.vowel_winners[vowel]
                ]
            ).T

            out[vowel] = params
        return out
    
    @cached_property
    def winner_param_mean(
        self
    ) -> defaultdict[str, NDArray[Shape["FormantParam, 1"], Float]]:
        out = defaultdict(lambda: np.array([]))

        for vowel in self.winner_param:
            N = len(self.vowel_dict[vowel])
            winner_mean =  self.winner_param[vowel].reshape(-1, N).mean(axis = 1)
            winner_mean = winner_mean[:, np.newaxis]
            out[vowel] = winner_mean
        return out

    
    @property
    def winner_param_cov(
        self
    )->defaultdict[str, NDArray[Shape["FormantParam, FormantParam"], Float]]:
        out = defaultdict(lambda: np.array([]))

        for vowel in  self.winner_param:
            param_cov = param_to_cov(self.winner_param[vowel])
            out[vowel] = param_cov
        
        return out

    @cached_property
    def winner_param_icov(
        self
    )->defaultdict[str, NDArray[Shape["FormantParam, FormantParam"], Float]]:
        out = defaultdict(lambda: np.array([]))

        for vowel in self.winner_param_cov:
            params_icov = cov_to_icov(self.winner_param_cov[vowel])
            out[vowel] = params_icov

        return out


    @property 
    def speaker(self):
        return self._speaker

    @speaker.setter
    def speaker(self, speaker:Speaker):
        self._speaker = speaker
    
    def to_tracks_df(self)->pl.DataFrame:
        """
        This will return a data frame of formant 
        tracks for all speakers.

        Returns:
            (pl.DataFrame): A dataframe of formant tracks for all speakers.
        """
        df = pl.concat(
            [x.to_tracks_df() for x in self.values()]
        )
        
        joinable = False
        if self.speaker:
            joinable = all([
                x in self.speaker.df.columns
                for x in ["file_name", "speaker_num"]
            ])
        
        if joinable:
            df = df.join(
                self.speaker.df, 
                on = ["file_name", "speaker_num"],
                how = "left"
            )

        return df

    def to_param_df(
            self, 
            output:Literal['param', 'log_param'] = "log_param"
        ) -> pl.DataFrame:
        """
        This will return a dataframe of the DCT parameters for all speakers.
        If `output` is passed `param`, it will be the DCT parameters in the
        original Hz. If passed `log_param`, it will be the DCT parameters
        over log(Hz).

        Args:
            output (Literal['param', 'log_param'], optional): 
                Which set of DCT parameters to return. Defaults to "log_param".

        Returns:
            (pl.DataFrame): A DataFrame of DCT parameters for all speakers.
        """
        df = pl.concat(
            [x.to_param_df(output = output) for x in self.values()]
        )
        joinable = False
        if self.speaker:
            joinable = all([
                x in self.speaker.df.columns
                for x in ["file_name", "speaker_num"]
            ])
        
        if joinable:
            df = df.join(
                self.speaker.df, 
                on = ["file_name", "speaker_num"],
                how = "left"
            )

        return df

    def to_point_df(self) -> pl.DataFrame:
        """
        This will return a DataFrame of point measurements
        for all speakers
        Returns:
            (pl.DataFrame): A DataFrame of vowel point measurements.
        """

        df = pl.concat(
            [x.to_point_df() for x in self.values()]
        )

        joinable = False
        if self.speaker:
            joinable = all([
                x in self.speaker.df.columns
                for x in ["file_name", "speaker_num"]
            ])
        
        if joinable:
            df = df.join(
                self.speaker.df, 
                on = ["file_name", "speaker_num"],
                how = "left"
            )

        return df        
