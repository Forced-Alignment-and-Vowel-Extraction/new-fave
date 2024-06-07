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
from new_fave.measurements.calcs import mahalanobis, \
    mahal_log_prob,\
    param_to_cov,\
    cov_to_icov,\
    clear_cached_properties

from collections import defaultdict
import numpy as np
from typing import Literal
import polars as pl

import scipy.stats as stats
from scipy.fft import idst, idct
from joblib import Parallel, delayed, cpu_count

from collections.abc import Sequence, Iterable
from dataclasses import dataclass, field
from nptyping import NDArray, Shape, Float

from functools import lru_cache, cached_property

NCPU = cpu_count()

import warnings

def blank():
    return VowelClass()

def blank_list():
    return []

EMPTY_LIST = blank_list()

class StatPropertyMixins:

    @cached_property
    def winner_param(
        self
    ) -> NDArray[Shape["Param, Formant, N"], Float]:
        params = np.array(
            [
                x.parameters
                for x in self.winners
            ]
        ).T
        return params
    
    @cached_property
    def winner_maxformant(
        self
    ) -> NDArray[Shape["1, N"], Float]:
        max_formants = np.array([[
            x.maximum_formant
            for x in self.winners
        ]])

        return max_formants

    @cached_property
    def winner_param_mean(
        self
    ) -> NDArray[Shape["ParamFormant, 1"], Float]:
        N = len(self.winners)
        winner_mean =  self.winner_param.reshape(-1, N).mean(axis = 1)
        winner_mean = winner_mean[:, np.newaxis]
        return winner_mean
    
    @cached_property
    def winner_param_cov(
        self
    ) -> NDArray[Shape["ParamFormant, ParamFormant"], Float]:
        param_cov = param_to_cov(self.winner_param)
        return param_cov
    
    @cached_property
    def winner_param_icov(
        self
    ) ->  NDArray[Shape["ParamFormant, ParamFormant"], Float]:
        params_icov = cov_to_icov(self.winner_param_cov)
        return params_icov    
    
    @cached_property
    def winner_maxformant_mean(
        self
    ) -> float:
        return self.winner_maxformant.mean()
    
    @cached_property
    def winner_maxformant_cov(
        self
    ) -> NDArray[Shape["1, 1"], Float]:
        cov = param_to_cov(self.winner_maxformant)
        cov = cov.reshape(1,1)
        return cov
    
    @cached_property
    def winner_maxformant_icov(
        self
    ) -> NDArray[Shape["1, 1"], Float]:        
        icov = cov_to_icov(self.winner_maxformant_cov)
        return icov


@dataclass
class VowelMeasurement(Sequence):
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
    def __post_init__(
            self
        ):
        super().__init__()
        #self.label = self.track.label
        self.candidates = self.track.candidates
        self.n_formants = self.track.n_formants
        self._winner = self.track.winner
        self.interval = self.track.interval
        self.group = self.track.group
        self.id = self.track.id
        self.file_name = self.track.file_name
        self._label = None
        self._expanded_formants = None
        self._optimized = 0

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
        self.interval.label = x
            

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
    def vowel_class(self, vclass):
        self._vclass = vclass

    @property
    def winner(self)->OneTrack:
        return self._winner
    
    @winner.setter
    def winner(self, idx):
        self._winner = self.candidates[idx]
        self.vowel_class.vowel_system._reset_winners()
        self.vowel_class._reset_winners()
        self._expanded_formants = None
        self._optimized += 1

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
        

    @property
    def cand_param(
        self
    ) -> NDArray[Shape["Param, Formant, Cand"], Float]:
        params = np.array(
            [
                x.parameters
                for x in self.candidates
            ]
        ).T

        return params

    @property
    def cand_maxformant(
        self
    ) -> NDArray[Shape["1, Cand"], Float]:
        return np.array([[
            c.maximum_formant
            for c in self.candidates
        ]])
    
    @property
    def cand_error(
        self
    ) -> NDArray[Shape["Cand"], Float]:
         with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.array([
                c.smooth_error
                for c in self.candidates
            ])

    @property
    def cand_param_mahal_speaker_global(
        self
    ) -> NDArray[Shape["Cand"], Float]:
        N = len(self.candidates)
        square_params = self.cand_param.reshape(-1, N)
        mahal = mahalanobis(
            square_params,
            self.vowel_class.vowel_system.winner_param_mean,
            self.vowel_class.vowel_system.winner_param_icov
        )
        return mahal
    
    @property
    def cand_param_logprob_speaker_global(
        self
    ) -> NDArray[Shape["Cand"], Float]:
        log_prob  = mahal_log_prob(
            self.cand_param_mahal_speaker_global,
            self.cand_param
        )
        return log_prob
    
    @property
    def cand_param_mahal_speaker_byvclass(
        self
    ) -> NDArray[Shape["Cand"], Float]:
        N = len(self.candidates)
        square_params = self.cand_param.reshape(-1, N)
        inv_covmat = self.vowel_class.winner_param_icov
        param_means = self.vowel_class.winner_param_mean
        mahal = mahalanobis(square_params, param_means, inv_covmat)
        return mahal
    
    @property
    def cand_param_logprob_speaker_byvclass(
        self
    ) -> NDArray[Shape["Cand"], Float]:
        log_prob = mahal_log_prob(
            self.cand_param_mahal_speaker_byvclass,
            self.cand_param
        )
        if len(self.vowel_class) < 10:
            log_prob = np.zeros(shape = log_prob.shape)
        return log_prob
    

    @property
    def cand_param_mahal_corpus_byvowel(
        self
    ) -> NDArray[Shape["Cand"], Float]:
        
        N = len(self.candidates)
        square_params = self.cand_param.reshape(-1, N)
        inv_covmat = self\
            .vowel_class\
            .vowel_system\
            .corpus\
            .winner_param_icov[self.label]
        param_means = self\
            .vowel_class\
            .vowel_system\
            .corpus\
            .winner_param_mean[self.label]
        
        mahal = mahalanobis(square_params, param_means, inv_covmat)
        return mahal

    @property
    def cand_param_logprob_corpus_byvowel(
        self
    ) -> NDArray[Shape["Cand"], Float]:
        log_prob = mahal_log_prob(
            self.cand_param_mahal_corpus_byvowel,
            self.cand_param
        )
        return log_prob        

    @property 
    def cand_maxformant_mahal_speaker_global(
        self
    ) -> NDArray[Shape["Cand"], Float]:
        inv_covmat = self.vowel_class.vowel_system.winner_maxformant_icov
        maximum_formant_means = self.vowel_class.vowel_system.winner_maxformant_mean        
        mahal = mahalanobis(self.cand_maxformant, maximum_formant_means, inv_covmat)
        return mahal
    
    @property
    def cand_maxformant_logprob_speaker_global(
        self
    ) -> NDArray[Shape["Cand"], Float]:
        log_prob = mahal_log_prob(
            self.cand_maxformant_mahal_speaker_global,
            self.cand_maxformant
        )

        return log_prob

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
        point_dict["max_formant"] = self.winner.maximum_formant
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
class VowelClass(Sequence, StatPropertyMixins):
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
        tracks (list[VowelMeasurement]): 
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
    tracks: list[VowelMeasurement] = field(default_factory= lambda : [])
    def __post_init__(self):
        super().__init__()
        self._winners = [x.winner for x in self.tracks]
        self._winner_param = None
        self._winner_param_mean = None
        self._winner_param_cov = None
        self._winner_param_icov = None
        for t in self.tracks:
            t.vowel_class = self

    def __getitem__(self, i):
        return self.tracks[i]
    
    def __len__(self):
        return len(self.tracks)
    
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
    def vowel_system(self, vowel_system):
        self._vowel_system = vowel_system

    @cached_property
    def winners(self):
        return [x.winner for x in self.tracks]
    
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
            [x.to_param_df(output=output) for x in self.tracks]
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
            [x.to_tracks_df() for x in self.tracks]
        )

        return df    
    
    def to_point_df(self) -> pl.DataFrame:
        """Return a DataFrame of point measurements

        Returns:
            (pl.DataFrame): 
                A DataFrame of vowel point measures.
        """        
        df = pl.concat(
            [x.to_point_df() for x in self.tracks]
        )

        return df

class VowelClassCollection(defaultdict, StatPropertyMixins):
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
    def __init__(self, track_list:list[VowelMeasurement] = EMPTY_LIST):
        super().__init__(blank)
        self.track_list = track_list
        self.tracks_dict = defaultdict(blank_list)
        if isinstance(self.track_list, Iterable):
            self._make_tracks_dict()
            self._dictify()
        self._vowel_system()
        self._file_name = None
        self._corpus = None


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
    
    @property
    def vowel_measurements(
        self
    ) -> list[VowelMeasurement]:
        return [
            x  
            for vc in self.values()
            for x in vc.tracks
        ]
    
    @cached_property
    def textgrid(
        self
    ) -> AlignedTextGrid:
        return get_textgrid(self.vowel_measurements[0].interval)
    
    @property
    def file_name(
        self
    ) -> str:
        if self._file_name:
            return self._file_name
        
        self._file_name = self.vowel_measurements[0].winner.file_name
        return self._file_name
    
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
    
class SpeakerCollection(defaultdict, StatPropertyMixins):
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

    def __init__(self, track_list:list[VowelMeasurement] = []):
        self.track_list = track_list
        self.speakers_dict = defaultdict(blank_list)
        self._make_tracks_dict()
        self._dictify()
        self._speaker = None
        self._associate_corpus()
    
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
