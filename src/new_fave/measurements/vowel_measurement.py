from fasttrackpy import CandidateTracks, OneTrack
from aligned_textgrid import AlignedTextGrid
from fave_measurement_point.heuristic import Heuristic
from fave_measurement_point.formants import FormantArray
from collections import defaultdict
import numpy as np
from typing import Literal
import polars as pl

import scipy.stats as stats
from scipy.fft import idst, idct
from joblib import Parallel, delayed, cpu_count

NCPU = cpu_count()

import warnings

def blank():
    return VowelClass()

def blank_list():
    return []

def first_deriv(coefs, size = 100):
    hatu = coefs.copy()
    for i in range(hatu.size):
        hatu[i]=-(i)*hatu[i]
    hatu[:-1]=hatu[1:]
    hatu[-1]=0
    dotu=idst(hatu, n = size, type=2)
    return dotu.tolist()

class SpeakerCollection(defaultdict):
    def __init__(self, track_list:list):
        self.speakers_dict = defaultdict(blank_list)
        self._make_tracks_dict(track_list)
        self._dictify()
    
    def __setitem__(self, __key, __value) -> None:
        super().__setitem__(__key, __value)

    def _make_tracks_dict(self, track_list):
        for v in track_list:
            file_speaker = (v.file_name, v.group)
            self.speakers_dict[file_speaker].append(v)
        
    def _dictify(self):
        for fs in self.speakers_dict:
            self[fs] = VowelClassCollection(
                self.speakers_dict[fs]
            )
    
    def to_tracks_df(self):
        df = pl.concat(
            [x.to_tracks_df() for x in self.values()]
        )
        return df

    def to_param_df(
            self, 
            output:Literal['param', 'log_param'] = "log_param"
        ) -> pl.DataFrame:
        """_summary_

        Args:
            output (Literal[&#39;param&#39;, &#39;log_param&#39;], optional): _description_. Defaults to "log_param".

        Returns:
            pl.DataFrame: _description_
        """
        df = pl.concat(
            [x.to_param_df(output = output) for x in self.values()]
        )
        return df


    def to_point_df(self):
        df = pl.concat(
            [x.to_point_df() for x in self.values()]
        )
        return df


class VowelClassCollection(defaultdict):
    def __init__(self, track_list:list):

        super().__init__(blank)
        self.tracks_dict = defaultdict(blank_list)
        self._make_tracks_dict(track_list)
        self._dictify()
        self._vowel_system()
        self._ecdf = None
        self._params_means = None
        self._params_icov = None
        self._maximum_formant_means = None
        self._max_formant_icov = None


    def __setitem__(self, __key, __value) -> None:
        super().__setitem__(__key, __value)

    def _reset_winners(self):
        self._ecdf = None
        self._params_means = None
        self._params_icov = None
        self._maximum_formant_means = None
        self._max_formant_icov = None        

    def _make_tracks_dict(self, track_list):
        for v in track_list:
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
    
    @property
    def winners(self):
        return [
            x
            for vc in self.values()
            for x in vc.winners
        ]
    
    @property
    def vowel_measurements(self):
        return [
            x  
            for vc in self.values()
            for x in vc.tracks
        ]

    @property
    def winner_params(self):
        params = np.array(
            [
                x.parameters
                for x in self.winners
            ]
        ).T

        return params
    
    @property
    def winner_formants(self):
        formants = np.hstack(
            [
                x.formants
                for x in self.winners
            ]
        )

        return formants
    
    @property
    def winner_expanded_formants(self):
        formants = np.hstack(
            [
                x.expanded_formants[:, :, x.winner_index]
                for x in self.vowel_measurements
            ]
        )

        return formants
        
    @property
    def winners_maximum_formant(self):
        max_formants = np.array([[
            x.maximum_formant
            for x in self.winners
        ]])

        return max_formants

    
    @property
    def params_means(self):
        if self._params_means is not None:
            return self._params_means
        N = len(self.winners)
        winner_mean =  self.winner_params.reshape(-1, N).mean(axis = 1)
        winner_mean = winner_mean[:, np.newaxis]
        self._params_means = winner_mean
        return winner_mean
    
    @property
    def params_covs(self):
        N = len(self.winners)
        square_param = self.winner_params.reshape(-1, N)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            param_cov = np.cov(square_param)
        return param_cov
    
    @property
    def params_icov(self):
        if self._params_icov is not None:
            return self._params_icov
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                params_icov = np.linalg.inv(self.params_covs)
                self._params_icov = params_icov
                return params_icov
            except:
                params_icov = np.array([
                    [np.nan] * self.params_covs.size
                ]).reshape(
                    self.params_covs.shape[0],
                    self.params_covs.shape[1]
                )
                self._params_icov = params_icov
                return params_icov

    @property
    def maximum_formant_means(self):
        if self._maximum_formant_means is not None:
            return self._maximum_formant_means
        self._maximum_formant_means = self.winners_maximum_formant.mean()
        return self.winners_maximum_formant.mean()
    
    @property
    def maximum_formant_cov(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        
            cov = np.cov(self.winners_maximum_formant).reshape(1,1)
        return cov
    
    @property
    def max_formant_icov(self):
        if self._max_formant_icov is not None:
            return self._max_formant_icov
        
        try:
            icov = np.linalg.inv(self.maximum_formant_cov)
            self._max_formant_icov = icov
            return icov
        except:
            self._max_formant_icov = np.array([[np.nan]])
            return np.array([[np.nan]])    
                
    def to_tracks_df(self):
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
    
    def to_point_df(self):
        """Return a DataFrame of point measurements

        Returns:
            (pl.DataFrame): 
                A DataFrame of vowel point measures.
        """                
        df = pl.concat(
            [x.to_point_df() for x in self.values()]
        )

        return df

class VowelClass():
    def __init__(
            self,
            label: str,
            tracks: list
        ):
        self.label = label
        self.tracks = tracks
        self._winners = [x.winner for x in self.tracks]
        for t in self.tracks:
            t.vowel_class = self

    @property
    def vowel_system(self):
        return self._vowel_system
    
    @vowel_system.setter
    def vowel_system(self, vowel_system: VowelClassCollection):
        self._vowel_system = vowel_system

    @property
    def winners(self):
        self._winners = [x.winner for x in self.tracks]
        return self._winners
    
    @property
    def winner_params(self):
        params = np.array(
            [
                x.parameters
                for x in self.winners
            ]
        ).T

        return params
    
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
    
    
class VowelMeasurement():
    def __init__(
            self, 
            track: CandidateTracks,
            heuristic: Heuristic = Heuristic()
        ):
        self.track = track
        self.label = track.label
        self.candidates = track.candidates
        self.n_formants = track.n_formants
        self._winner = track.winner
        self.heuristic = heuristic
        self.interval = track.interval
        self.group = track.group
        self.id = track.id
        self.file_name = track.file_name
        self._expanded_formants = None

    @property
    def formant_array(self):
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
    def vowel_class(self, vclass: VowelClass):
        self._vclass = vclass

    @property
    def winner(self):
        return self._winner
    
    @winner.setter
    def winner(self, idx):
        self._winner = self.candidates[idx]
        self.vowel_class.vowel_system._reset_winners()
    
    @property
    def winner_index(self):
        return self.candidates.index(self.winner)

    @property
    def expanded_formants(self):
        if self._expanded_formants is not None:
            return self._expanded_formants
        
        self._expanded_formants = np.apply_along_axis(
            lambda x: idct(x.T, n = 20, orthogonalize=True, norm = "forward"),
            0,
            self.cand_params
        )
        return self._expanded_formants
        

    @property
    def cand_params(self):
        params = np.array(
            [
                x.parameters
                for x in self.candidates
            ]
        ).T

        return params

    @property
    def cand_max_formants(self):
        return np.array([[
            c.maximum_formant
            for c in self.candidates
        ]])
    
    @property
    def cand_errors(self):
         with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.array([
                c.smooth_error
                for c in self.candidates
            ])

    @property
    def cand_mahals(self):
        N = len(self.candidates)
        square_params = self.cand_params.reshape(-1, N)
        inv_covmat = self.vowel_class.vowel_system.params_icov
        param_means = self.vowel_class.vowel_system.params_means
        x_mu = square_params - param_means
        left = np.dot(x_mu.T, inv_covmat)
        mahal = np.dot(left, x_mu)
        return mahal.diagonal()
    
    @property
    def cand_mahal_log_prob(self):
        winner_shape = self.cand_params.shape
        df = winner_shape[0] * winner_shape[1]
        log_prob = stats.chi2.logsf(
            self.cand_mahals,
            df = df
        )
        if np.isfinite(log_prob).mean() < 0.5:
            log_prob = np.zeros(shape = log_prob.shape)
        return log_prob

    
    @property 
    def max_formant_mahal(self):
        inv_covmat = self.vowel_class.vowel_system.max_formant_icov
        maximum_formant_means = self.vowel_class.vowel_system.maximum_formant_means
        x_mu = self.cand_max_formants - maximum_formant_means
        left = np.dot(x_mu.T, inv_covmat)
        mahal = np.dot(left, x_mu)
        return mahal.diagonal()
    
    @property
    def max_formant_log_prob(self):
        log_prob = stats.chi2.logsf(
            self.max_formant_mahal,
            df = 1
        )

        if np.isfinite(log_prob).mean() < 0.5:
            log_prob = np.zeros(shape = log_prob.shape)

        return log_prob * 0.5

    @property
    def error_log_prob(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            err_norm = self.cand_errors - np.nanmin(self.cand_errors)
            err_surv = 1 - (err_norm/np.nanmax(err_norm))
            err_log_prob = np.log(err_surv)
        return err_log_prob

    @property
    def point_measure(self):
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
    
    @property
    def vm_context(self):
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
            )
        )

        df = df.join(self.vm_context, on = "id")

        return(df)
