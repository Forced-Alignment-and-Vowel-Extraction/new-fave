from fasttrackpy import CandidateTracks, OneTrack
from aligned_textgrid import AlignedTextGrid
from fave_measurement_point.heuristic import Heuristic
from fave_measurement_point.formants import FormantArray
from new_fave.utils.textgrid import get_textgrid
from collections import defaultdict
import numpy as np
from typing import Literal
import polars as pl

import scipy.stats as stats
from scipy.fft import idst, idct
from joblib import Parallel, delayed, cpu_count

from collections.abc import Sequence

NCPU = cpu_count()

import warnings

def blank():
    return VowelClass()

def blank_list():
    return []

# def first_deriv(coefs, size = 100):
#     hatu = coefs.copy()
#     for i in range(hatu.size):
#         hatu[i]=-(i)*hatu[i]
#     hatu[:-1]=hatu[1:]
#     hatu[-1]=0
#     dotu=idst(hatu, n = size, type=2)
#     return dotu.tolist()

class SpeakerCollection(defaultdict):
    """
    A class to represent the vowel system of all 
    speakers in a TextGrid. It is a subclass of `defaultdict`,
    and can be keyed by the `(file_name, group_name)` tuple.

    Args:
        track_list (list[CandidateTracks]):
            A list of `fasttrackpy.CandidateTrack`s.
    """
    def __init__(self, track_list:list[CandidateTracks]):
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
        return df


class VowelClassCollection(defaultdict):
    """
    A class for an entire vowel system. It is a subclass
    of `defaultdict`, so it can be keyed by vowel class 
    label

    Args:
        track_list (list[CandidateTracks):
            A list of `fasttrackpy.CandidateTrack`s.

    Attributes:
        maximum_formant_cov (np.array): 
            The covariance matrix for the winners maximum formant
            across the entire vowel system
        maximum_formant_means (np.array): 
            The mean maximum formant for the winners
            across the entire vowel system
        max_formant_icov (np.array): 
            The inverse covariance matrix for the winners maximum formant
            across the entire vowel system
        params_covs (np.array): 
            The covariance matrix for the winners' DCT
            parameters.
        params_icov (np.array): 
            The inverse covariance matrix for the winners' 
            DCT parameters.
        params_means (np.array): 
            An `np.array` for the winners' DCT parameters
            in the entire vowel system.
        vowel_measurements (list[VowelMeasurement]): 
            A list of all vowel measurements in the 
            vowel system
        winner_formants (np.array): 
            An `np.array` for the formants 
            for the winners in the entire vowel system.
        winner_params (np.array): 
            An `np.array` of DCT parameters for
            the winners in entire vowel system.
        winners (list[fasttrackpy.OneTrack]): 
            The winning `fasttrackpy.OneTrack` for 
            the entire vowel system
        winners_maximum_formant (np.array): 
            An `np.array` of the maximum formants
            for the winners in the entire vowel system           
    """
    def __init__(self, track_list:list[CandidateTracks]):

        super().__init__(blank)
        self.tracks_dict = defaultdict(blank_list)
        self._make_tracks_dict(track_list)
        self._dictify()
        self._vowel_system()
        self._params_means = None
        self._params_icov = None
        self._maximum_formant_means = None
        self._max_formant_icov = None
        self._textgrid = None
        self._file_name = None


    def __setitem__(self, __key, __value) -> None:
        super().__setitem__(__key, __value)

    def _reset_winners(self):
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
    def textgrid(self):
        if self._textgrid:
            return self._textgrid
        
        self._textgrid = get_textgrid(self.vowel_measurements[0].interval)
        return self._textgrid
    
    @property
    def file_name(self):
        if self._file_name:
            return self._file_name
        
        self._file_name = self.vowel_measurements[0].winner.file_name
        return self._file_name

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

class VowelClass(Sequence):
    """A class used to represent a vowel class.

    Args:
        label (str):
            The vowel class label
        tracks (list): 
            A list of VowelMeasurements
    Attributes:
        label (str): 
            label of the vowel class
        tracks (list): 
           A list of `VowelMeasurement`s
        vowel_system (VowelClassCollection):
            A the containing vowel system
        winners: 
            A list of winner OneTracks from
            the vowel class
        winner_params:
            An `np.array` of winner DCT parameters
            from the vowel class.
    """
    def __init__(
            self,
            label: str,
            tracks: list
        ):
        super().__init__()
        self.label = label
        self.tracks = tracks
        self._winners = [x.winner for x in self.tracks]
        for t in self.tracks:
            t.vowel_class = self

    def __getitem__(self, i):
        return self.tracks[i]
    
    def __len__(self):
        return len(self.tracks)
    
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
    
    
class VowelMeasurement(Sequence):
    """ A class used to represent a vowel measurment.

    Args:
        track (fasttrackpy.CandidateTracks): 
            A fasttrackpy.CandidateTrracks object
        heuristic (Heuristic, optional): 
            A point measurement Heuristic to use. 
            Defaults to Heuristic().
    
    Attributes: 
        track (CandidateTracks):
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
        interval (object):
            interval of the track
        label (str):
            label of the track
        n_formants (int):
            number of formants in the track


        winner: fasttrackpy.OneTrack
            The winning formant track
        winner_index (int):
            The index of the winning formant track

        error_log_prob (np.array):
            A conversion of the log-mean-squared-error to a 
            log-probabilities, based on an empirical cumulative
            density function.
        cand_errors (np.array):
            A numpy array of the log-mean-squared-errors
            for each candidate track.
        cand_mahals (np.array):
            The mahalanobis distance across DCT parameters
            for each candidate from the vowel system 
            distribution.
        cand_mahal_log_prob (np.array):
            A conversion of `cand_mahals` to log-probabilies.
        cand_max_formants (np.array):
            A numpy array of the maximum formants for
            this VowelMeasurement
        cand_params (np.array):
            A numpy array of the candidate track 
            DCT parameters.                
        max_formant_log_prob (np.array):
            A conversion of `max_formant_mahal` to log-probabilities.            
        max_formant_mahal (np.array):
            The mahalanobis distance of each
            maximum formant to the speaker's entire
            distribution.

        point_measure (pl.DataFrame):
            A polars dataframe of the point measurement for this vowel.
        vm_context (pl.DataFrame):
            A polars dataframe of contextual information for the vowel measurement.
    """
    def __init__(
            self, 
            track: CandidateTracks,
            heuristic: Heuristic = Heuristic()
        ):
        super().__init__()
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

    def __getitem__(self,i):
        return self.candidates[i]
    
    def __len__(self):
        return len(self.candidates)

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
        self._expanded_formants = None
    
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
            )
        )

        df = df.join(self.vm_context, on = "id")

        return(df)
