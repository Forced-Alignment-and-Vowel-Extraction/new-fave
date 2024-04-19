from fasttrackpy import CandidateTracks, OneTrack
from aligned_textgrid import AlignedTextGrid
from fave_measurement_point.heuristic import Heuristic
from fave_measurement_point.formants import FormantArray
from collections import defaultdict
import numpy as np

import polars as pl

import scipy.stats as stats
from scipy.fft import idst
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


class VowelClassCollection(defaultdict):
    def __init__(self, track_list:list, param_optim = 3):

        super().__init__(blank)
        self.param_optim = param_optim
        self.tracks_dict = defaultdict(blank_list)
        self._make_tracks_dict(track_list)
        self._dictify()
        self._vowel_system()
        self._kernel = None
        self._ecdf = None

    def __setitem__(self, __key, __value) -> None:
        super().__setitem__(__key, __value)

    def _make_tracks_dict(self, track_list):
        for v in track_list:
            self.tracks_dict[v.label].append(v)

    def _dictify(self):
        for v in self.tracks_dict:
            self[v] = VowelClass(
                v, 
                self.tracks_dict[v], 
                param_optim=self.param_optim)

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
                x.parameters[:, 0:self.param_optim]
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
    def winners_max_rates(self):
        rates = np.hstack([vc.winners_max_rates for vc in self.values()])
        return rates
    
    @property
    def rate_ecdfs(self):
        if not self._ecdf:
            ecdfs = [ 
                stats.ecdf(self.winners_max_rates[i,:])
                for i in range(self.winners_max_rates.shape[0])
            ]
            self._ecdf = ecdfs

        return self._ecdf

    
    @property
    def kernel(self):
        if not self._kernel:
            kernel = stats.gaussian_kde(self.winner_formants,bw_method=2)
            return kernel
        
        return self._kernel
    
    @property
    def winners_maximum_formant(self):
        max_formants = np.array([[
            x.maximum_formant
            for x in self.winners
        ]])

        return max_formants

    
    @property
    def params_means(self):
        N = len(self.winners)
        winner_mean =  self.winner_params.reshape(-1, N).mean(axis = 1)
        winner_mean = winner_mean[:, np.newaxis]
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                params_icov = np.linalg.inv(self.params_covs)
                return params_icov
            except:
                params_icov = np.array([
                    [np.nan] * self.params_covs.size
                ]).reshape(
                    self.params_covs.shape[0],
                    self.params_covs.shape[1]
                )
                return params_icov

    @property
    def maximum_formant_means(self):
        return self.winners_maximum_formant.mean()
    
    @property
    def maximum_formant_cov(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        
            cov = np.cov(self.winners_maximum_formant).reshape(1,1)
        return cov
    
    @property
    def max_formant_icov(self):
        try:
            icov = np.linalg.inv(self.maximum_formant_cov)
            return icov
        except:
            return np.array([[np.nan]])    
                
    def to_tracks_df(self):
        df = pl.concat(
            [x.to_tracks_df() for x in self.values()]
        )

        return df
    
    def to_point_df(self):
        df = pl.concat(
            [x.to_point_df() for x in self.values()]
        )

        return df

class VowelClass():
    def __init__(
            self,
            label: str,
            tracks: list,
            param_optim = 3
        ):
        self.label = label
        self.tracks = tracks
        self._winners = [x.winner for x in self.tracks]
        self.param_optim = param_optim
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
                x.parameters[:, 0:self.param_optim]
                for x in self.winners
            ]
        ).T

        return params
    
    @property
    def winners_maximum_formant(self):
        max_formants = np.array([[
            x.maximum_formant
            for x in self.winners
        ]])

        return max_formants

    @property
    def winners_max_rates(self):
        rates = [cand.rates[:, :, cand.winner_index] for cand in self.tracks]
        max_rate = np.array([
            (rate**2).max(axis = 0) 
            for rate in rates
        ]).T

        return max_rate


    @property
    def params_means(self):
        N = len(self.tracks)
        winner_mean =  self.winner_params.reshape(-1, N).mean(axis = 1)
        winner_mean = winner_mean[:, np.newaxis]
        return winner_mean
    
    @property
    def params_covs(self):
        N = len(self.tracks)
        square_param = self.winner_params.reshape(-1, N)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            param_cov = np.cov(square_param)
        return param_cov
    
    @property
    def params_icov(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                params_icov = np.linalg.inv(self.params_covs)
                return params_icov
            except:
                params_icov = np.array([
                    [np.nan] * self.params_covs.size
                ]).reshape(
                    self.params_covs.shape[0],
                    self.params_covs.shape[1]
                )
                return params_icov

    @property
    def maximum_formant_means(self):
        return self.winners_maximum_formant.mean()
    
    @property
    def maximum_formant_cov(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        
            cov = np.cov(self.winners_maximum_formant).reshape(1,1)
        return cov
    
    @property
    def max_formant_icov(self):
        try:
            icov = np.linalg.inv(self.maximum_formant_cov)
            return icov
        except:
            return np.array([[np.nan]])
        
    def to_tracks_df(self):
        df = pl.concat(
            [x.to_tracks_df() for x in self.tracks]
        )

        return df
    
    def to_point_df(self):
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
        self.vowel_class.vowel_system._kernel = None
        self.vowel_class.vowel_system._ecdf = None
    
    @property
    def winner_index(self):
        return self.candidates.index(self.winner)

    @property
    def cand_params(self):
        params = np.array(
            [
                x.parameters[:, 0:self.vowel_class.param_optim]
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
        inv_covmat = self.vowel_class.params_icov
        param_means = self.vowel_class.params_means
        # if np.any(~np.isfinite(inv_covmat)):
        #     inv_covmat = self.vowel_class.vowel_system.params_icov
        #     param_means = self.vowel_class.vowel_system.params_means
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
        if np.any(~np.isfinite(log_prob)):
            log_prob = np.zeros(shape = log_prob.shape)
        return log_prob

    
    @property 
    def max_formant_mahal(self):
        inv_covmat = self.vowel_class.max_formant_icov
        maximum_formant_means = self.vowel_class.maximum_formant_means
        if np.any(~np.isfinite(inv_covmat)):
            inv_covmat = self.vowel_class.vowel_system.max_formant_icov
            maximum_formant_means = self.vowel_class.vowel_system.maximum_formant_means
        elif len(self.vowel_class.tracks) < 5:
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
        return log_prob

    @property
    def error_log_prob(self):
        err_ecdf = stats.ecdf(self.cand_errors)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            err_log_prob = np.log(err_ecdf.sf.evaluate(self.cand_errors))
        return err_log_prob
    
    @property
    def rates(self):
        N = self.formant_array.time.size
        rates =  np.apply_along_axis(
            first_deriv, 
            arr = self.cand_params, 
            axis = 0, 
            **{"size": N}
        )[:,0:2]
        return rates
    
    @property
    def rate_log_prob(self):
        rates = self.rates
        max_rate = (rates**2).max(axis = 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = np.array([
                self.vowel_class.vowel_system.rate_ecdfs[i].sf.evaluate(
                    max_rate[i,:]
                )
                for i in range(2)
            ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_prob =  np.log(prob).sum(axis = 0)

        return log_prob - log_prob.max()
    
    @property
    def cand_log_kde(self):
        kernel = self.vowel_class.vowel_system.kernel
        formants = [cand.formants.mean(axis = 1)
                         for cand in self.candidates]
        log_kde = Parallel(n_jobs=NCPU)(
            delayed(kernel.logpdf)(formant) for formant in formants
        )

        log_kde_sum = np.array([
            kde.sum()
            for kde in log_kde
        ])
        

        return (log_kde_sum - log_kde_sum.max())*2

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
            "pre_word": pre_word,
            "fol_word": fol_word,
            "pre_seg": pre_seg,
            "fol_seg": fol_seg,
            "abs_pre_seg": abs_pre_seg,
            "abs_fol_seg": abs_fol_seg,
            "context": context
        })

        return df
    
    def to_tracks_df(self):
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
    
    def to_point_df(self):
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
