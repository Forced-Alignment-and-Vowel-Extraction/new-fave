from fasttrackpy import CandidateTracks, OneTrack
from aligned_textgrid import AlignedTextGrid
from collections import defaultdict
import numpy as np

import scipy.stats as stats

import warnings

class VowelClassCollection(defaultdict):
    def __init__(self, track_list:list, param_optim = 3):
        super().__init__(lambda : VowelClass())
        self.param_optim = param_optim
        self.tracks_dict = defaultdict(lambda: [])
        self._make_tracks_dict(track_list)
        self._dictify()
        self._vowel_system()
        

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
    def vspace_params(self):
        return np.concatenate([
            self[v].winner_params
            for v in self
        ],
        axis=2)
    
    @property
    def vspace_param_mean(self):
        return self.vspace_params.mean(axis = (1,2))
    
    @property
    def vspace_param_std(self):
        return self.vspace_params.std(axis = (1,2))

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
    def winner_param_norm(self):
        winner_params = self.winner_params
        centers = self.vowel_system.vspace_param_mean[
            :, 
            np.newaxis, 
            np.newaxis
            ]
        scales = self.vowel_system.vspace_param_std[
            :, 
            np.newaxis, 
            np.newaxis
            ]        
        
        normed_params = (winner_params - centers)/scales
        return normed_params

    @property
    def winners_maximum_formant(self):
        max_formants = np.array([[
            x.maximum_formant
            for x in self.winners
        ]])

        return max_formants

    
    @property
    def params_means(self):
        N = len(self.tracks)
        winner_mean =  self.winner_param_norm.reshape(-1, N).mean(axis = 1)
        winner_mean = winner_mean[:, np.newaxis]
        return winner_mean
    
    @property
    def params_covs(self):
        N = len(self.tracks)
        square_param = self.winner_param_norm.reshape(-1, N)
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
    
    
class VowelMeasurement():
    def __init__(
            self, 
            track: CandidateTracks
        ):
        self.track = track
        self.label = track.label
        self.candidates = track.candidates
        self.n_formants = track.n_formants
        self._winner = track.winner

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
    def cand_param_norm(self):
        params = self.cand_params
        centers = self.vowel_class.vowel_system.vspace_param_mean[
            :, 
            np.newaxis, 
            np.newaxis
            ]
        scales = self.vowel_class.vowel_system.vspace_param_std[
            :, 
            np.newaxis, 
            np.newaxis
            ]
        param_norm = (params - centers)/scales
        return param_norm
      
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
        square_params = self.cand_param_norm.reshape(-1, N)
        inv_covmat = self.vowel_class.params_icov
        x_mu = square_params - self.vowel_class.params_means
        left = np.dot(x_mu.T, inv_covmat)
        mahal = np.dot(left, x_mu)
        return mahal.diagonal()
    
    @property
    def cand_mahal_log_prob(self):
        winner_shape = self.cand_param_norm.shape
        df = winner_shape[0] * winner_shape[1]
        log_prob = stats.chi2.logsf(
            self.cand_mahals,
            df = df
        )
        return log_prob

    
    @property 
    def max_formant_mahal(self):
        inv_covmat = self.vowel_class.max_formant_icov
        x_mu = self.cand_max_formants - \
            self.vowel_class.maximum_formant_means
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