from fasttrackpy import CandidateTracks, OneTrack
from aligned_textgrid import AlignedTextGrid
from collections import defaultdict
import numpy as np

import warnings

class VowelClassCollection(defaultdict):
    def __init__(self, track_list:list):
        super().__init__(lambda : VowelClass())
        self.tracks_dict = defaultdict(lambda: [])
        self._make_tracks_dict(track_list)
        self._dictify()

    def __setitem__(self, __key, __value) -> None:
        super().__setitem__(__key, __value)

    def _make_tracks_dict(self, track_list):
        for v in track_list:
            self.tracks_dict[v.label].append(v)

    def _dictify(self):
        for v in self.tracks_dict:
            self[v] = VowelClass(v, self.tracks_dict[v])

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
    def winners(self):
        self._winners = [x.winner for x in self.tracks]
        return self._winners

    @property
    def winner_params(self):
        first_param = np.vstack(
            [
                x.parameters[:,0] 
                for x in self.winners
            ]
        ).T

        return first_param
    
    @property
    def params_means(self):
        winner_mean =  self.winner_params.mean(axis = 1)
        winner_mean = winner_mean[:, np.newaxis]
        return winner_mean
    
    @property
    def params_covs(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            param_cov = np.cov(self.winner_params)
        return param_cov
    
    @property
    def params_icov(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params_icov = np.linalg.inv(self.params_covs)
        return params_icov
    
class VowelMeasurement():
    def __init__(
            self, 
            track: CandidateTracks
        ):
        self.track = track
        self.label = track.label
        self.candidates = track.candidates
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
        first_param = np.vstack(
            [
                x.parameters[:,0] 
                for x in self.candidates
            ]
        ).T

        return first_param
    
    @property
    def cand_mahals(self):
        inv_covmat = self.vowel_class.params_icov
        x_mu = self.cand_params - self.vowel_class.params_means
        left = np.dot(x_mu.T, inv_covmat)
        mahal = np.dot(left, x_mu)
        return mahal.diagonal()
    



