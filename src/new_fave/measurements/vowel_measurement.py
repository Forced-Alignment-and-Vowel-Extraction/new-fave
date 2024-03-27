from fasttrackpy import CandidateTracks, OneTrack
from aligned_textgrid import AlignedTextGrid
from collections import defaultdict
import numpy as np

import warnings

class VowelClassCollection(defaultdict):
    def __init__(self, track_list:list[CandidateTracks]):
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

    @property
    def winners(self):
        return self._winners
    
    @winners.setter
    def winners(self, idces):
        self._winners = [
            t.candidates[idx] 
            for t, idx in zip(self.tracks, idces)
        ]

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
        return self.winner_params.mean(axis = 1)
    
    @property
    def params_covs(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            param_cov = np.cov(self.winner_params)
        return param_cov
    
class VowelMeasurement():
    def __init__(
            self, 
            track: CandidateTracks
        ):
        self.track = track

    @property
    def vowel_class(self):
        if self._vclass:
            return self._vclass
    
    @vowel_class.setter
    def vowel_class(self, vclass: VowelClass):
        self._vclass = vclass

    


