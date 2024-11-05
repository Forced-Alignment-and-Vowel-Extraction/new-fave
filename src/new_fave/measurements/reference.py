import polars as pl
import polars.selectors as cs
import numpy as np
from pathlib import Path
import logging
from new_fave.measurements.calcs import mahalanobis
from tqdm import tqdm
import warnings

logger = logging.getLogger("reference")
logger.setLevel(level=logging.INFO)


class ReferenceValues:
    """
    A class to represent a reference set of vowel measurements
    
    Args:
        logparam_corpus (str | Path, optional):
            Path to logparam files. Defaults to None.
        param_corpus (str | Path, optional):
            Path to param files. Defaults to None.
        points_corpus (str | Path, optional):
            Path to points files. Defaults to None.

    """
    reference_type = None
    mean_dict = None
    icov_dict = None

    def __init__(
            self,
            logparam_corpus: str|Path = None,
            param_corpus: str|Path = None,
            points_corpus: str|Path = None
        ):
    
        provided_corpora = [c for c in [logparam_corpus, param_corpus, points_corpus] if c is not None]
        if len(provided_corpora) > 1:
            logger.warning(
                (
                    "Multiple reference value corpora were provided, but "
                    "only one kind of reference value corpus will be used. "
                    "The preference order of references are: "
                    "logparam > param > points."
                )
            )

        if points_corpus:
            points_corpus = Path(points_corpus).glob("*_points.csv")
            logparam_corpus = None
            param_corpus = None

            self.reference_type = "points"
            self._process_points(points_corpus)

        if param_corpus:
            param_corpus = Path(param_corpus).glob("*_param.csv")
            logparam_corpus = None
            points_corpus = None
            
            self.reference_type = "param"
            self._process_param(param_corpus)            

        if logparam_corpus:
            logparam_corpus = Path(logparam_corpus).glob("*_logparam.csv")
            param_corpus = None
            points_corpus = None

            self.reference_type = "logparam"
            self._process_param(logparam_corpus)

    def _process_param(self, param_corpus):
        df = pl.concat(
            (pl.scan_csv(f) for f in param_corpus),
            how="diagonal"
        )

        gdf = (
            df
            .select(
                ["id", "file_name", "group", "label","F1", "F2","F3", "param"]
            )
            .melt(
                id_vars=["id", "file_name", "group", "label", "param"]
            )
            .collect()
            .pivot(
                columns=["param", "variable"],
                index = ["id", "file_name", "label"],
                values="value",
                sort_columns = True,
                separator="_"
            )
            .drop(["id", "file_name", "group"])
            .group_by(["label"])
        )

        self._make_dicts(gdf)

    def _process_points(self, points_corpus):
        df = pl.concat(
            (pl.scan_csv(f) for f in points_corpus),
            how="diagonal"
        )

        gdf = (
            df
            .select(
                "label",
                cs.matches("[FB][0-9]")
            )
            .with_columns(
                cs.float().mul(1/np.sqrt(2))
            )
            .collect()
            .group_by(["label"])
        )

        self._make_dicts(gdf)

    def _make_dicts(self, gdf):
        lab_dict = {
            k[0]:d.drop("label").to_numpy().T for k,d in gdf
        }

        mean_dict = {
            k : np.expand_dims(v.mean(axis = 1),1) for k,v in lab_dict.items()
        }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            icov_dict = {}
            for k,v in lab_dict.items():
                try:
                    cov_mat = np.linalg.inv(np.cov(v))
                except:
                    cov_mat = np.empty((v.shape[0], v.shape[0]))
                icov_dict[k] = cov_mat


        new_lab_dict = dict()
        logger.info("Processing Reference Values")
        for k in tqdm(lab_dict):
            mahals = mahalanobis(
                lab_dict[k],
                mean_dict[k],
                icov_dict[k]
            )
            if (mahals < 5).sum() <= 10:
                new_lab_dict[k] = lab_dict[k]
            else:
                new_lab_dict[k] = np.copy(lab_dict[k].T[mahals < 5].T)

        self.mean_dict = {
            k : np.expand_dims(v.mean(axis = 1),1) for k,v in new_lab_dict.items()
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            icov_dict = {}
            for k,v in new_lab_dict.items():
                try:
                    cov_mat = np.linalg.inv(np.cov(v))
                except:
                    cov_mat = np.empty((v.shape[0], v.shape[0]))
                icov_dict[k] = cov_mat
            self.icov_dict = icov_dict