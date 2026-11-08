"""Baseline model.

The model works on the *candidate* table produced by utils.build_candidates
(one row per identifier mention + its context) and returns a prediction table
with columns article_id, dataset_id, type (one row per predicted citation).

    B0  RegexMajorityBaseline   - keep dataset-like candidates, predict the
                                  majority type of their kind (DOI / Accession)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .utils import attach_gold, f1_tuples

PRED_COLS = ["article_id", "dataset_id", "type"]


class BaseModel:
    name = "base"

    def fit(self, cands: pd.DataFrame, gold: pd.DataFrame) -> "BaseModel":
        return self

    def predict(self, cands: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Shared filter: which candidates look like datasets at all?
# --------------------------------------------------------------------------- #
class DatasetFilterMixin:
    """Learns from training labels which DOI prefixes (repositories) and
    accession families actually occur as datasets, and drops the rest.

    Most DOIs in a paper point to other papers, so this filter is what keeps
    precision from collapsing.
    """

    min_precision = 0.05  # keep a DOI prefix only if >=5% of its mentions are datasets
    drop_references_doi = False

    def _fit_filter(self, cands: pd.DataFrame, gold: pd.DataFrame) -> None:
        lab = attach_gold(cands, gold).drop_duplicates(["article_id", "dataset_id"])
        lab["is_ds"] = lab["label"].ne("None")
        stats = lab.groupby(["kind", "pattern"])["is_ds"].agg(["mean", "sum"])
        keep = stats[(stats["mean"] >= self.min_precision) & (stats["sum"] > 0)]
        self.keep_patterns_ = set(keep.index)

    def _filter(self, cands: pd.DataFrame) -> pd.DataFrame:
        mask = [(k, p) in self.keep_patterns_ for k, p in zip(cands["kind"], cands["pattern"])]
        out = cands[mask]
        if self.drop_references_doi:
            out = out[~((out["kind"] == "DOI") & out["in_references"])]
        return out


# --------------------------------------------------------------------------- #
# B0 - regex + majority class
# --------------------------------------------------------------------------- #
class RegexMajorityBaseline(DatasetFilterMixin, BaseModel):
    name = "B0_regex_majority"

    def fit(self, cands, gold):
        self._fit_filter(cands, gold)
        g = gold.copy()
        g["kind"] = np.where(g["dataset_id"].str.contains(r"10\.\d{4,9}/"), "DOI", "Accession")
        self.majority_ = g.groupby("kind")["type"].agg(lambda s: s.value_counts().idxmax()).to_dict()
        return self

    def predict(self, cands):
        c = self._filter(cands).drop_duplicates(["article_id", "dataset_id"]).copy()
        c["type"] = c["kind"].map(self.majority_).fillna("Secondary")
        return c[PRED_COLS].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def oracle_predictions(cands: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Perfect classifier on top of our candidates = recall upper bound."""
    lab = attach_gold(cands, gold)
    lab = lab[lab["label"] != "None"].drop_duplicates(["article_id", "dataset_id"])
    return lab.rename(columns={"label": "type"})[PRED_COLS]


def cross_validate(model_factory, cands: pd.DataFrame, gold: pd.DataFrame,
                   articles: list[str], n_splits: int = 5) -> dict:
    """GroupKFold over ARTICLES (a paper never appears in train and val).

    Returns the metric on the concatenated out-of-fold predictions plus the
    per-fold F1 scores.
    """
    articles = np.array(sorted(articles))
    gkf = GroupKFold(n_splits=n_splits)
    oof, fold_f1 = [], []
    for tr_idx, va_idx in gkf.split(articles, groups=articles):
        tr, va = set(articles[tr_idx]), set(articles[va_idx])
        model = model_factory()
        model.fit(cands[cands["article_id"].isin(tr)], gold[gold["article_id"].isin(tr)])
        pred = model.predict(cands[cands["article_id"].isin(va)])
        fold_f1.append(f1_tuples(pred, gold[gold["article_id"].isin(va)])["f1"])
        oof.append(pred)
    oof = pd.concat(oof, ignore_index=True) if oof else pd.DataFrame(columns=PRED_COLS)
    res = f1_tuples(oof, gold[gold["article_id"].isin(articles)])
    res["f1_fold_mean"], res["f1_fold_std"] = float(np.mean(fold_f1)), float(np.std(fold_f1))
    res["oof"] = oof
    return res
