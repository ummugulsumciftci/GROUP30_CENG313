"""Identifier patterns, normalisation, candidate extraction and the metric."""
from __future__ import annotations

import random
import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Identifier patterns
# --------------------------------------------------------------------------- #
DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>\]\[{}|\\^`]+", re.I)

# Accession-ID families that are common in the labels. The key is used as the
# "pattern" feature, the regex must contain exactly one capturing group.
ACCESSION_PATTERNS: dict[str, re.Pattern] = {
    "GEO": re.compile(r"\b(GS[EM]\d{3,8})\b"),
    "SRA": re.compile(r"\b([SED]R[RPXS]\d{5,9})\b"),
    "BioProject": re.compile(r"\b(PRJ[NED][A-Z]\d{3,8})\b"),
    "BioSample": re.compile(r"\b(SAM[NED][A-Z]?\d{5,10})\b"),
    "ArrayExpress": re.compile(r"\b(E-[A-Z]{4}-\d{1,6})\b"),
    "EMDB": re.compile(r"\b(EMD-\d{4,5})\b"),
    "PDB": re.compile(r"\b(?:PDB|Protein Data Bank)(?:\s+(?:ID|IDs|code|codes|entry|accession))?[\s:]+([1-9][A-Za-z0-9]{3})\b"),
    "ChEMBL": re.compile(r"\b(CHEMBL\d{1,9})\b"),
    "InterPro": re.compile(r"\b(IPR\d{6})\b"),
    "Pfam": re.compile(r"\b(PF\d{5})\b"),
    "Ensembl": re.compile(r"\b(ENS[A-Z]{0,4}[GTP]\d{11})\b"),
    "Cellosaurus": re.compile(r"\b(CVCL_[A-Z0-9]{4})\b"),
    "GISAID": re.compile(r"\b(EPI_ISL_\d{4,9})\b"),
    "dbSNP": re.compile(r"\b(rs\d{4,10})\b"),
    "EMPIAR": re.compile(r"\b(EMPIAR-\d{5})\b"),
    "RefSeq": re.compile(r"\b((?:NM|NR|NP|XM|XR|XP|NC|NG|NT|NW|NZ)_\d{6,9})(?:\.\d+)?\b"),
    "UniProt": re.compile(r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])\b"),
    "GenBank": re.compile(r"\b([A-Z]{2}\d{6}|[A-Z]\d{5})(?:\.\d)?\b"),
}

_DOI_PREFIX_STRIP = re.compile(r"^(?:https?://)?(?:dx\.)?(?:doi\.org/)|^doi:\s*", re.I)


def _balance_brackets(s: str) -> str:
    """Remove trailing ')' or ']' that do not belong to the identifier."""
    for open_, close in (("(", ")"), ("[", "]")):
        while s.endswith(close) and s.count(close) > s.count(open_):
            s = s[:-1]
    return s


def normalize_doi(doi: str) -> str:
    """'doi:10.5061/DRYAD.X.' -> 'https://doi.org/10.5061/dryad.x'."""
    d = _DOI_PREFIX_STRIP.sub("", doi.strip())
    d = d.rstrip(".,;:'\"")
    d = _balance_brackets(d).rstrip(".,;:")
    return "https://doi.org/" + d.lower()


def is_doi(s: str) -> bool:
    return bool(re.search(r"10\.\d{4,9}/", str(s)))


def normalize_dataset_id(s: str) -> str:
    s = str(s).strip()
    return normalize_doi(s) if is_doi(s) else s


def id_kind(dataset_id: str) -> str:
    return "DOI" if is_doi(dataset_id) else "Accession"


def doi_prefix(dataset_id: str) -> str | None:
    """'https://doi.org/10.5061/dryad.x' -> '10.5061' (registrant code)."""
    m = re.search(r"(10\.\d{4,9})/", str(dataset_id))
    return m.group(1) if m else None


def accession_family(dataset_id: str) -> str:
    for name, pat in ACCESSION_PATTERNS.items():
        if pat.fullmatch(str(dataset_id)) or pat.search(str(dataset_id)):
            return name
    return "Other"


def article_doi(article_id: str) -> str:
    """article ids are DOIs with the first '/' replaced by '_'."""
    return normalize_doi(article_id.replace("_", "/", 1))


# --------------------------------------------------------------------------- #
# Candidate extraction
# --------------------------------------------------------------------------- #
def find_candidates(article_id: str, text: str, window: int = 250,
                    ref_start: int | None = None) -> list[dict]:
    """Return every DOI / accession match in the text with its context.

    One row per *mention*; the same dataset_id can appear several times.
    The article's own DOI is skipped.
    """
    own = article_doi(article_id)
    ref_start = len(text) if ref_start is None else ref_start
    rows = []

    def add(ds_id, kind, pattern, start, end):
        lo, hi = max(0, start - window), min(len(text), end + window)
        rows.append({
            "article_id": article_id,
            "dataset_id": ds_id,
            "kind": kind,
            "pattern": pattern,
            "start": start,
            "rel_pos": start / max(len(text), 1),
            "in_references": start >= ref_start,
            "context": text[lo:hi],
        })

    for m in DOI_RE.finditer(text):
        ds = normalize_doi(m.group(0))
        if ds == own or len(ds) < len("https://doi.org/10.1234/x"):
            continue
        add(ds, "DOI", doi_prefix(ds), m.start(), m.end())

    for name, pat in ACCESSION_PATTERNS.items():
        for m in pat.finditer(text):
            ds = m.group(1).upper() if name == "PDB" else m.group(1)
            add(ds, "Accession", name, m.start(1), m.end(1))
    return rows


def build_candidates(texts: dict[str, str], window: int = 250, ref_start_fn=None) -> pd.DataFrame:
    rows = []
    for aid, text in texts.items():
        rs = ref_start_fn(text) if ref_start_fn else None
        rows.extend(find_candidates(aid, text, window, rs))
    cols = ["article_id", "dataset_id", "kind", "pattern", "start", "rel_pos", "in_references", "context"]
    return pd.DataFrame(rows, columns=cols)


def attach_gold(cands: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Add a 'label' column: Primary / Secondary if the candidate is a true
    dataset reference of that article, otherwise 'None'."""
    g = gold[["article_id", "dataset_id", "type"]].drop_duplicates(["article_id", "dataset_id"])
    out = cands.merge(g, on=["article_id", "dataset_id"], how="left")
    out["label"] = out["type"].fillna("None")
    return out.drop(columns="type")


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #
def f1_tuples(pred: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """Competition metric: F1 over unique (article_id, dataset_id, type).

    A prediction counts as TP only if the dataset_id AND the type are right.
    """
    key = ["article_id", "dataset_id", "type"]
    p = set(map(tuple, pred[key].drop_duplicates().values)) if len(pred) else set()
    g = set(map(tuple, gold[key].drop_duplicates().values))
    tp, fp, fn = len(p & g), len(p - g), len(g - p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def make_submission(pred: pd.DataFrame, path: str = "submission.csv") -> pd.DataFrame:
    sub = pred[["article_id", "dataset_id", "type"]].drop_duplicates().reset_index(drop=True)
    sub.insert(0, "row_id", range(len(sub)))
    sub.to_csv(path, index=False)
    return sub


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
