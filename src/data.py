"""Data loading and text extraction for the Make Data Count competition.

Folder layout of the competition data (as downloaded from Kaggle):

    <root>/
        train_labels.csv          article_id, dataset_id, type
        sample_submission.csv     row_id, article_id, dataset_id, type
        train/PDF/<article_id>.pdf
        train/XML/<article_id>.xml   (only for part of the articles)
        test/PDF/<article_id>.pdf
        test/XML/<article_id>.xml
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from .utils import normalize_dataset_id, id_kind

COMPETITION = "make-data-count-finding-data-references"

DEFAULT_DATA_DIRS = [
    f"/kaggle/input/{COMPETITION}",  # Kaggle notebook
    "/content/data",                 # Google Colab (after kaggle download)
    "./data",                        # local run
]


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def find_data_dir(candidates: list[str] | None = None) -> Path:
    """Return the first directory that contains train_labels.csv."""
    for c in candidates or DEFAULT_DATA_DIRS:
        p = Path(c)
        if (p / "train_labels.csv").exists():
            return p
    raise FileNotFoundError(
        "train_labels.csv not found. Download the data first "
        f"(kaggle competitions download -c {COMPETITION})."
    )


class Paths:
    """Small helper so that file paths are written in one place only."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @property
    def labels(self) -> Path:
        return self.root / "train_labels.csv"

    @property
    def sample_submission(self) -> Path:
        return self.root / "sample_submission.csv"

    def pdf_dir(self, split: str = "train") -> Path:
        return self.root / split / "PDF"

    def xml_dir(self, split: str = "train") -> Path:
        return self.root / split / "XML"


# --------------------------------------------------------------------------- #
# Labels and file inventory
# --------------------------------------------------------------------------- #
def load_labels(paths: Paths) -> pd.DataFrame:
    """Read train_labels.csv and add helper columns.

    type == "Missing" marks articles that cite no dataset; these rows are kept
    for EDA but are never part of the scored tuples.
    """
    df = pd.read_csv(paths.labels)
    df["dataset_id_raw"] = df["dataset_id"]
    is_missing = df["type"].eq("Missing")
    df.loc[~is_missing, "dataset_id"] = df.loc[~is_missing, "dataset_id"].map(normalize_dataset_id)
    df["is_missing"] = is_missing
    df["id_kind"] = df["dataset_id"].map(id_kind).where(~is_missing, "Missing")
    return df


def scored_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Unique (article_id, dataset_id, type) tuples that the metric uses."""
    out = labels.loc[~labels["is_missing"], ["article_id", "dataset_id", "type"]]
    return out.drop_duplicates().reset_index(drop=True)


def list_articles(paths: Paths, split: str = "train") -> pd.DataFrame:
    """One row per article with the PDF / XML files that exist for it."""
    pdfs = {p.stem: p for p in paths.pdf_dir(split).glob("*.pdf")}
    xmls = {p.stem: p for p in paths.xml_dir(split).glob("*.xml")}
    ids = sorted(set(pdfs) | set(xmls))
    return pd.DataFrame({
        "article_id": ids,
        "pdf_path": [str(pdfs[i]) if i in pdfs else None for i in ids],
        "xml_path": [str(xmls[i]) if i in xmls else None for i in ids],
        "has_pdf": [i in pdfs for i in ids],
        "has_xml": [i in xmls for i in ids],
    })


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #
def extract_pdf_text(path: str | Path) -> str:
    """Extract raw text from a PDF.

    PyMuPDF (fitz) is fast and keeps the reading order well; pypdf is a pure
    python fallback (used e.g. on Kaggle, where PyMuPDF is not installed).
    """
    try:
        import fitz  # PyMuPDF

        with fitz.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except ImportError:
        pass
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # one broken page must not drop the whole article
            pages.append("")
    return "\n".join(pages)


def extract_xml_text(path: str | Path) -> str:
    """Return all text nodes of a JATS / TEI XML file, ignoring tags."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        raw = Path(path).read_text(encoding="utf-8", errors="ignore")
        return re.sub(r"<[^>]+>", " ", raw)
    parts = []
    for el in root.iter():
        # attributes such as xlink:href often hold the DOI / URL of the dataset
        for k, v in el.attrib.items():
            if k.endswith("href") and v:
                parts.append(v)
        if el.text:
            parts.append(el.text)
        if el.tail:
            parts.append(el.tail)
    return " ".join(parts)


_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
# a DOI cut at a line end is glued back only when it clearly continues:
# it ends in / _ - (any next char) or in "." followed by a lower-case letter / digit
_DOI_BREAK = re.compile(r"(10\.\d{4,9}/\S*(?:[/_-][ \t]*\n[ \t]*(?=\w)|\.[ \t]*\n[ \t]*(?=[a-z0-9])))")


def clean_text(text: str) -> str:
    """Normalise text so that identifiers are not split by layout artefacts."""
    text = unicodedata.normalize("NFKC", text)
    for dash in ("\u2010", "\u2011", "\u2013", "\u2014"):
        text = text.replace(dash, "-")
    text = text.replace("\u00ad", "")              # soft hyphen
    text = _DOI_BREAK.sub(lambda m: re.sub(r"\s+", "", m.group(1)), text)  # DOI broken over a line end
    text = _HYPHEN_BREAK.sub(r"\1-\2", text)       # keep hyphen, drop line break
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def article_text(row: dict, source: str = "both") -> str:
    """Text of one article. source in {"pdf", "xml", "both", "xml_first"}."""
    pdf_path, xml_path = row.get("pdf_path"), row.get("xml_path")
    pdf_path = pdf_path if isinstance(pdf_path, str) else None   # NaN / None -> None
    xml_path = xml_path if isinstance(xml_path, str) else None
    pdf, xml = "", ""
    if pdf_path and source in ("pdf", "both", "xml_first"):
        try:
            pdf = extract_pdf_text(pdf_path)
        except Exception as e:  # keep the XML text if only the PDF is broken
            print(f"[warn] PDF failed, using XML only: {Path(pdf_path).stem}: {e}")
    if xml_path and source in ("xml", "both", "xml_first"):
        xml = extract_xml_text(xml_path)
    if source == "xml_first":
        return clean_text(xml) if xml else clean_text(pdf)
    return clean_text(" \n ".join(t for t in (pdf, xml) if t))


def load_texts(articles: pd.DataFrame, source: str = "both",
               cache_path: str | Path | None = None, verbose: bool = True) -> dict[str, str]:
    """Extract (and optionally cache as JSON) the text of every article."""
    if cache_path and Path(cache_path).exists():
        return json.loads(Path(cache_path).read_text())
    texts, t0 = {}, time.time()
    for i, row in enumerate(articles.itertuples(index=False), 1):
        try:
            texts[row.article_id] = article_text(row._asdict(), source)
        except Exception as e:  # a broken file must not stop the pipeline
            print(f"[warn] {row.article_id}: {e}")
            texts[row.article_id] = ""
        if verbose and i % 100 == 0:
            print(f"{i}/{len(articles)} articles, {time.time() - t0:.0f}s")
    if cache_path:
        Path(cache_path).write_text(json.dumps(texts))
    return texts


_REF_HEADER = re.compile(r"\b(references|bibliography|literature cited|works cited)\b", re.I)


def reference_start(text: str) -> int:
    """Character offset where the reference list probably starts.

    Uses the LAST "References"-like heading in the second half of the text;
    returns len(text) if none is found.
    """
    half = len(text) // 2
    starts = [m.start() for m in _REF_HEADER.finditer(text) if m.start() > half]
    return starts[-1] if starts else len(text)
