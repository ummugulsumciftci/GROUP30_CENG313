# GROUP30_CENG313 – Make Data Count: Finding Data References

CENG313 term project based on the Kaggle competition
[Make Data Count – Finding Data References](https://www.kaggle.com/competitions/make-data-count-finding-data-references).

## Team

| Name | Student ID | GitHub |
|------|-----------|--------|
| Ümmügülsüm Çiftçi | C21112022014 | [@ummugulsumciftci](https://github.com/ummugulsumciftci) |

## Competition and dataset

The goal is to find every reference to a research **dataset** in a scientific article and to classify it as

* **Primary** – data generated for this article, or
* **Secondary** – existing data reused by the article.

Datasets are cited either by **DOI** (e.g. `https://doi.org/10.5061/dryad.6m3n9`) or by an **accession ID**
(e.g. `GSE12345`, `PDB 1Y2T`, `E-MEXP-568`).

| File | Content |
|------|---------|
| `train/PDF/*.pdf` | full-text training articles (one per `article_id`) |
| `train/XML/*.xml` | XML version for part of the training articles |
| `train_labels.csv` | `article_id, dataset_id, type` (`Primary` / `Secondary` / `Missing` = no dataset) |
| `test/PDF`, `test/XML` | articles to predict |
| `sample_submission.csv` | `row_id, article_id, dataset_id, type` |

**Metric:** F1 over unique `(article_id, dataset_id, type)` tuples.

## Approach

### Assignment 1 – EDA and baseline (`assignment1_eda_baseline.ipynb`)
1. Text extraction from PDF (PyMuPDF, pypdf fallback) and XML, cleaning of identifiers broken by line breaks.
2. EDA: file inventory, label/type distribution, DOI vs accession, repositories, text length,
   recall ceiling, candidate noise, position of citations, discriminative context words.
3. Regex candidate extraction (DOIs + 16 accession families) with a ±250 character context window.
4. Baseline **B0 – regex + majority class**, evaluated with 5-fold GroupKFold by article: candidates are
   kept only if their DOI prefix / accession family was a dataset in the training folds, and each kept
   candidate gets the most frequent type of its kind (DOI or accession ID).

### Assignment 2 – proposed solution (`assignment2_solution.ipynb`)
*To be added.*

## Results

| Model | Precision | Recall | F1 (CV) | Kaggle public | Kaggle private |
|-------|-----------|--------|---------|---------------|----------------|
| Oracle on regex candidates (upper bound) | 1.000 | | | – | – |
| B0 regex + majority | | | | | |

## Repository structure

```
GROUP30_CENG313/
├── README.md
├── requirements.txt
├── .gitignore
├── assignment1_eda_baseline.ipynb
├── assignment2_solution.ipynb      (Assignment 2)
└── src/
    ├── __init__.py
    ├── data.py     paths, label loading, PDF/XML text extraction, cleaning
    ├── models.py   baseline model B0 + GroupKFold cross-validation
    └── utils.py    identifier regexes, normalisation, candidates, F1 metric, submission
```

## How to run

**Google Colab**
1. Open `assignment1_eda_baseline.ipynb` in Colab (*File → Open notebook → GitHub*).
2. Set `GITHUB_USER` and `REPO_DIR` in the first cell and run it (clones the repo, installs PyMuPDF). For a private repo, add a `GITHUB_TOKEN` secret in Colab first.
3. Accept the competition rules on Kaggle, download `kaggle.json` (*Kaggle → Settings → API → Create New Token*) and upload it when the second cell asks.
4. *Runtime → Run all.*

**Local**
```bash
git clone https://github.com/ummugulsumciftci/GROUP30_CENG313.git && cd GROUP30_CENG313
pip install -r requirements.txt
kaggle competitions download -c make-data-count-finding-data-references -p data && unzip data/*.zip -d data
jupyter notebook assignment1_eda_baseline.ipynb
```

**Kaggle (submission):** upload `src/` as a Kaggle Dataset, attach it and the competition data to a new
notebook (internet off), add `sys.path.append("/kaggle/input/<src-dataset>")` and run the notebook's
submission section, which writes `submission.csv`.

## Progress log

| Date | Member | Work |
|------|--------|------|
| 04-11-2026 | Ümmügülsüm Çiftçi | Repository created; README, .gitignore and requirements added |

## Kaggle submission history

| Date | Notebook version | Model | Public LB | Private LB |
|------|------------------|-------|-----------|------------|
| | | | | |
