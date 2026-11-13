# GROUP30_CENG313 – Make Data Count: Finding Data References

**Group ID:** 30

CENG313 term project based on the Kaggle competition
[Make Data Count – Finding Data References](https://www.kaggle.com/competitions/make-data-count-finding-data-references).

**Current Kaggle score (late submission, baseline B0):** public 0.480 · private 0.319 – screenshot in
[`submissions/`](submissions/)

## Team

| Name | Student ID | GitHub |
|------|-----------|--------|
| Ümmügülsüm Çiftçi | C21112022014 | [@ummugulsumciftci](https://github.com/ummugulsumciftci) |
| Ayça Yalçın | 23118080775 | [@Yalcinayca](https://github.com/Yalcinayca) |

## Competition and dataset

The goal is to find every reference to a research **dataset** in a scientific article and to classify it as

* **Primary** – data generated for this article, or
* **Secondary** – existing data reused by the article.

Datasets are cited either by **DOI** (e.g. `https://doi.org/10.5061/dryad.6m3n9`) or by an **accession ID**
(e.g. `GSE12345`, `PDB 1Y2T`, `E-MEXP-568`).

| File | Content |
|------|---------|
| `train/PDF/*.pdf` | 524 full-text training articles (one per `article_id`) |
| `train/XML/*.xml` | XML version for 400 of the training articles |
| `train_labels.csv` | `article_id, dataset_id, type` (`Primary` / `Secondary` / `Missing` = no counted dataset citation) |
| `test/PDF`, `test/XML` | articles to predict (the 30 visible test articles are also in the training set; the real test set is hidden) |
| `sample_submission.csv` | `row_id, article_id, dataset_id, type` |

**Metric:** F1 over unique `(article_id, dataset_id, type)` tuples – a prediction counts only if both the
identifier and its type are correct.

## Approach

### Assignment 1 – EDA and baseline (`assignment1_eda_baseline.ipynb`)
1. Text extraction from PDF (PyMuPDF, pypdf fallback) and XML, cleaning of identifiers broken by line breaks.
2. EDA: file inventory, label/type distribution, "Missing" rows, DOI vs accession, repositories, text length,
   recall ceiling, candidate noise, position of citations, discriminative context words.
3. Regex candidate extraction (DOIs + 18 accession families) with a ±250 character context window.
4. Baseline **B0 – regex + majority class**, evaluated with 5-fold GroupKFold by article: candidates are
   kept only if their DOI prefix / accession family was a dataset in the training folds, and each kept
   candidate gets the most frequent type of its kind (DOI or accession ID).

**Key EDA findings**
* 59 % of the labelled articles have no counted dataset; the 719 scored citations are 62 % Secondary, 38 % Primary,
  and the 10 largest articles contain 36 % of them.
* The identifier kind is the strongest signal: 66 % of DOIs are Primary, 86 % of accession IDs are Secondary.
* "Missing" rows still carry identifiers (figshare 10.6084, CCDC 10.5517) – being in a data repository is not enough.
* Only 2.9 % of the 20,416 regex candidates are real datasets; the candidates cover 82.6 % of the gold citations.
* Primary citations appear in data availability statements at the end of the paper, Secondary ones in
  methods/results.

**Baseline findings:** B0 learns DOI → Primary and accession → Secondary. Its main weakness is precision
(702 spurious predictions), because it ignores the context of a mention; 116 citations get the wrong type and
308 are missed (125 not found by the regexes, 183 removed by the repository filter).

### Assignment 2 – proposed solution (`assignment2_solution.ipynb`)
*To be added.* Planned: a context-based classifier for filtering and typing each mention, DOI repair and
additional accession patterns, compared against B0 with the same CV protocol.

## Results

5-fold GroupKFold cross-validation on the 523 labelled training articles.

| Model | Precision | Recall | F1 (CV) | F1 per fold (mean ± std) | Kaggle public | Kaggle private |
|-------|-----------|--------|---------|--------------------------|---------------|----------------|
| Oracle on regex candidates (upper bound) | 1.000 | 0.826 | 0.905 | – | – | – |
| B0 regex + majority | 0.265 | 0.410 | 0.322 | 0.321 ± 0.087 | 0.480 | 0.319 |

The private leaderboard score of B0 (0.319) is almost identical to its cross-validation F1 (0.322), which shows that
article-level GroupKFold is a reliable estimate of the performance on unseen articles. The public score (0.480) is
computed on a smaller part of the hidden test set and is therefore noisier. The screenshot of the submission is in
[`submissions/`](submissions/).

## Repository structure

```
GROUP30_CENG313/
├── README.md
├── requirements.txt
├── .gitignore
├── assignment1_eda_baseline.ipynb
├── assignment2_solution.ipynb      (placeholder, completed in Assignment 2)
├── src/
│   ├── __init__.py   empty, makes src importable
│   ├── data.py       paths, label loading, PDF/XML text extraction, cleaning
│   ├── models.py     baseline model B0, oracle, GroupKFold cross-validation
│   └── utils.py      identifier regexes, normalisation, candidates, F1 metric, submission
├── data/             NOT committed (downloaded with the Kaggle API, excluded in .gitignore)
└── submissions/      Kaggle score screenshots
```

## Data download

The data is not part of this repository. Join the competition on Kaggle and accept its rules, create an API key
(*Kaggle → Settings → API → Create Legacy API Key*, which downloads `kaggle.json`), then run:

```bash
kaggle competitions download -c make-data-count-finding-data-references -p data
unzip data/make-data-count-finding-data-references.zip -d data
```

In Colab, the second cell of the notebook does this automatically: it asks for `kaggle.json` and unpacks the data to
`/content/data`.

## How to run

**Google Colab** – [open `assignment1_eda_baseline.ipynb` in Colab](https://colab.research.google.com/github/ceng313-2026/GROUP30_CENG313/blob/main/assignment1_eda_baseline.ipynb)
1. Open `assignment1_eda_baseline.ipynb` in Colab (link above, or *File → Open notebook → GitHub*).
2. Set `GITHUB_USER` and `REPO_DIR` in the first cell and run it (clones the repo, installs PyMuPDF). For a private repo, add a `GITHUB_TOKEN` secret in Colab first.
3. Accept the competition rules on Kaggle, create `kaggle.json` (*Kaggle → Settings → API → Create Legacy API Key*) and upload it when the second cell asks.
4. *Runtime → Run all* (text extraction takes about one minute; the extracted texts are cached in `/content/train_texts.json`).

**Local**
```bash
git clone https://github.com/ceng313-2026/GROUP30_CENG313.git && cd GROUP30_CENG313
pip install -r requirements.txt
# download the data into data/ as described in "Data download"
jupyter notebook assignment1_eda_baseline.ipynb
```

**Kaggle (submission):** this is a code competition, so predictions must be produced by a Kaggle notebook with
internet disabled. Upload the `src/` folder as a private Kaggle Dataset and attach it, together with the competition
data, to a new notebook. The notebook copies `src/` to `/kaggle/working/src`, trains B0 on all training articles,
predicts the test articles and writes `/kaggle/working/submission.csv`; then *Save Version → Save & Run All* and
submit that version. PyMuPDF is not available offline on Kaggle, so PDF text is extracted with pypdf there
(slower – about 25 min for the training articles – and slightly different text than in Colab).

## Progress log

| Date | Member | Work |
|------|--------|------|
| 04-11-2026 | Ümmügülsüm Çiftçi | Repository created; README, .gitignore, requirements, `src/utils.py` and `src/data.py` added |
| 05-11-2026 | Ümmügülsüm Çiftçi | EDA: file inventory, labels, identifiers, text extraction |
| 06-11-2026 | Ümmügülsüm Çiftçi | EDA: extended accession regexes, candidates, positions, context words |
| 08-11-2026 | Ayça Yalçın | Baseline B0 (regex + majority) in `src/models.py`, 5-fold GroupKFold CV, error analysis, submission file |
| 08-11-2026 | Ümmügülsüm Çiftçi | EDA interpretations, summary and README results |
| 08-11-2026 | Ümmügülsüm Çiftçi | PDF extraction made robust to broken pages (pypdf on Kaggle); Kaggle notebook and first submission (B0) |
| 09-11-2026 | Ümmügülsüm Çiftçi | Kaggle results added to README (public 0.480, private 0.319) |
| 13-11-2026 | Ümmügülsüm Çiftçi | Notebook restructured into the 7 required sections: problem definition, data overview, preprocessing, interpretation of every plot, conclusion |


## Kaggle submission history

| Date | Notebook version | Model | Public LB | Private LB |
|------|------------------|-------|-----------|------------|
| 08-11-2026 | Version 2 ([notebook](https://www.kaggle.com/code/mmglsmifti/group30-ceng313)) | B0 regex + majority | 0.47957 | 0.31936 |

Late submission (after the competition deadline) from @ummugulsumciftci's Kaggle account. Version 1 was rejected
because internet was enabled in the notebook.
