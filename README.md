# SHP

Code and data for *"Confidence-Based Active
Learning Strategies for Addressing the Cold User Problem in
Recommender Systems Using Personalised Matrix Factorisation"*
(Ying Ying Tsai, MSc Data Science and Marketing Analytics, Erasmus
School of Economics, Erasmus University Rotterdam, supervised by
dr. Flavius Frasincar).

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Running Experiments](#running-experiments)
6. [Reproducing Paper Results](#reproducing-paper-results)
7. [Reproducibility](#reproducibility)
8. [Citation](#citation)
9. [License](#license)
10. [Contact](#contact)

---

## Overview

### The research problem

Recommender systems learn a user's preferences from their past
interactions (purchases, clicks, ratings). When a **new user** joins a
platform, no such history exists, and the system has no basis for
personalisation. This is the **cold user problem**, and it affects
every collaborative filtering system: e-commerce platforms must
recommend products to first-time visitors, streaming services must
suggest content to new subscribers, all without any prior signal about
that specific person's taste.

### Personalised active learning

One response to the cold user problem is **active learning**: instead
of passively waiting for interactions to accumulate, the system
deliberately chooses a small number of items to show the new user,
observes their response (did they engage with it or not), and updates
its estimate of that user's preferences before choosing the next item.
Non-personalised strategies (e.g. always show the most popular items)
apply the same fixed sequence to every new user; **personalised**
strategies adapt the sequence to each user's own revealed responses as
the session unfolds.

### The contribution

This paper proposes and evaluates a family of four confidence-based
personalised active learning strategies (SHHCP, SHLCP, SHMCP, SHECP;
see [Research Contributions](#research-contributions)), an
**incremental partial-SGD update** that makes personalised item
selection computationally feasible at the scale of a full user
population, and a **leakage-free evaluation protocol** that closes a
subtle methodological gap in how this class of methods has previously
been assessed. Under this corrected, fair evaluation, the paper's
central finding is that non-personalised, item-level baselines remain
highly competitive on prediction accuracy (RMSE) against personalised
selection in this dataset's sparsity regime - a result that runs
counter to the field's usual framing and is discussed at length in the
paper itself. This repository contains everything needed to
reproduce that finding from the raw dataset.

### SHHCP, SHLCP, SHMCP, SHECP

Four **S**ingle **H**euristic personalised active learning strategies,
differing only in *which* item they select at each step of the
elicitation session, given the model's current estimate of the cold
user's latent preference vector:

| Strategy | Full name | Selection rule | Learning-theory analogue |
|---|---|---|---|
| **SHHCP** | Single Heuristic Highest Confidence Prediction | Item the model is *most* confident the user will like | Pure exploitation |
| **SHLCP** | Single Heuristic Lowest Confidence Prediction | Item the model is *least* confident about | Pure exploration |
| **SHMCP** | Single Heuristic Median Confidence Prediction | Item closest to the median predicted score | Uncertainty sampling |
| **SHECP** | Single Heuristic Epsilon-greedy Confidence Prediction | Explores (SHLCP rule) with probability `epsilon_r`, exploits (SHHCP rule) otherwise; `epsilon_r` decays each round toward a floor | Epsilon-greedy (reinforcement learning) |

### Methodological contributions vs. engineering improvements

To be explicit about which parts of this repository are *research
contributions* (novel methodological ideas evaluated in the paper)
versus *engineering work* (needed to make those ideas practical, but
not themselves a claim):

| Methodological contributions | Engineering improvements |
|---|---|
| The four confidence-based selection strategies (SHHCP/SHLCP/SHMCP/SHECP) | The partial-SGD update's vectorised, parallelised implementation (`multiprocessing.Pool`) |
| Confidence-weighted shrinkage as a correction for cold-user estimate noise | Deterministic seeding via `hashlib` instead of Python's built-in `hash()` (a reproducibility fix, not a methodological claim) |
| The decaying learning-rate schedule for the incremental update | The model-cache mechanism (`base_model_cache.pkl`) that avoids re-running expensive setup |
| The leakage-free warm-user-only cross-validation protocol | Holm-Bonferroni / Benjamini-Hochberg multiple-comparisons correction (a statistical rigor tool applied to the results, not a new method) |
| Item-based cold-start initialisation (starting a new user's latent vector at the first shown item's own vector, rather than zero) | |

---

## Repository Structure

```
SHP/
├── README.md               <- this file
├── LICENSE                 <- MIT (code only - see LICENSE for the dataset note)
├── requirements.txt        <- pinned package versions (pip)
├── environment.yml         <- equivalent conda environment
├── .gitignore
│
├── data/
│   └── useritemmatrix.csv  <- the raw dataset (see "Dataset" below)
│
├── results/                <- generated by running the scripts; empty at checkout
│   └── (base_model_cache.pkl, *.csv - see "Running Experiments")
│
├── figures/                <- generated by the plotting scripts; empty at checkout
│   └── (*.png, *.pdf)
│
└── scripts/
    ├── model/               <- the recommender model + the four strategies
    │   ├── build_model_cache.py          <- STEP 1: one-time setup
    │   ├── baseline_ranking_metrics.py   <- STEP 2: non-personalised baselines
    │   ├── personalised_strategies.py    <- STEP 3: the four strategies (importable + runnable)
    │   └── run_complete_pipeline.py      <- convenience: runs steps 1-3 + significance in one command
    │
    ├── experiments/         <- ablations and statistical significance tests
    │   ├── decaying_lr_test.py                    <- ablation: constant vs. decaying learning rate
    │   ├── shrinkage_test.py                       <- ablation: shrinkage constant c sweep
    │   ├── b_ablation.py                            <- ablation: batch size B sweep
    │   ├── regularization_ablation.py               <- ablation: factor regularisation lambda2 sweep
    │   ├── shecp_grid_search.py                     <- ablation: SHECP epsilon floor/decay grid
    │   ├── measure_update_cost.py                   <- measured full-retrain-vs-partial-update speedup
    │   ├── significance_test.py                     <- paired Wilcoxon tests (personalised vs. baselines)
    │   └── ranking_significance_and_correction.py   <- NDCG@10 test + Holm/BH correction
    │
    └── plotting/             <- figure generation (run after the experiments above)
        ├── make_thesis_figures.py       <- RMSE and HR@10/NDCG@10 comparison figures
        └── make_shecp_grid_figure.py    <- SHECP floor/decay heatmap
```

**Why this split?** `model/` is the actual recommender system -
everything needed to produce the 's headline results (Chapter 4)
starting from raw data. `experiments/` contains the supporting
hyperparameter studies and statistical tests that justify the choices
made in `model/` - these are not needed to
reproduce the headline results themselves, only to reproduce *why*
those particular hyperparameters were chosen. `plotting/` only ever
reads already-computed CSVs from `results/`; it never runs the model.

---

## Installation

### Requirements

- **Python 3.11** (the exact version this code was developed and
  tested against; see [Reproducibility](#reproducibility))
- A C compiler (needed to build `scikit-surprise`, which compiles a
  Cython extension on install) - on macOS, install Xcode Command Line
  Tools (`xcode-select --install`); on Ubuntu/Debian,
  `sudo apt install build-essential`; on Windows, install the
  "Desktop development with C++" workload from Visual Studio Build
  Tools.

### Option A: pip + virtual environment (recommended)

```bash
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate cold-user-active-learning
```

### Verify the install

```bash
python -c "import numpy, pandas, scipy, sklearn, matplotlib, surprise; print('OK')"
```

---

## Dataset

### Source

The dataset records binary implicit feedback (purchase and retained =
1; returned or not purchased = 0) from real customer transactions at
de Bijenkorf, a Dutch premium department store, first used in this
line of research by Geurts et al. (2020). It is provided in this
repository as `data/useritemmatrix.csv`.

**Format**: a CSV with columns `userId`, `itemId`, `interaction`
(0 or 1). One row per observed (user, item) interaction.

**Note on file size**: `useritemmatrix.csv` is ~92 MB. If forking or
re-uploading this repository to your own GitHub, consider using
[Git LFS](https://git-lfs.github.com/) for this file rather than a
plain commit.

### Preprocessing (handled automatically)

`scripts/model/build_model_cache.py` performs all preprocessing; there
is no separate preprocessing script to run first. In order, it:

1. Drops any user with zero recorded interactions.
2. Assigns integer category codes (`user_idx`, `item_idx`) to each raw
   `userId`/`itemId`.
3. Randomly selects 25% of users as **cold users** (withheld from all
   base-model training), leaving the remaining 75% as **warm users**.
4. Restricts eligible items to those with at least 10 warm-user
   interactions (`MIN_ITEM_INTERACTIONS` in the script).

### Recreating the cold/warm split

The split is regenerated fresh every time `build_model_cache.py` runs,
using a fixed random seed (`np.random.RandomState(1)`), so it is
identical across runs on the same data file - there is no separate
"split file" to manage. If you need the *exact* set of cold users used
by a particular run, load `results/base_model_cache.pkl` and read its
`cold_users` key (see [Reproducibility](#reproducibility) for what
else is inside that file).

### Expected files before running anything

| Path | Required before running |
|---|---|
| `data/useritemmatrix.csv` | Everything (this is the only required input file) |

Every other file under `results/` and `figures/` is generated by the
scripts in this repository - see [Running Experiments](#running-experiments).

---

## Running Experiments

**Every script in this repository is run from the repository root**
(this folder), so that its relative paths to `data/`, `results/`, and
`figures/` resolve correctly. If cloned into a directory whose name
contains spaces (e.g. the default `SHP`), quote the path when changing into it:

```bash
cd "SHP"
python scripts/model/build_model_cache.py     # NOT: cd scripts/model && python build_model_cache.py
```

### The order matters - what depends on what

```
STEP 1  build_model_cache.py
           │
           ▼  (writes results/base_model_cache.pkl)
           │
   ┌───────┼────────────────────────────────────────────┐
   ▼       ▼                                             ▼
STEP 2  STEP 3                                    STEP 4 (optional)
baseline_  personalised_                          the five ablations in
ranking_   strategies.py                          scripts/experiments/
metrics.py                                         (decaying_lr_test.py,
   │       │                                        shrinkage_test.py,
   │       │  (writes results/                      b_ablation.py,
   │       │   personalised_results.csv)             regularization_
   │       │                                         ablation.py,
   │       ▼                                         shecp_grid_search.py)
   │    STEP 5  significance_test.py                (each independent;
   │       │    (needs results/personalised_         run any subset,
   │       │     results.csv from STEP 3)             any order)
   │       ▼
   │    STEP 6  ranking_significance_and_correction.py
   │            (needs results/significance_results.csv from STEP 5)
   │
   └──────►  STEP 7 (optional)  scripts/plotting/*.py
             (needs results/baseline_results.csv from STEP 2 and
              results/personalised_results.csv from STEP 3;
              make_shecp_grid_figure.py additionally needs
              results/shecp_grid_results.csv from the STEP 4 ablation
              of the same name)
```

In short: **STEP 1 must run before everything else. STEP 3 must run
before STEP 5. STEP 5 must run before STEP 6.** Everything else is
independent and can be run in any order, or skipped if you only need
specific numbers.

### Exact commands, in the order you would actually run them

```bash
# STEP 1 - one-time setup: trains the base SVD model, tunes ALPHA and
# (n_factors, reg_all, epochs). ~20-25 minutes. Writes
# results/base_model_cache.pkl, which every later step loads instead
# of redoing this setup.
python scripts/model/build_model_cache.py

# STEP 2 - non-personalised baselines (Random, Popularity, PopError):
# RMSE + HR@K/NDCG@K on 1,000 cold users. ~8 minutes. Writes
# results/baseline_results.csv 
python scripts/model/baseline_ranking_metrics.py

# STEP 3 - the four personalised strategies, full scale. Running the
# file directly only executes a 20-user smoke test (a quick
# correctness check); the full 1,000-user x 4-strategy x 4-k
# evaluation is only triggered via run_complete_pipeline.py (below) or
# by importing this module's run() function yourself. ~80-85 minutes
# for the full run. Writes results/personalised_results.csv
python scripts/model/personalised_strategies.py        # smoke test only
python scripts/model/run_complete_pipeline.py           # full pipeline: see below

# STEP 5 - paired Wilcoxon significance tests, personalised vs. each
# baseline (48 tests). Needs STEP 3's output. ~16 minutes. Writes
# results/significance_results.csv.
python scripts/experiments/significance_test.py

# STEP 6 - NDCG@10 significance test + Holm-Bonferroni/Benjamini-
# Hochberg multiple-comparisons correction. Needs STEP 5's output.
# ~16 minutes. Writes results/ranking_significance_results.csv and
# results/significance_results_corrected.csv
python scripts/experiments/ranking_significance_and_correction.py
```

### The all-in-one command

`run_complete_pipeline.py` runs STEPS 1-3, 5, and 6 above in the
correct order automatically (skipping STEP 1 if
`results/base_model_cache.pkl` already exists), including the full
1,000-user STEP 3 evaluation (not just its smoke test):

```bash
python scripts/model/run_complete_pipeline.py    # ~2-2.5 hours total
```

### STEP 4. The hyperparameter ablations (optional)

Each is independent of the others and of STEPS 2/3/5/6 above (all read
`results/base_model_cache.pkl` directly) - run any subset, in any
order:

```bash
python scripts/experiments/decaying_lr_test.py           # ~5 min   
python scripts/experiments/shrinkage_test.py              # ~5 min   
python scripts/experiments/b_ablation.py                  # ~10 min  
python scripts/experiments/regularization_ablation.py     # ~5 min   
python scripts/experiments/shecp_grid_search.py           # ~2 hours 
python scripts/experiments/measure_update_cost.py         # ~1 min   
```

### STEP 7. Figures (optional)

Run after the results CSVs they read from already exist:

```bash
python scripts/plotting/make_thesis_figures.py       # needs STEPS 2+3's output
python scripts/plotting/make_shecp_grid_figure.py     # needs shecp_grid_search.py's output (STEP 4)
```

---


## Reproducibility

### Random seeds

Every source of randomness in this repository is deterministically
seeded:

- **Cold/warm split, ALPHA search, GridSearchCV**: fixed
  `np.random.RandomState` seeds in `build_model_cache.py`
  (`RandomState(1)` for the split, `RandomState(99)` for ALPHA
  search; `GridSearchCV(..., n_jobs=1)` for a deterministic
  cross-validation order).
- **Validation/test item splits, per-work-item epsilon-greedy and
  negative-sampling draws**: derived from `_stable_seed(u, shown)`, a
  `hashlib.md5`-based deterministic seed - used in place of Python's
  built-in `hash()`, which is randomised per-process
  (`PYTHONHASHSEED`) unless explicitly fixed, and would otherwise make
  every split non-reproducible both across separate runs *and* across
  the worker processes of a single parallel run.
- **SHECP grid search's epsilon-greedy draws**: a single
  `np.random.RandomState(123)` stream, deliberately *not*
  re-seeded per user, and the grid search itself deliberately *not*
  parallelised - so the sequence of draws (and therefore the result)
  cannot depend on how many workers are used.

Re-running any script in this repository, on the same machine/library
versions, reproduces its output exactly.

### Software versions used to produce the 's results

| Package | Version |
|---|---|
| Python | 3.11.0 |
| numpy | 1.26.4 |
| pandas | 3.0.1 |
| scipy | 1.17.1 |
| scikit-surprise | 1.1.4 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |

(Exact versions pinned in `requirements.txt`/`environment.yml`.)

### Hardware and OS

- macOS 15.6.1 (Darwin 24.6.0), Apple M4 Pro, 12 CPU cores.
- The full personalised-strategy evaluation
  (`run_complete_pipeline.py`) uses 10 worker processes
  (`FULL_N_WORKERS` in that script); reduce this if running on a
  machine with fewer cores.

### Parallel execution

Only `scripts/model/personalised_strategies.py`'s full-scale run uses
multiprocessing (`multiprocessing.Pool`, `spawn` start method, one
worker process per requested core). Every other script in this
repository is single-process. `scripts/experiments/shecp_grid_search.py`
is deliberately sequential even though it runs on the full 1,000-user
population - see [Random seeds](#random-seeds) above for why.

### Measured runtimes (this hardware; scale accordingly on other machines)

| Script | Runtime |
|---|---|
| `build_model_cache.py` | ~20-25 min |
| `baseline_ranking_metrics.py` | ~8 min |
| `personalised_strategies.py` (full, via `run_complete_pipeline.py`) | ~80-85 min |
| `significance_test.py` | ~16 min |
| `ranking_significance_and_correction.py` | ~16 min |
| `decaying_lr_test.py` | ~5 min |
| `shrinkage_test.py` | ~5 min |
| `b_ablation.py` | ~10 min |
| `regularization_ablation.py` | ~5 min |
| `shecp_grid_search.py` | ~2 hours |
| **Full pipeline** (`run_complete_pipeline.py`) | **~2-2.5 hours** |

---

## Citation

This work builds directly on:

```bibtex
@article{geurts2020,
  author  = {Geurts, T. and Giannikis, G. and Frasincar, F.},
  title   = {Active learning strategies for solving the cold user
             problem in model-based recommender systems},
  journal = {Web Intelligence},
  volume  = {18},
  number  = {4},
  pages   = {269--283},
  year    = {2020}
}

@inproceedings{he2017,
  author    = {He, X. and Liao, L. and Zhang, H. and Nie, L.
               and Hu, X. and Chua, T.-S.},
  title     = {Neural collaborative filtering},
  booktitle = {26th International Conference
               on World Wide Web (WWW)},
  pages     = {173--182},
  publisher = {ACM},
  year      = {2017}
}

@article{efron1975,
  author  = {Efron, Bradley and Morris, Carl},
  title   = {Data Analysis Using Stein's Estimator and Its Generalizations},
  journal = {Journal of the American Statistical Association},
  volume  = {70},
  number  = {350},
  pages   = {311--319},
  year    = {1975}
}

@article{robbins1951,
  author  = {Robbins, Herbert and Monro, Sutton},
  title   = {A Stochastic Approximation Method},
  journal = {The Annals of Mathematical Statistics},
  volume  = {22},
  number  = {3},
  pages   = {400--407},
  year    = {1951}
}
```

---

## License

The source code in this repository is released under the **MIT
License** - see [LICENSE](LICENSE). MIT was chosen as a permissive,
widely-recognised license that places minimal restrictions on reuse,
appropriate for academic research code intended to support
reproducibility.

The dataset (`data/useritemmatrix.csv`) is **not** covered by the MIT
license; see the note at the bottom of [LICENSE](LICENSE) for its
provenance.

---

## Contact

Ying Ying Tsai
MSc Data Science and Marketing Analytics, Erasmus School of Economics
Erasmus University Rotterdam
