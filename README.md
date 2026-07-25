# Confidence-Based Active Learning Strategies for the Cold User Problem in Recommender Systems

Code and data accompanying the MSc thesis *"Confidence-Based Active
Learning Strategies for Addressing the Cold User Problem in
Recommender Systems Using Personalised Matrix Factorisation"*
(Ying Ying Tsai, MSc Data Science and Marketing Analytics, Erasmus
School of Economics, Erasmus University Rotterdam, supervised by
dr. Flavius Frasincar).

---

## Table of Contents

1. [Overview](#overview)
2. [Research Contributions](#research-contributions)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Dataset](#dataset)
6. [Running Experiments](#running-experiments)
7. [Reproducing Thesis Results](#reproducing-thesis-results)
8. [Methodology](#methodology)
9. [Configuration](#configuration)
10. [Reproducibility](#reproducibility)
11. [Citation](#citation)
12. [License](#license)
13. [Contact](#contact)

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

This thesis proposes and evaluates a family of four confidence-based
personalised active learning strategies (SHHCP, SHLCP, SHMCP, SHECP;
see [Research Contributions](#research-contributions)), an
**incremental partial-SGD update** that makes personalised item
selection computationally feasible at the scale of a full user
population, and a **leakage-free evaluation protocol** that closes a
subtle methodological gap in how this class of methods has previously
been assessed. Under this corrected, fair evaluation, the thesis's
central finding is that non-personalised, item-level baselines remain
highly competitive on prediction accuracy (RMSE) against personalised
selection in this dataset's sparsity regime - a result that runs
counter to the field's usual framing and is discussed at length in the
thesis itself. This repository contains everything needed to
reproduce that finding, and every other number and figure in the
thesis, from the raw dataset.

---

## Research Contributions

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

SHHCP tends to converge onto a narrow, already-well-understood slice
of the item space as the session grows ("the exploitation trap");
SHLCP is the most consistently accurate across elicitation budgets;
SHECP is strongest at small budgets, where its early high exploration
rate resembles SHLCP, before its accuracy converges toward SHHCP's as
epsilon decays. See `scripts/model/personalised_strategies.py` for the
exact selection-rule implementation and the thesis's Chapter 4 for the
full empirical comparison.

### Incremental partial-SGD update

The straightforward way to update a matrix-factorisation model after a
cold user reveals a new interaction is to **retrain the entire model**
on all warm-user data plus the new observation -- correct, but far too
slow to apply after every single interaction across a population of
hundreds of thousands of cold users. This thesis instead derives a
**partial update**: only the cold user's own parameters and the
revealed item's own vector/bias are adjusted via one step of gradient
descent; every other user's and item's parameters are left untouched,
since the observation carries no direct information about them. This
reduces the per-interaction update cost from a full retrain (which
scales with the entire warm-user dataset) to a small, constant number
of arithmetic operations per latent factor -- independent of dataset
size. `scripts/model/personalised_strategies.py`'s
docstrings give the full derivation and complexity argument; a
measured (not just asymptotic) wall-clock comparison
(`scripts/experiments/measure_update_cost.py`, ~6.5x10^6x speedup) is
one of the thesis's supporting results.

### Confidence-weighted shrinkage

RMSE penalises confident-but-wrong predictions harshly. A cold user's
personalised estimate, built from only 10-100 revealed interactions,
is inherently noisier than an item's own average interaction rate,
which is estimated from the entire warm-user population. **Shrinkage**
blends the two: the final prediction is a weighted mix of the
personalised estimate and the stable item-level baseline, with the
weight increasing as more evidence (a larger elicitation budget `k`)
accumulates -- an empirical-Bayes-style correction (Efron & Morris,
1975) that is a direct response to this noise-versus-evidence
trade-off, not merely a post-hoc rescaling.

### Leakage-free evaluation

If a cold user's own behaviour is allowed to influence which
hyperparameters get selected for the *base* model (the one trained on
warm users), the evaluation is contaminated: the model has effectively
already "seen" information about the very users it will later be
tested on. This repository's cross-validation protocol
(`scripts/model/build_model_cache.py`) is run **exclusively on warm
users**, so no cold-user information of any kind reaches model
selection. Applying this fix retroactively is what first exposed the
thesis's headline finding (non-personalised baselines are far more
competitive than previously reported) -- the earlier, leakier
evaluation had been silently flattering the personalised strategies by
giving their comparison baseline an unfair advantage.

### Methodological contributions vs. engineering improvements

To be explicit about which parts of this repository are *research
contributions* (novel methodological ideas evaluated in the thesis)
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
Single Heuristic Personalized (SHP)/
├── README.md               <- this file
├── LICENSE                 <- MIT (code only -- see LICENSE for the dataset note)
├── requirements.txt        <- pinned package versions (pip)
├── environment.yml         <- equivalent conda environment
├── .gitignore
│
├── data/
│   └── useritemmatrix.csv  <- the raw dataset (see "Dataset" below)
│
├── results/                <- generated by running the scripts; empty at checkout
│   └── (base_model_cache.pkl, *.csv -- see "Running Experiments")
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

**Why this split?** `model/` is the actual recommender system --
everything needed to produce the thesis's headline results (Chapter 4)
starting from raw data. `experiments/` contains the supporting
hyperparameter studies and statistical tests that justify the choices
made in `model/` (Chapter 3's tables) -- these are not needed to
reproduce the headline results themselves, only to reproduce *why*
those particular hyperparameters were chosen. `plotting/` only ever
reads already-computed CSVs from `results/`; it never runs the model.

---

## Installation

### Requirements

- **Python 3.11** (the exact version this code was developed and
  tested against; see [Reproducibility](#reproducibility))
- A C compiler (needed to build `scikit-surprise`, which compiles a
  Cython extension on install) -- on macOS, install Xcode Command Line
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
repository as `data/useritemmatrix.csv` -- no separate download step
is required.

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
identical across runs on the same data file -- there is no separate
"split file" to manage. If you need the *exact* set of cold users used
by a particular run, load `results/base_model_cache.pkl` and read its
`cold_users` key (see [Reproducibility](#reproducibility) for what
else is inside that file).

### Expected files before running anything

| Path | Required before running |
|---|---|
| `data/useritemmatrix.csv` | Everything (this is the only required input file) |

Every other file under `results/` and `figures/` is generated by the
scripts in this repository -- see [Running Experiments](#running-experiments).

---

## Running Experiments

**Every script in this repository is run from the repository root**
(this folder), so that its relative paths to `data/`, `results/`, and
`figures/` resolve correctly. If cloned into a directory whose name
contains spaces (e.g. the default `Single Heuristic Personalized
(SHP)`), quote the path when changing into it:

```bash
cd "Single Heuristic Personalized (SHP)"
python scripts/model/build_model_cache.py     # NOT: cd scripts/model && python build_model_cache.py
```

### The order matters -- what depends on what

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
# STEP 1 -- one-time setup: trains the base SVD model, tunes ALPHA and
# (n_factors, reg_all, epochs). ~20-25 minutes. Writes
# results/base_model_cache.pkl, which every later step loads instead
# of redoing this setup.
python scripts/model/build_model_cache.py

# STEP 2 -- non-personalised baselines (Random, Popularity, PopError):
# RMSE + HR@K/NDCG@K on 1,000 cold users. ~8 minutes. Writes
# results/baseline_results.csv (thesis Table 4.1).
python scripts/model/baseline_ranking_metrics.py

# STEP 3 -- the four personalised strategies, full scale. Running the
# file directly only executes a 20-user smoke test (a quick
# correctness check); the full 1,000-user x 4-strategy x 4-k
# evaluation is only triggered via run_complete_pipeline.py (below) or
# by importing this module's run() function yourself. ~80-85 minutes
# for the full run. Writes results/personalised_results.csv (thesis
# Table 4.2, Table 4.3).
python scripts/model/personalised_strategies.py        # smoke test only
python scripts/model/run_complete_pipeline.py           # full pipeline: see below

# STEP 5 -- paired Wilcoxon significance tests, personalised vs. each
# baseline (48 tests). Needs STEP 3's output. ~16 minutes. Writes
# results/significance_results.csv.
python scripts/experiments/significance_test.py

# STEP 6 -- NDCG@10 significance test + Holm-Bonferroni/Benjamini-
# Hochberg multiple-comparisons correction. Needs STEP 5's output.
# ~16 minutes. Writes results/ranking_significance_results.csv and
# results/significance_results_corrected.csv (thesis Section 4.4).
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

### The optional hyperparameter ablations (STEP 4)

These reproduce Chapter 3's hyperparameter-selection tables. Each is
independent of the others and of STEPS 2/3/5/6 above (all read
`results/base_model_cache.pkl` directly) -- run any subset, in any
order:

```bash
python scripts/experiments/decaying_lr_test.py           # ~5 min   -- thesis Table 3.3
python scripts/experiments/shrinkage_test.py              # ~5 min   -- thesis Table 3.4
python scripts/experiments/b_ablation.py                  # ~10 min  -- Chapter 5, Limitation 6
python scripts/experiments/regularization_ablation.py     # ~5 min   -- Chapter 5, Limitation 6
python scripts/experiments/shecp_grid_search.py           # ~2 hours -- thesis Table 3.8 (full 1,000-user population; the only ablation run at full scale rather than the 200-user tuning subset)
python scripts/experiments/measure_update_cost.py         # ~1 min   -- thesis Section 3.7.1 (measured, not just asymptotic, full-retrain-vs-partial-update speedup)
```

### Figures (STEP 7, optional)

Run after the results CSVs they read from already exist:

```bash
python scripts/plotting/make_thesis_figures.py       # needs STEPS 2+3's output
python scripts/plotting/make_shecp_grid_figure.py     # needs shecp_grid_search.py's output (STEP 4)
```

---

## Reproducing Thesis Results

Every table and figure below is traceable to one exact script and one
exact output file, so a number in the thesis can always be checked
against a specific, re-runnable computation.

### Chapter 3 -- Methodology

| Thesis item | Script | Output file |
|---|---|---|
| Table 3.1 (dataset characteristics) | `scripts/model/build_model_cache.py` | printed to stdout during STEP 1 |
| Table 3.2 (notation) | -- (not a computed result) | -- |
| Table 3.3 (constant vs. decaying LR) | `scripts/experiments/decaying_lr_test.py` | `results/decaying_lr_test_results.csv` |
| Table 3.4 (shrinkage constant c search) | `scripts/experiments/shrinkage_test.py` | `results/shrinkage_test_results.csv` |
| Table 3.5 (PopError alpha search) | `scripts/model/build_model_cache.py` | printed to stdout (ALPHA line) |
| Table 3.6 (warm-user CV grid) | `scripts/model/build_model_cache.py` | printed to stdout (GridSearchCV line); full grid in `results/base_model_cache.pkl`'s `best_params` |
| Table 3.7 (SGD-steps ablation) | *(fixed at 1 in this repository; see* `NUM_SGD_STEPS` *in* `scripts/model/personalised_strategies.py`*. This value comes from the thesis's original ablation and is not re-derived by any script in this repository --* `decaying_lr_test.py` *always applies a single SGD step and does not vary this parameter.)* | -- |
| Table 3.8 (SHECP floor/decay grid) | `scripts/experiments/shecp_grid_search.py` | `results/shecp_grid_results.csv`; figure via `scripts/plotting/make_shecp_grid_figure.py` -> `figures/fig_shecp_grid.{png,pdf}` |
| Section 3.7.1 (measured partial-update speedup, ~6.5x10^6x) | `scripts/experiments/measure_update_cost.py` | `results/update_cost_results.txt` |
| Chapter 5, Limitation 6 (batch size B, regularisation lambda2) | `scripts/experiments/b_ablation.py`, `scripts/experiments/regularization_ablation.py` | `results/b_ablation_results.csv`, `results/regularization_ablation_results.csv` |

### Chapter 4 -- Preliminary Results

| Thesis item | Script | Output file |
|---|---|---|
| Table 4.1 (non-personalised baseline RMSE) | `scripts/model/baseline_ranking_metrics.py` | `results/baseline_results.csv` |
| Table 4.2 (personalised strategy RMSE) | `scripts/model/personalised_strategies.py` (full run, via `run_complete_pipeline.py`) | `results/personalised_results.csv` (`rmse` column, grouped by strategy/k) |
| Table 4.3 (HR@K/NDCG@K for personalised strategies) | same | `results/personalised_results.csv` (`hr5`, `hr10`, `ndcg5`, `ndcg10` columns) |
| Table 4.4 (best personalised vs. best baseline) | derived from Tables 4.1 + 4.2 | -- (no separate file; compare the two CSVs directly) |
| Figure 4.1 (RMSE vs. baseline range) | `scripts/plotting/make_thesis_figures.py` | `figures/fig_rmse_vs_baselines.{png,pdf}` |
| Figure 4.2 (HR@10/NDCG@10 vs. best baseline) | `scripts/plotting/make_thesis_figures.py` | `figures/fig_ranking_metrics.{png,pdf}` |
| Section 4.4 significance testing (raw 33/7/8 split) | `scripts/experiments/significance_test.py` | `results/significance_results.csv` |
| Section 4.4 Holm-corrected significance (28/48; 27 baseline-favouring, 1 personalised-favouring) | `scripts/experiments/ranking_significance_and_correction.py` | `results/significance_results_corrected.csv` |
| Section 4.4 NDCG@10 significance (23/48 Holm-significant, all baseline-favouring) | `scripts/experiments/ranking_significance_and_correction.py` | `results/ranking_significance_results.csv` |

### Sign convention note (important when reading the significance CSVs)

`rmse_margin` in `significance_results.csv` is defined as
`mean_rmse_baseline - mean_rmse_personalised`: since RMSE is
lower-is-better, a **positive** margin means the personalised strategy
is more accurate. `ndcg10_margin` in
`ranking_significance_results.csv` is defined as
`mean_ndcg10_baseline - mean_ndcg10_personalised`: since NDCG is
higher-is-better, a **positive** margin here means the *baseline* is
more accurate -- the opposite convention from the RMSE margin. Both
scripts' docstrings state this explicitly; double-check which metric
you are reading before interpreting a margin's sign.

---

## Methodology

### Biased SVD

The base recommender is a biased Singular Value Decomposition model
(as implemented in the [Surprise](http://surpriselib.com/) library),
predicting an interaction score as:

```
a_hat(u, i) = mu + b_u + b_i + p_u . q_i
```

where `mu` is the global mean interaction rate, `b_u`/`b_i` are
user/item bias terms, and `p_u`/`q_i` are the user's/item's latent
factor vectors. Trained **once**, on warm-user data only, via
leakage-free cross-validated hyperparameter search
(`scripts/model/build_model_cache.py`); frozen for the remainder of
every experiment.

### Partial SGD (incremental update)

For a cold user, only `p_u^c` (initialised at the first shown item's
own vector, not zero) and `b_u^c` (initialised to 0) are trainable;
the base model's `q_i`/`b_i` for every item stay frozen, except for a
**per-user local copy** that is updated only for items that specific
cold user has actually been shown (so one cold user's updates never
leak into another's evaluation). After each revealed interaction
`(u, i, r_ui)`, one step of gradient descent updates the four
quantities directly involved:

```
e_ui = r_ui - a_hat(u, i)
b_u^c <- b_u^c + gamma_1 (e_ui - lambda_1 b_u^c)
b_i   <- b_i   + gamma_1 (e_ui - lambda_1 b_i)
p_u^c <- p_u^c + gamma_2 (e_ui q_i - lambda_2 p_u^c)
q_i   <- q_i   + gamma_2 (e_ui p_u^c - lambda_2 q_i)
```

A **decaying learning rate**, `gamma_eff(t) = gamma_0 / sqrt(1+t)`
(`t` = number of prior updates this session), is applied on top of
this base update -- confirmed to improve validation RMSE in
`scripts/experiments/decaying_lr_test.py` (Robbins & Monro, 1951:
decreasing step sizes are a classical requirement for stochastic
approximation to converge rather than oscillate).

### Active learning (item selection)

At each step, the next batch of `B` items is chosen according to the
active strategy (SHHCP/SHLCP/SHMCP/SHECP -- see
[Research Contributions](#research-contributions)), using the *raw*
(non-shrunk) predicted score. See
`scripts/model/personalised_strategies.py`'s `_select_batch_vectorised`
(or the ablation scripts' simpler dict-based equivalents) for the
exact selection rule per strategy.

### Confidence-weighted shrinkage

At **evaluation** time only (never during item selection), the final
prediction blends the personalised term toward the stable baseline:

```
a_hat_shrunk(u, i) = mu + b_i + alpha(k) * (b_u^c + p_u^c . q_i),
    alpha(k) = k / (k + c)
```

where `k` is the total number of items revealed so far this session
and `c` is a tuned constant (`scripts/experiments/shrinkage_test.py`).

### Ranking evaluation: RMSE, HR@K, NDCG@K

- **RMSE** is computed on each cold user's held-out test split (the
  50% of their remaining unseen items not used for validation).
- **HR@K** and **NDCG@K** follow the sampled-candidate methodology of
  He et al. (2017): each held-out positive item is ranked against 99
  sampled negatives (`N_NEG` in the scripts), rather than the full
  item catalogue -- standard Recall@K is otherwise trivially inflated
  to near-unity on this dataset's sparsity and cannot distinguish
  between strategies. `HR@K = 1[rank(i+) <= K]`;
  `NDCG@K = HR@K / log2(rank(i+) + 1)`.

---

## Configuration

Every hyperparameter below is a module-level constant in the script
named in its "Where tuned" column below -- base-model hyperparameters
(`n_factors`, `reg_all`, `n_epochs`, `ALPHA`, `COLD_USER_FRACTION`,
`MIN_ITEM_INTERACTIONS`) live in `scripts/model/build_model_cache.py`;
cold-side/personalisation hyperparameters (`GAMMA1`, `GAMMA2`,
`LMBDA1`, `LMBDA2`, `NUM_SGD_STEPS`, `BATCH_SIZE`, `SHECP_FLOOR`,
`SHECP_DECAY`, `USE_DECAYING_LR`, `SHRINKAGE_C`, `N_NEG`) live in
`scripts/model/personalised_strategies.py`; and any value that was
swept in an ablation is mirrored as its own module-level constant in
the corresponding script under `scripts/experiments/`. Change the
constant, not the code that uses it.

| Hyperparameter | Value | Where tuned | Meaning |
|---|---|---|---|
| `n_factors` (latent dimension `F`) | 50 | `build_model_cache.py`, GridSearchCV | Number of latent factors in the base SVD |
| `reg_all` | 1e-4 | `build_model_cache.py`, GridSearchCV | Base-model L2 regularisation |
| `n_epochs` (base model) | 50 | fixed, following Geurts et al. (2020) | Base SVD training epochs |
| `ALPHA` (PopError mixing coefficient) | 0.7 | `build_model_cache.py`, HR@10 search | Popularity vs. ambiguity weight in PopError's score |
| `GAMMA1`, `GAMMA2` (learning rates) | 0.005, 0.005 | fixed, standard biased-SVD range | Bias / factor-vector update step size |
| `LMBDA1`, `LMBDA2` (regularisation) | 1e-7, 1e-6 | `regularization_ablation.py` (lambda2 swept; lambda1 held fixed) | Bias / factor-vector L2 penalty in the incremental update |
| `NUM_SGD_STEPS` | 1 | fixed, per the thesis's original ablation (not re-derived by any script in this repository; more steps overfit each noisy interaction) | Partial-SGD steps per revealed interaction |
| `BATCH_SIZE` (`B`) | 3 | `b_ablation.py` | Items revealed per active-learning round |
| `SHECP_FLOOR` | 0.05 | `shecp_grid_search.py`, validated on the full 1,000-user population | SHECP's minimum exploration probability |
| `SHECP_DECAY` | 0.95 | `shecp_grid_search.py` | SHECP's exploration-probability decay rate per round |
| `USE_DECAYING_LR` | True | `decaying_lr_test.py` | Whether the incremental update uses the decaying-LR schedule |
| `SHRINKAGE_C` | 100 | `shrinkage_test.py` | Confidence-weighted shrinkage constant |
| `COLD_USER_FRACTION` | 0.25 | fixed, following Geurts et al. (2020); not itself varied | Fraction of users withheld as cold |
| `MIN_ITEM_INTERACTIONS` | 10 | fixed | Minimum warm-user interactions for item eligibility |
| `N_NEG` | 99 | fixed, following He et al. (2017) | Sampled negatives per positive item in HR@K/NDCG@K |
| `NUM_EVAL_USERS` | 1000 | fixed | Cold users evaluated in the full-scale run |

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
  `hashlib.md5`-based deterministic seed -- used in place of Python's
  built-in `hash()`, which is randomised per-process
  (`PYTHONHASHSEED`) unless explicitly fixed, and would otherwise make
  every split non-reproducible both across separate runs *and* across
  the worker processes of a single parallel run.
- **SHECP grid search's epsilon-greedy draws**: a single
  `np.random.RandomState(123)` stream, deliberately *not*
  re-seeded per user, and the grid search itself deliberately *not*
  parallelised -- so the sequence of draws (and therefore the result)
  cannot depend on how many workers are used.

Re-running any script in this repository, on the same machine/library
versions, reproduces its output exactly.

### Software versions used to produce the thesis's results

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
population -- see [Random seeds](#random-seeds) above for why.

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

If you use this code or dataset in your own research, please cite:

```bibtex
@mastersthesis{tsai2026coldstart,
  author = {Tsai, Ying Ying},
  title  = {Confidence-Based Active Learning Strategies for Addressing
            the Cold User Problem in Recommender Systems Using
            Personalised Matrix Factorisation},
  school = {Erasmus University Rotterdam},
  year   = {2026},
  type   = {Master's thesis},
  note   = {Supervisor: F. Frasincar}
}
```

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
License** -- see [LICENSE](LICENSE). MIT was chosen as a permissive,
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
