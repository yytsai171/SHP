"""
build_model_cache.py
=====================
Setup script for the cold-user active-learning pipeline.

Loads the raw interaction dataset, partitions users into cold/warm
subsets, tunes the PopError mixing coefficient (ALPHA), runs the
leakage-free warm-user-only hyperparameter search for the base biased-SVD
model, trains that model, and saves everything downstream scripts need
to ``results/base_model_cache.pkl``.

Why this script exists
-----------------------
None of the downstream scripts (the four personalised strategies, the
three non-personalised baselines, every ablation) need to redo the
expensive setup phase -- 5-fold cross-validated grid search over 9
``(n_factors, reg_all)`` configurations on ~1.9M warm-user interaction
rows. They all load this cache instead, turning a ~20-minute setup into
a sub-second pickle load.

Reproducibility
----------------
Deterministic given the fixed seeds (``random.seed(1)``,
``np.random.seed(1)``, ``GridSearchCV(..., n_jobs=1)``); re-running this
script reproduces the identical base model every time on the same
machine/library versions (see README.md "Reproducibility").

Thesis reference
------------------
Corresponds to thesis Section 3.3 (Experimental Setup: the cold/warm
split), Section 3.4 (item eligibility), and the "Leakage-free warm-user
cross-validation" paragraph of Section 3.8 (Evaluation Metrics and
Hyperparameter Tuning).

Usage
-----
    python scripts/model/build_model_cache.py

Input
-----
    data/useritemmatrix.csv
        Columns: userId, itemId, interaction (binary 0/1).
        See README.md "Dataset".

Output
------
    results/base_model_cache.pkl
        A pickled dict with keys:
        data, eligible_items, item_to_iidx, most_popular_iid, i_0_inner,
        svd_base, mu_base, n_factors, best_params, ALPHA, cold_users,
        GAMMA1, GAMMA2, LMBDA1, LMBDA2.

Runtime
-------
    ~20-25 minutes on a single CPU core (GridSearchCV runs with
    n_jobs=1 deliberately, to keep the search itself reproducible --
    see README.md "Reproducibility").
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import GridSearchCV, KFold
from surprise.trainset import Trainset

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_PATH: str = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'useritemmatrix.csv')
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
MODEL_CACHE: str = os.path.join(RESULTS_DIR, 'base_model_cache.pkl')

# Fraction of users withheld as "cold" (never seen during base-model
# training). Fixed at 0.25 following Geurts et al. (2020); not tuned --
# see thesis Chapter 5, Limitation 7.
COLD_USER_FRACTION: float = 0.25

# Minimum warm-user interaction count for an item to be "eligible"
# (shown to cold users / used as a candidate). Thesis Section 3.3.
MIN_ITEM_INTERACTIONS: int = 10

# PopError alpha search grid and validation-sample size. Thesis Table 3.5.
ALPHA_GRID: List[float] = [0.5, 0.7, 0.9]
N_ALPHA_VAL: int = 200

# Number of sampled negatives per positive item in the HR@K evaluation
# used only for the ALPHA search here (He et al., 2017 methodology;
# thesis Section 3.9 "Evaluation Metrics").
N_NEG: int = 99

# Base-model hyperparameter grid (thesis Table 3.6). n_epochs is fixed
# at 50 following Geurts et al. (2020); only n_factors and reg_all vary.
# random_state is fixed so every candidate model's internal (pu, qi, bu,
# bi) initialisation is seeded -- Surprise's SVD falls back to NumPy's
# *global* RNG when random_state is left at its default (None), which
# would otherwise make GridSearchCV's winner non-reproducible across
# runs regardless of n_jobs (see README.md "Reproducibility").
SVD_PARAM_GRID: Dict[str, List[Any]] = {
    'n_factors': [50, 100, 200],
    'reg_all': [1e-7, 1e-6, 1e-4],
    'n_epochs': [50],
    'biased': [True],
    'random_state': [1],
}

# Cold-user incremental-update hyperparameters (thesis Eq. 3.9-3.10).
# Not tuned by this script -- see decaying_lr_test.py, shrinkage_test.py,
# regularization_ablation.py for the ablations that inform these values.
GAMMA1: float = 0.005  # learning rate, bias update
GAMMA2: float = 0.005  # learning rate, factor-vector update
LMBDA1: float = 1e-7   # L2 regularisation, bias update
LMBDA2: float = 1e-6   # L2 regularisation, factor-vector update


def load_and_split_data(data_path: str, cold_fraction: float,
                         min_item_interactions: int,
                         rng: np.random.RandomState
                         ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame,
                                    pd.Series, List[Any], Any]:
    """Loads the raw interaction CSV and partitions users into cold/warm.

    Parameters
    ----------
    data_path : str
        Path to the ``useritemmatrix.csv`` file (columns: userId,
        itemId, interaction).
    cold_fraction : float
        Fraction of unique users to withhold as cold users.
    min_item_interactions : int
        Minimum warm-user interaction count for an item to be eligible.
    rng : np.random.RandomState
        Random state used for the cold/warm split. Must be pre-seeded
        by the caller for reproducibility.

    Returns
    -------
    data : pd.DataFrame
        Full dataset with added ``user_idx``/``item_idx`` integer
        category codes.
    cold_users : np.ndarray
        Integer array of ``user_idx`` values selected as cold users.
    warm_data : pd.DataFrame
        Subset of ``data`` excluding cold users.
    item_counts : pd.Series
        Warm-user interaction count per raw ``itemId``, descending
        eligibility relevance.
    eligible_items : list
        Raw ``itemId`` values with at least ``min_item_interactions``
        warm-user interactions.
    most_popular_iid : Any
        The single most-interacted-with eligible item (raw itemId),
        used as every cold user's first shown item.

    Notes
    -----
    Runtime is O(n) in the number of interaction rows for the groupby/
    category-code operations. Corresponds to thesis Section 3.3
    (cold/warm split) and Section 3.4 (item eligibility).
    """
    data = pd.read_csv(data_path)
    data = data.groupby('userId').filter(lambda x: len(x) > 0)
    data['user_idx'] = data['userId'].astype('category').cat.codes
    data['item_idx'] = data['itemId'].astype('category').cat.codes

    all_users = data['user_idx'].unique()
    cold_users = rng.choice(
        all_users, size=int(len(all_users) * cold_fraction), replace=False
    )

    warm_data = data[~data['user_idx'].isin(cold_users)]
    item_counts = warm_data['itemId'].value_counts()
    eligible_items = item_counts[item_counts >= min_item_interactions].index.tolist()
    most_popular_iid = item_counts.idxmax()

    return data, cold_users, warm_data, item_counts, eligible_items, most_popular_iid


def compute_misclassification_error_scores(
    warm_data: pd.DataFrame, eligible_items: List[Any]
) -> Dict[Any, float]:
    """Computes each eligible item's misclassification-error score.

    MisclassError(i) = 1 - max(P(like | i), P(dislike | i)), i.e. how
    close the item's warm-user interaction rate is to 50/50 (maximum
    ambiguity) versus lopsidedly positive or negative (low ambiguity).
    Used as the ambiguity component of the PopError score
    (thesis Eq. 3.3).

    Parameters
    ----------
    warm_data : pd.DataFrame
        Interactions restricted to warm users (must contain columns
        'itemId', 'interaction').
    eligible_items : list
        Raw itemId values to compute a score for.

    Returns
    -------
    dict
        Mapping from raw itemId to its misclassification-error score
        in [0, 0.5].

    Complexity
    ----------
    O(n) in the number of warm-user interaction rows for the groupby,
    then O(|eligible_items|) for the per-item score computation.
    """
    item_mean_interaction = warm_data.groupby('itemId')['interaction'].mean()
    error_scores: Dict[Any, float] = {}
    for item in eligible_items:
        p = float(item_mean_interaction.get(item, 0.5))
        error_scores[item] = min(p, 1.0 - p)
    return error_scores


def _sampled_hr_at_k(pos_score: float, neg_scores: List[float],
                      k_list: List[int]) -> Dict[int, float]:
    """Computes HR@K for one positive item against sampled negatives.

    Parameters
    ----------
    pos_score : float
        The positive item's predicted score.
    neg_scores : list of float
        Predicted scores of the sampled negative items.
    k_list : list of int
        Cutoff values K to evaluate.

    Returns
    -------
    dict
        Mapping from K to 1.0 (positive ranked in the top K) or 0.0.

    Notes
    -----
    Rank is computed as ``1 + count(neg_scores > pos_score)``, i.e.
    ties are resolved in the positive item's favour (rank-1 = best).
    """
    rank = 1 + sum(1 for s in neg_scores if s > pos_score)
    return {k: (1.0 if rank <= k else 0.0) for k in k_list}


def learn_alpha(warm_data: pd.DataFrame, item_counts: pd.Series,
                 eligible_items: List[Any], error_scores: Dict[Any, float],
                 alpha_grid: List[float], n_val_users: int, n_neg: int,
                 rng: np.random.RandomState) -> tuple[float, float]:
    """Selects PopError's mixing coefficient ALPHA by HR@10 on warm users.

    PopError(i) = alpha * log10(freq(i)) + (1 - alpha) * MisclassError(i)
    (thesis Eq. 3.2). ALPHA is evaluated by HR@10 rather than RMSE
    because it governs which items PopError *selects*, not a predicted
    rating -- see README.md "Methodology" for the full justification
    (thesis Section 3.8, "Search space" paragraph).

    Parameters
    ----------
    warm_data : pd.DataFrame
        Interactions restricted to warm users.
    item_counts : pd.Series
        Warm-user interaction count per raw itemId.
    eligible_items : list
        Raw itemId values eligible for selection.
    error_scores : dict
        Output of ``compute_misclassification_error_scores``.
    alpha_grid : list of float
        Candidate ALPHA values to evaluate.
    n_val_users : int
        Number of warm users to sample for validation.
    n_neg : int
        Number of sampled negatives per positive item (He et al., 2017).
    rng : np.random.RandomState
        Random state for user sampling and negative sampling. Must be
        pre-seeded by the caller for reproducibility.

    Returns
    -------
    best_alpha : float
        The ALPHA_GRID value achieving the highest mean HR@10.
    best_alpha_hr : float
        That value's mean HR@10.

    Complexity
    ----------
    O(|alpha_grid| * n_val_users * n_neg) predicted-score comparisons.
    """
    eligible_set = set(eligible_items)
    warm_users_all = warm_data['user_idx'].unique()
    n_alpha_sample = min(n_val_users, len(warm_users_all))
    alpha_val_users = rng.choice(warm_users_all, size=n_alpha_sample, replace=False)

    alpha_val_set = set(alpha_val_users.tolist())
    warm_alpha_df = warm_data[warm_data['user_idx'].isin(alpha_val_set)]
    warm_alpha_grouped = {u: grp for u, grp in warm_alpha_df.groupby('user_idx')}

    best_alpha = alpha_grid[0]
    best_alpha_hr = -1.0

    for alpha_candidate in alpha_grid:
        pop_scores_alpha = {
            item: alpha_candidate * np.log10(item_counts[item])
                  + (1 - alpha_candidate) * error_scores[item]
            for item in eligible_items
        }
        hr10_vals = []
        for u in alpha_val_users:
            u_data = warm_alpha_grouped.get(u)
            if u_data is None:
                continue
            u_interacted = set(u_data['itemId'].tolist())
            u_pos_df = u_data[(u_data['interaction'] == 1) &
                               (u_data['itemId'].isin(eligible_set))]
            pos_items = u_pos_df['itemId'].tolist()
            if not pos_items:
                continue
            cand_negs = [iid for iid in eligible_items if iid not in u_interacted]
            user_hr10 = []
            for pos_iid in pos_items:
                n_sample = min(n_neg, len(cand_negs))
                if n_sample == 0:
                    continue
                sampled = rng.choice(cand_negs, size=n_sample, replace=False)
                m = _sampled_hr_at_k(
                    pop_scores_alpha[pos_iid],
                    [pop_scores_alpha[nid] for nid in sampled],
                    k_list=[10],
                )
                user_hr10.append(m[10])
            if user_hr10:
                hr10_vals.append(float(np.mean(user_hr10)))
        avg_hr10 = float(np.mean(hr10_vals)) if hr10_vals else 0.0
        if avg_hr10 > best_alpha_hr:
            best_alpha_hr = avg_hr10
            best_alpha = alpha_candidate

    return best_alpha, best_alpha_hr


def tune_and_train_base_model(
    warm_data: pd.DataFrame, param_grid: Dict[str, List[Any]]
) -> tuple[Dict[str, Any], SVD]:
    """Runs leakage-free warm-user-only GridSearchCV, then trains the
    final base biased-SVD model with the winning configuration on the
    full warm-user set.

    Leakage-free protocol: the 5-fold cross-validation is run
    exclusively on warm-user data, so cold-user behaviour never
    influences hyperparameter selection (thesis Section 3.8,
    "Leakage-free warm-user cross-validation").

    Parameters
    ----------
    warm_data : pd.DataFrame
        Interactions restricted to warm users (columns: user_idx,
        item_idx, interaction).
    param_grid : dict
        Grid search space, e.g. ``SVD_PARAM_GRID``.

    Returns
    -------
    best_params : dict
        The winning (n_factors, reg_all, n_epochs, biased) configuration.
    svd_base : surprise.SVD
        The trained base model, fit on all warm-user data with
        ``best_params``.

    Complexity
    ----------
    GridSearchCV: O(|param_grid| * cv_folds) SVD training runs, each
    O(n_factors * n_interactions * n_epochs). Run with n_jobs=1
    (sequential) for reproducibility -- see README.md "Reproducibility".

    Notes
    -----
    ``cv`` is passed as an explicit, seeded ``KFold(random_state=1)``
    rather than the bare integer ``5``: Surprise expands an integer
    ``cv`` into ``KFold(n_splits=cv)``, whose default
    ``random_state=None`` reshuffles the 5 folds on every run,
    independently of ``n_jobs``. Combined with ``SVD_PARAM_GRID``'s
    fixed ``random_state``, this makes the winning
    ``(n_factors, reg_all)`` reproducible across runs.
    """
    reader = Reader(rating_scale=(0, 1))
    warm_dataset = Dataset.load_from_df(
        warm_data[['user_idx', 'item_idx', 'interaction']], reader
    )

    cv = KFold(n_splits=5, random_state=1)
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=cv, n_jobs=1)
    gs.fit(warm_dataset)
    best_params = gs.best_params['rmse']

    base_dataset = Dataset.load_from_df(
        warm_data[['user_idx', 'item_idx', 'interaction']], reader
    )
    base_trainset: Trainset = base_dataset.build_full_trainset()

    svd_base = SVD(
        n_factors=best_params['n_factors'],
        reg_all=best_params['reg_all'],
        n_epochs=best_params['n_epochs'],
        biased=True,
        random_state=1,
    )
    svd_base.fit(base_trainset)

    return best_params, svd_base


def build_item_index_map(
    data: pd.DataFrame, eligible_items: List[Any], svd_base: SVD
) -> Dict[Any, int]:
    """Maps each eligible item's raw itemId to its inner index in the
    trained Surprise trainset, skipping items the trainset never saw
    (e.g. an eligible item with zero warm-user positive interactions
    would still appear in `eligible_items` but may be absent from
    `_raw2inner_id_items` if it never appears as a rated pair).

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset (used to recover each itemId's raw category code).
    eligible_items : list
        Raw itemId values to map.
    svd_base : surprise.SVD
        The trained base model (provides trainset raw/inner id lookup).

    Returns
    -------
    dict
        Mapping from raw itemId to Surprise inner item index.
    """
    item_to_raw_idx = (
        data[['itemId', 'item_idx']]
        .drop_duplicates('itemId')
        .set_index('itemId')['item_idx']
        .to_dict()
    )
    item_to_iidx: Dict[Any, int] = {}
    for iid in eligible_items:
        raw_idx = item_to_raw_idx.get(iid)
        if raw_idx is not None and raw_idx in svd_base.trainset._raw2inner_id_items:
            item_to_iidx[iid] = svd_base.trainset.to_inner_iid(raw_idx)
    return item_to_iidx


def main() -> None:
    """Runs the full setup pipeline and writes ``results/base_model_cache.pkl``.

    Note on RNG handling: the cold/warm split uses a freshly-constructed
    ``np.random.RandomState(1)`` rather than the legacy global
    ``np.random.seed(1)`` + module-level ``np.random.choice`` pattern --
    both draw from an identically-seeded Mersenne Twister as the very
    first draw of the stream, so this produces byte-identical output to
    the original implementation while making the RNG dependency explicit
    (see README.md "Reproducibility").
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    t_start = time.time()

    # ---- 1. Data loading and cold/warm split (thesis Section 3.3) ----
    split_rng = np.random.RandomState(1)
    data, cold_users, warm_data, item_counts, eligible_items, most_popular_iid = \
        load_and_split_data(DATA_PATH, COLD_USER_FRACTION, MIN_ITEM_INTERACTIONS,
                             rng=split_rng)
    print(f"[{time.time()-t_start:6.1f}s] Dataset loaded. "
          f"Cold users: {len(cold_users):,}  Eligible items: {len(eligible_items):,}",
          flush=True)

    # ---- 2. Misclassification-error scores (thesis Eq. 3.3) ----
    error_scores = compute_misclassification_error_scores(warm_data, eligible_items)

    # ---- 3. Learn ALPHA (thesis Table 3.5) ----
    alpha_val_rng = np.random.RandomState(99)
    ALPHA, best_alpha_hr = learn_alpha(
        warm_data, item_counts, eligible_items, error_scores,
        ALPHA_GRID, N_ALPHA_VAL, N_NEG, alpha_val_rng
    )
    print(f"[{time.time()-t_start:6.1f}s] ALPHA learned: {ALPHA} (HR@10={best_alpha_hr:.4f})",
          flush=True)

    # ---- 4-5. Hyperparameter tuning + base model training (thesis Table 3.6) ----
    best_params, svd_base = tune_and_train_base_model(warm_data, SVD_PARAM_GRID)
    n_factors = svd_base.pu.shape[1]
    mu_base = float(svd_base.trainset.global_mean)
    print(f"[{time.time()-t_start:6.1f}s] GridSearchCV done. Best params: {best_params}",
          flush=True)

    item_to_iidx = build_item_index_map(data, eligible_items, svd_base)
    i_0_inner = item_to_iidx.get(most_popular_iid)
    print(f"[{time.time()-t_start:6.1f}s] Base SVD trained. "
          f"n_factors={n_factors}, items indexed={len(item_to_iidx)}", flush=True)

    # ---- Save cache ----
    with open(MODEL_CACHE, 'wb') as f:
        pickle.dump({
            'data': data,
            'eligible_items': eligible_items,
            'item_to_iidx': item_to_iidx,
            'most_popular_iid': most_popular_iid,
            'i_0_inner': i_0_inner,
            'svd_base': svd_base,
            'mu_base': mu_base,
            'n_factors': n_factors,
            'best_params': best_params,
            'ALPHA': ALPHA,
            'cold_users': cold_users,
            'GAMMA1': GAMMA1, 'GAMMA2': GAMMA2,
            'LMBDA1': LMBDA1, 'LMBDA2': LMBDA2,
        }, f)

    print(f"[{time.time()-t_start:6.1f}s] Cache saved to {MODEL_CACHE}", flush=True)
    print(f"TOTAL SETUP TIME: {time.time()-t_start:.1f} seconds", flush=True)


if __name__ == '__main__':
    main()
