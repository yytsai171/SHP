"""
hierarchical_regularization_test.py
======================================
Tests a regularization mechanism borrowed from Zhou, Yang & Zha's (2011)
Functional Matrix Factorization (fMF) paper: instead of shrinking a cold
user's latent vector purely toward zero (the existing lambda2 L2 term),
shrink it toward the mean latent vector of a "cohort" of warm users who
share a similar observed response pattern so far -- fMF's decision-tree
node aggregate `u_C`, adapted to SHP's per-user online setting since SHP
has no explicit shared decision tree across users.

Cohort definition (adapted for this dataset)
------------------------------------------------
fMF's own cohorts come from literal tree-node membership (same sequence
of Like/Dislike/Unknown answers). SHP has no equivalent shared structure
(different heuristics show different items to different users), so this
script uses the closest available substitute: whether the cold user has
observed at least one REAL negative (dislike/return) response so far
during elicitation. This dataset's warm-user population is heavily
skewed toward all-positive histories (median warm-user positive rate is
1.0 -- most warm users have never returned anything), so bucketing by
raw positive rate would be nearly degenerate. Bucketing by "has ever
returned an item" is far more balanced: 18.4% of warm users have >=1
recorded negative interaction, 81.6% do not. Three cohort targets are
precomputed once from warm-user data:

  - ``global_mean_pu``   : mean pu vector, ALL warm users (used before
    any real response has been observed this session).
  - ``cohort_allpos_pu`` : mean pu vector, warm users with ZERO recorded
    negatives (used once real responses have been observed but all are
    positive so far).
  - ``cohort_hasneg_pu`` : mean pu vector, warm users with >=1 recorded
    negative (used once at least one real negative has been observed).

Regularization mechanism
-----------------------------
Adds a hierarchical shrinkage term to the existing partial-SGD update,
analogous to fMF's `lambda_h * ||u - u_C||^2`:

    pu_new = pu_cold + gamma2_eff * (error*qi - lambda2*pu_cold
                                      - lambda_h*(pu_cold - cohort_target))

``lambda_h=0`` recovers the existing default exactly (pure zero-shrinkage
L2), so the grid below includes 0 as the reference point.

Tested against the ORIGINAL published defaults (fabricated-negatives
behaviour included, decaying LR, item-init, shrinkage C=100) -- the same
methodology every sibling single-factor ablation in this repo
(shrinkage_extended_test.py, joint_regularization_ablation.py,
cold_start_init_ablation.py) was tested against, so results are directly
comparable to those. Whether this interacts differently once combined
with the no-fabricated-negatives fix is a separate question, not
answered here.

Reference strategy: SHLCP, same 200-user tuning subset and k grid as the
sibling ablations.

Usage
-----
    python scripts/experiments/hierarchical_regularization_test.py

Input
-----
    results/base_model_cache.pkl

Output
------
    results/hierarchical_regularization_test_results.csv
        Columns: lambda_h, k, val_rmse, n_users, seconds
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from surprise import accuracy

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
MODEL_CACHE: str = os.path.join(RESULTS_DIR, 'base_model_cache.pkl')
OUT_HIER_REG: str = os.path.join(RESULTS_DIR, 'hierarchical_regularization_test_results.csv')

K_VALUES: List[int] = [10, 25, 50, 100]
LAMBDA_H_GRID: List[float] = [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1]


def _stable_seed(u: int, shown: List[Any]) -> int:
    """Deterministic replacement for Python's built-in ``hash()`` on
    strings -- see README.md "Reproducibility".
    """
    key = f"{int(u)}|" + ','.join(sorted(str(x) for x in shown))
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def main() -> None:
    """Runs the hierarchical-regularization lambda_h sweep for SHLCP
    across all four elicitation budgets and writes
    ``results/hierarchical_regularization_test_results.csv``.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.time()

    with open(MODEL_CACHE, 'rb') as f:
        cache = pickle.load(f)

    data             = cache['data']
    eligible_items   = cache['eligible_items']
    item_to_iidx     = cache['item_to_iidx']
    most_popular_iid = cache['most_popular_iid']
    i_0_inner        = cache['i_0_inner']
    svd_base         = cache['svd_base']
    mu_base          = cache['mu_base']
    n_factors        = cache['n_factors']
    cold_users       = cache['cold_users']
    GAMMA1, GAMMA2   = cache['GAMMA1'], cache['GAMMA2']
    LMBDA1, LMBDA2   = cache['LMBDA1'], cache['LMBDA2']

    print(f"[{time.time()-t_start:6.1f}s] Cache loaded (setup skipped).", flush=True)

    # ── Precompute the three cohort targets from warm-user data ───────────
    warm_data = data[~data['user_idx'].isin(cold_users)]
    warm_users = warm_data['user_idx'].unique()
    warm_has_neg = warm_data.groupby('user_idx')['interaction'].min() == 0

    def _warm_pu(uid_raw: int) -> Optional[np.ndarray]:
        try:
            inner = svd_base.trainset.to_inner_uid(uid_raw)
        except ValueError:
            return None
        return svd_base.pu[inner]

    all_pu = np.array([p for p in (_warm_pu(u) for u in warm_users) if p is not None])
    hasneg_uids = warm_has_neg[warm_has_neg].index
    allpos_uids = warm_has_neg[~warm_has_neg].index
    hasneg_pu = np.array([p for p in (_warm_pu(u) for u in hasneg_uids) if p is not None])
    allpos_pu = np.array([p for p in (_warm_pu(u) for u in allpos_uids) if p is not None])

    global_mean_pu = all_pu.mean(axis=0)
    cohort_hasneg_pu = hasneg_pu.mean(axis=0)
    cohort_allpos_pu = allpos_pu.mean(axis=0)
    print(f"[{time.time()-t_start:6.1f}s] Cohort targets precomputed: "
          f"{len(all_pu)} warm users total ({len(hasneg_pu)} has-negative, "
          f"{len(allpos_pu)} all-positive).", flush=True)

    eval_cold_users = cold_users[:min(1000, len(cold_users))]
    hp_search_users = eval_cold_users[:min(200, len(eval_cold_users))]

    Pred = namedtuple('Prediction', ['uid', 'iid', 'r_ui', 'est', 'details'])

    def split_unseen_items(u: int, shown: List[Any],
                            val_frac: float = 0.5) -> Tuple[List[Any], List[Any]]:
        shown_set = set(shown)
        unseen = [iid for iid in eligible_items if iid not in shown_set]
        rng = np.random.RandomState(_stable_seed(u, shown))
        rng.shuffle(unseen)
        n_val = int(val_frac * len(unseen))
        return unseen[:n_val], unseen[n_val:]

    def cohort_target(n_real_seen: int, n_real_neg: int) -> np.ndarray:
        """Selects which of the three precomputed cohort means to shrink
        toward, given the real (non-fabricated) responses observed in
        this session so far."""
        if n_real_seen == 0:
            return global_mean_pu
        return cohort_hasneg_pu if n_real_neg > 0 else cohort_allpos_pu

    def partial_lfm_update_cold_hier(
        svd_model: Any, pu_cold: np.ndarray, bu_cold: float, i_inner: int, r_ui: float,
        local_qi: Dict[int, np.ndarray], local_bi: Dict[int, float],
        gamma1_eff: float, gamma2_eff: float, lmbda1: float, lmbda2: float,
        lambda_h: float, target: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """One decaying-LR partial-SGD update with an added hierarchical
        shrinkage term pulling ``pu_cold`` toward ``target`` (fMF Section
        3.5's `lambda_h * ||u - u_C||^2`, adapted to this per-step
        online setting). ``lambda_h=0`` recovers the existing default
        update exactly."""
        if i_inner not in local_qi:
            local_qi[i_inner] = svd_model.qi[i_inner].copy()
            local_bi[i_inner] = float(svd_model.bi[i_inner])
        mu  = svd_model.trainset.global_mean
        qi  = local_qi[i_inner]
        bi  = local_bi[i_inner]
        pred  = mu + bu_cold + bi + np.dot(pu_cold, qi)
        error = r_ui - pred
        bu_cold           = bu_cold + gamma1_eff * (error - lmbda1 * bu_cold)
        local_bi[i_inner] = bi      + gamma1_eff * (error - lmbda1 * bi)
        pu_new = pu_cold + gamma2_eff * (
            error * qi - lmbda2 * pu_cold - lambda_h * (pu_cold - target)
        )
        local_qi[i_inner] = qi + gamma2_eff * (error * pu_cold - lmbda2 * qi)
        return pu_new, bu_cold

    def select_batch_items_cold(pu_cold, bu_cold, shown, local_qi, local_bi, batch_size=3):
        mu        = svd_base.trainset.global_mean
        shown_set = set(shown)
        scores    = {}
        for iid in eligible_items:
            if iid in shown_set:
                continue
            i_inner = item_to_iidx.get(iid)
            if i_inner is None:
                continue
            qi          = local_qi.get(i_inner, svd_base.qi[i_inner])
            bi          = local_bi.get(i_inner, svd_base.bi[i_inner])
            scores[iid] = mu + bu_cold + bi + np.dot(pu_cold, qi)
        if not scores:
            return []
        b = min(batch_size, len(scores))
        return sorted(scores, key=scores.get)[:b]

    def run_session(u: int, k: int, lambda_h: float):
        if i_0_inner is not None:
            pu_cold = svd_base.qi[i_0_inner].copy()
        else:
            pu_cold = np.zeros(n_factors)
        bu_cold  = 0.0
        local_qi, local_bi = {}, {}
        shown = [most_popular_iid]
        n = 0
        n_real_seen = 0
        n_real_neg = 0

        first_row = data[(data['user_idx'] == u) & (data['itemId'] == most_popular_iid)]
        has_first = len(first_row) > 0
        r_first   = float(first_row['interaction'].iloc[0]) if has_first else 0.0
        if i_0_inner is not None:
            decay  = 1.0 / np.sqrt(1.0 + n)
            target = cohort_target(n_real_seen, n_real_neg)
            pu_cold, bu_cold = partial_lfm_update_cold_hier(
                svd_base, pu_cold, bu_cold, i_0_inner, r_first, local_qi, local_bi,
                GAMMA1 * decay, GAMMA2 * decay, LMBDA1, LMBDA2, lambda_h, target
            )
            n += 1
            if has_first:
                n_real_seen += 1
                if r_first == 0.0:
                    n_real_neg += 1

        while len(shown) < k:
            b     = min(3, k - len(shown))
            batch = select_batch_items_cold(pu_cold, bu_cold, shown, local_qi, local_bi, b)
            if not batch:
                break
            shown.extend(batch)
            for item in batch:
                row  = data[(data['user_idx'] == u) & (data['itemId'] == item)]
                has_row = len(row) > 0
                r_ui = float(row['interaction'].iloc[0]) if has_row else 0.0
                i_inner = item_to_iidx.get(item)
                if i_inner is not None:
                    decay  = 1.0 / np.sqrt(1.0 + n)
                    target = cohort_target(n_real_seen, n_real_neg)
                    pu_cold, bu_cold = partial_lfm_update_cold_hier(
                        svd_base, pu_cold, bu_cold, i_inner, r_ui, local_qi, local_bi,
                        GAMMA1 * decay, GAMMA2 * decay, LMBDA1, LMBDA2, lambda_h, target
                    )
                    n += 1
                    if has_row:
                        n_real_seen += 1
                        if r_ui == 0.0:
                            n_real_neg += 1
        return pu_cold, bu_cold, local_qi, local_bi, shown

    def evaluate_session_rmse(
        pu_cold: np.ndarray, bu_cold: float, local_qi: Dict[int, np.ndarray],
        local_bi: Dict[int, float], shown: List[Any], test_items: List[Any], u: int
    ) -> Optional[float]:
        shown_set = set(shown)
        test_df = data[(data['user_idx'] == u) &
                       (data['itemId'].isin(test_items)) &
                       (~data['itemId'].isin(shown_set))]
        if len(test_df) == 0:
            return None
        preds = []
        for row in test_df.itertuples():
            i_inner = item_to_iidx.get(row.itemId)
            if i_inner is None:
                continue
            qi  = local_qi.get(i_inner, svd_base.qi[i_inner])
            bi  = local_bi.get(i_inner, svd_base.bi[i_inner])
            est = float(np.clip(mu_base + bu_cold + bi + np.dot(pu_cold, qi), 0, 1))
            preds.append(Pred(uid=row.user_idx, iid=row.item_idx,
                               r_ui=row.interaction, est=est, details={}))
        if not preds:
            return None
        return accuracy.rmse(preds, verbose=False)

    rows = []
    for k in K_VALUES:
        print(f"\n=== k={k}: {len(hp_search_users)} users x {len(LAMBDA_H_GRID)} "
              f"lambda_h settings ===", flush=True)
        t_k = time.time()

        for lambda_h in LAMBDA_H_GRID:
            t_cell = time.time()
            rmses = []
            for u in hp_search_users:
                pu_cold, bu_cold, local_qi, local_bi, shown = run_session(u, k, lambda_h)
                val_items, _ = split_unseen_items(u, shown, val_frac=0.5)
                r = evaluate_session_rmse(pu_cold, bu_cold, local_qi, local_bi, shown, val_items, u)
                if r is not None:
                    rmses.append(r)
            avg = np.mean(rmses) if rmses else float('nan')
            rows.append({'lambda_h': lambda_h, 'k': k, 'val_rmse': round(avg, 4),
                         'n_users': len(rmses), 'seconds': round(time.time() - t_cell, 1)})
            print(f"  [lambda_h={lambda_h:.0e}] val RMSE={avg:.4f} "
                  f"(n={len(rmses)}, {time.time()-t_cell:.1f}s)", flush=True)

        print(f"  k={k} total time: {time.time()-t_k:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_HIER_REG, index=False)

    print(f"\n{'='*70}", flush=True)
    print("=== Best lambda_h per k (lowest val RMSE) ===", flush=True)
    for k in K_VALUES:
        dfk = df[df['k'] == k]
        best_row = dfk.loc[dfk['val_rmse'].idxmin()]
        default_row = dfk[dfk['lambda_h'] == 0.0].iloc[0]
        print(f"  k={k}: best lambda_h={best_row['lambda_h']:.0e}  RMSE={best_row['val_rmse']:.4f}  "
              f"(default lambda_h=0: RMSE={default_row['val_rmse']:.4f})", flush=True)

    print(f"\nSaved to {OUT_HIER_REG}", flush=True)
    print(f"TOTAL TIME: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
