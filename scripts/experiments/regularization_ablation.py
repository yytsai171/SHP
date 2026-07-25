"""
regularization_ablation.py
=============================
Ablation: sweeps the factor-vector regularisation coefficient lambda2
(holding the bias regularisation lambda1 fixed at its default), to
check whether stronger regularisation of the cold-user latent vector
p_u^c reduces RMSE growth at large elicitation budgets k.

Motivation: lambda2 is inherited from Geurts et al. (2020)'s full-batch
warm-user setting and was never re-tuned for the sparse, few-shot,
partial-update regime cold users are actually in. Since personalised
RMSE tends to worsen as k grows (per-interaction noise compounding
rather than averaging out), stronger regularisation should pull p_u^c
back toward zero (the safe baseline) exactly when there is too little
evidence to trust it -- and should show up as reduced RMSE growth at
large k specifically.

Reference strategy: SHLCP, probed at k in {10, 100} (the two extremes),
on the 200-user hyperparameter-tuning subset.

Usage
-----
    python scripts/experiments/regularization_ablation.py

Input
-----
    results/base_model_cache.pkl

Output
------
    results/regularization_ablation_results.csv
        Columns: lambda2, k, val_rmse, n_users, seconds
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
OUT_REG_ABL: str = os.path.join(RESULTS_DIR, 'regularization_ablation_results.csv')

LAMBDA2_GRID: List[float] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # default is 1e-6
K_PROBES: List[int] = [10, 100]


def _stable_seed(u: int, shown: List[Any]) -> int:
    """Deterministic replacement for Python's built-in ``hash()`` on
    strings -- see README.md "Reproducibility".
    """
    key = f"{int(u)}|" + ','.join(sorted(str(x) for x in shown))
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def main() -> None:
    """Runs the lambda2 regularisation-strength ablation for SHLCP at
    k in {10, 100} and writes ``results/regularization_ablation_results.csv``.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.time()

    with open(MODEL_CACHE, 'rb') as f:
        cache = pickle.load(f)

    data = cache['data']
    eligible_items = cache['eligible_items']
    item_to_iidx = cache['item_to_iidx']
    most_popular_iid = cache['most_popular_iid']
    i_0_inner = cache['i_0_inner']
    svd_base = cache['svd_base']
    mu_base = cache['mu_base']
    n_factors = cache['n_factors']
    cold_users = cache['cold_users']
    GAMMA1, GAMMA2 = cache['GAMMA1'], cache['GAMMA2']
    LMBDA1_DEFAULT = cache['LMBDA1']   # held fixed in this ablation

    print(f"[{time.time()-t_start:6.1f}s] Cache loaded (setup skipped).", flush=True)

    eval_cold_users = cold_users[:min(1000, len(cold_users))]
    hp_search_users = eval_cold_users[:min(200, len(eval_cold_users))]

    Pred = namedtuple('Prediction', ['uid', 'iid', 'r_ui', 'est', 'details'])

    def split_unseen_items(u: int, shown: List[Any],
                            val_frac: float = 0.5) -> Tuple[List[Any], List[Any]]:
        """Validation/test split, deterministically seeded per (u, shown)."""
        shown_set = set(shown)
        unseen = [iid for iid in eligible_items if iid not in shown_set]
        rng = np.random.RandomState(_stable_seed(u, shown))
        rng.shuffle(unseen)
        n_val = int(val_frac * len(unseen))
        return unseen[:n_val], unseen[n_val:]

    def partial_lfm_update_cold(
        svd_model: Any, pu_cold: np.ndarray, bu_cold: float, i_inner: int, r_ui: float,
        local_qi: Dict[int, np.ndarray], local_bi: Dict[int, float],
        gamma1: float, gamma2: float, lmbda1: float, lmbda2: float,
        num_sgd_steps: int = 1
    ) -> Tuple[np.ndarray, float]:
        """One partial-SGD update, with ``lmbda2`` (the value under test
        in this ablation) applied to the factor-vector regularisation
        term (thesis Eq. 3.9-3.10)."""
        if i_inner not in local_qi:
            local_qi[i_inner] = svd_model.qi[i_inner].copy()
            local_bi[i_inner] = float(svd_model.bi[i_inner])
        mu = svd_model.trainset.global_mean
        for _ in range(num_sgd_steps):
            qi = local_qi[i_inner]
            bi = local_bi[i_inner]
            pred = mu + bu_cold + bi + np.dot(pu_cold, qi)
            error = r_ui - pred
            bu_cold = bu_cold + gamma1 * (error - lmbda1 * bu_cold)
            local_bi[i_inner] = bi + gamma1 * (error - lmbda1 * bi)
            pu_new = pu_cold + gamma2 * (error * qi - lmbda2 * pu_cold)
            local_qi[i_inner] = qi + gamma2 * (error * pu_cold - lmbda2 * qi)
            pu_cold = pu_new
        return pu_cold, bu_cold

    def select_batch_items_cold(
        svd_model: Any, pu_cold: np.ndarray, bu_cold: float, shown: List[Any],
        eligible_items_pool: List[Any], item_to_iidx: Dict[Any, int],
        local_qi: Dict[int, np.ndarray], local_bi: Dict[int, float], batch_size: int = 3
    ) -> List[Any]:
        """SHLCP: selects the ``batch_size`` lowest-scoring unseen items."""
        mu = svd_model.trainset.global_mean
        shown_set = set(shown)
        scores = {}
        for iid in eligible_items_pool:
            if iid in shown_set:
                continue
            i_inner = item_to_iidx.get(iid)
            if i_inner is None:
                continue
            qi = local_qi.get(i_inner, svd_model.qi[i_inner])
            bi = local_bi.get(i_inner, svd_model.bi[i_inner])
            scores[iid] = mu + bu_cold + bi + np.dot(pu_cold, qi)
        if not scores:
            return []
        b = min(batch_size, len(scores))
        return sorted(scores, key=scores.get)[:b]

    def run_active_learning_session(
        u: int, k: int, lmbda2: float, batch_size: int = 3
    ) -> Tuple[np.ndarray, float, Dict[int, np.ndarray], Dict[int, float], List[Any]]:
        """Runs one full SHLCP session under the given ``lmbda2``."""
        if i_0_inner is not None:
            pu_cold = svd_base.qi[i_0_inner].copy()
        else:
            pu_cold = np.zeros(n_factors)
        bu_cold = 0.0
        local_qi = {}
        local_bi = {}
        shown = [most_popular_iid]

        first_row = data[(data['user_idx'] == u) & (data['itemId'] == most_popular_iid)]
        r_first = float(first_row['interaction'].iloc[0]) if len(first_row) > 0 else 0.0

        if i_0_inner is not None:
            pu_cold, bu_cold = partial_lfm_update_cold(
                svd_base, pu_cold, bu_cold, i_0_inner, r_first, local_qi, local_bi,
                gamma1=GAMMA1, gamma2=GAMMA2, lmbda1=LMBDA1_DEFAULT, lmbda2=lmbda2
            )

        while len(shown) < k:
            b = min(batch_size, k - len(shown))
            batch = select_batch_items_cold(
                svd_base, pu_cold, bu_cold, shown, eligible_items, item_to_iidx,
                local_qi, local_bi, batch_size=b
            )
            if not batch:
                break
            shown.extend(batch)
            for next_item in batch:
                row = data[(data['user_idx'] == u) & (data['itemId'] == next_item)]
                r_ui = float(row['interaction'].iloc[0]) if len(row) > 0 else 0.0
                i_inner = item_to_iidx.get(next_item)
                if i_inner is not None:
                    pu_cold, bu_cold = partial_lfm_update_cold(
                        svd_base, pu_cold, bu_cold, i_inner, r_ui, local_qi, local_bi,
                        gamma1=GAMMA1, gamma2=GAMMA2, lmbda1=LMBDA1_DEFAULT, lmbda2=lmbda2
                    )

        return pu_cold, bu_cold, local_qi, local_bi, shown

    def evaluate_session_rmse(
        pu_cold: np.ndarray, bu_cold: float, local_qi: Dict[int, np.ndarray],
        local_bi: Dict[int, float], shown: List[Any], test_items: List[Any], u: int
    ) -> Optional[float]:
        """Scores RMSE on ``test_items``; returns None if none are valid."""
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
            qi = local_qi.get(i_inner, svd_base.qi[i_inner])
            bi = local_bi.get(i_inner, svd_base.bi[i_inner])
            est = float(np.clip(mu_base + bu_cold + bi + np.dot(pu_cold, qi), 0, 1))
            preds.append(Pred(uid=row.user_idx, iid=row.item_idx,
                               r_ui=row.interaction, est=est, details={}))
        if not preds:
            return None
        return accuracy.rmse(preds, verbose=False)

    results: Dict[Tuple[float, int], float] = {}
    log = []

    print(f"\n=== Regularisation (lambda2) ablation: SHLCP, "
          f"{len(hp_search_users)} users ===", flush=True)

    for lmbda2 in LAMBDA2_GRID:
        for k in K_PROBES:
            t_cell = time.time()
            rmses = []
            for u in hp_search_users:
                pu_cold, bu_cold, local_qi, local_bi, shown = run_active_learning_session(u, k, lmbda2)
                val_items, _ = split_unseen_items(u, shown, val_frac=0.5)
                r = evaluate_session_rmse(pu_cold, bu_cold, local_qi, local_bi, shown, val_items, u)
                if r is not None:
                    rmses.append(r)
            avg = np.mean(rmses) if rmses else float('nan')
            results[(lmbda2, k)] = avg
            log.append((lmbda2, k, avg, len(rmses), time.time() - t_cell))
            print(f"  [lambda2={lmbda2:.0e}][k={k}] val RMSE={avg:.4f} "
                  f"(n={len(rmses)}, {time.time()-t_cell:.1f}s)", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("=== Best lambda2 per k ===", flush=True)
    for k in K_PROBES:
        best = min(LAMBDA2_GRID, key=lambda l: results[(l, k)])
        print(f"  k={k}: best lambda2 = {best:.0e}  RMSE={results[(best,k)]:.4f}  "
              f"(default 1e-6: RMSE={results[(1e-6,k)]:.4f})", flush=True)

    df = pd.DataFrame(log, columns=['lambda2', 'k', 'val_rmse', 'n_users', 'seconds'])
    df.to_csv(OUT_REG_ABL, index=False)
    print(f"\nSaved to {OUT_REG_ABL}", flush=True)
    print(f"TOTAL TIME: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
