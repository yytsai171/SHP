"""
joint_regularization_ablation.py
===================================
Grids over BOTH lambda1 (bias regularisation) and lambda2
(factor-vector regularisation) jointly, rather than sweeping either one
alone with the other held fixed at its Geurts et al. (2020) default
(1e-7 / 1e-6), to check whether a stronger effect exists in the joint
(lambda1, lambda2) space than either coordinate can find alone.

Restricted to k=100 only, to keep runtime bounded.

Reference strategy: SHLCP, on a 200-user hyperparameter-tuning subset.

Usage
-----
    python scripts/experiments/joint_regularization_ablation.py

Input
-----
    results/base_model_cache.pkl

Output
------
    results/joint_regularization_ablation_results.csv
        Columns: lambda1, lambda2, k, val_rmse, n_users, seconds
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
OUT_JOINT_REG: str = os.path.join(RESULTS_DIR, 'joint_regularization_ablation_results.csv')

# lambda1 default (Geurts et al. 2020) is 1e-7; lambda2 default is 1e-6.
LAMBDA1_GRID: List[float] = [1e-7, 1e-6, 1e-5]
LAMBDA2_GRID: List[float] = [1e-6, 1e-5, 1e-4, 1e-3]
K_PROBE: int = 100


def _stable_seed(u: int, shown: List[Any]) -> int:
    """Deterministic replacement for Python's built-in ``hash()`` on
    strings, which is randomised per-process unless PYTHONHASHSEED is
    fixed.
    """
    key = f"{int(u)}|" + ','.join(sorted(str(x) for x in shown))
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def main() -> None:
    """Runs the joint (lambda1, lambda2) regularisation-strength
    ablation for SHLCP at k=100 and writes
    ``results/joint_regularization_ablation_results.csv``.
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

    print(f"[{time.time()-t_start:6.1f}s] Cache loaded (setup skipped).", flush=True)

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

    def partial_lfm_update_cold(
        svd_model: Any, pu_cold: np.ndarray, bu_cold: float, i_inner: int, r_ui: float,
        local_qi: Dict[int, np.ndarray], local_bi: Dict[int, float],
        gamma1: float, gamma2: float, lmbda1: float, lmbda2: float,
    ) -> Tuple[np.ndarray, float]:
        """One partial-SGD update with both lambda1 and lambda2 (the
        joint grid under test) applied."""
        if i_inner not in local_qi:
            local_qi[i_inner] = svd_model.qi[i_inner].copy()
            local_bi[i_inner] = float(svd_model.bi[i_inner])
        mu = svd_model.trainset.global_mean
        qi = local_qi[i_inner]
        bi = local_bi[i_inner]
        pred = mu + bu_cold + bi + np.dot(pu_cold, qi)
        error = r_ui - pred
        bu_cold = bu_cold + gamma1 * (error - lmbda1 * bu_cold)
        local_bi[i_inner] = bi + gamma1 * (error - lmbda1 * bi)
        pu_new = pu_cold + gamma2 * (error * qi - lmbda2 * pu_cold)
        local_qi[i_inner] = qi + gamma2 * (error * pu_cold - lmbda2 * qi)
        return pu_new, bu_cold

    def select_batch_items_cold(
        pu_cold: np.ndarray, bu_cold: float, shown: List[Any],
        local_qi: Dict[int, np.ndarray], local_bi: Dict[int, float], batch_size: int = 3
    ) -> List[Any]:
        """SHLCP: selects the ``batch_size`` lowest-scoring unseen items."""
        mu = svd_base.trainset.global_mean
        shown_set = set(shown)
        scores = {}
        for iid in eligible_items:
            if iid in shown_set:
                continue
            i_inner = item_to_iidx.get(iid)
            if i_inner is None:
                continue
            qi = local_qi.get(i_inner, svd_base.qi[i_inner])
            bi = local_bi.get(i_inner, svd_base.bi[i_inner])
            scores[iid] = mu + bu_cold + bi + np.dot(pu_cold, qi)
        if not scores:
            return []
        b = min(batch_size, len(scores))
        return sorted(scores, key=scores.get)[:b]

    def run_active_learning_session(
        u: int, k: int, lmbda1: float, lmbda2: float, batch_size: int = 3
    ) -> Tuple[np.ndarray, float, Dict[int, np.ndarray], Dict[int, float], List[Any]]:
        """Runs one full SHLCP session under the given (lmbda1, lmbda2)."""
        if i_0_inner is not None:
            pu_cold = svd_base.qi[i_0_inner].copy()
        else:
            pu_cold = np.zeros(n_factors)
        bu_cold = 0.0
        local_qi: Dict[int, np.ndarray] = {}
        local_bi: Dict[int, float] = {}
        shown = [most_popular_iid]

        first_row = data[(data['user_idx'] == u) & (data['itemId'] == most_popular_iid)]
        has_first = len(first_row) > 0
        r_first = float(first_row['interaction'].iloc[0]) if has_first else 0.0

        if i_0_inner is not None and has_first:
            pu_cold, bu_cold = partial_lfm_update_cold(
                svd_base, pu_cold, bu_cold, i_0_inner, r_first, local_qi, local_bi,
                gamma1=GAMMA1, gamma2=GAMMA2, lmbda1=lmbda1, lmbda2=lmbda2
            )

        while len(shown) < k:
            b = min(batch_size, k - len(shown))
            batch = select_batch_items_cold(pu_cold, bu_cold, shown, local_qi, local_bi, b)
            if not batch:
                break
            shown.extend(batch)
            for next_item in batch:
                row = data[(data['user_idx'] == u) & (data['itemId'] == next_item)]
                has_row = len(row) > 0
                r_ui = float(row['interaction'].iloc[0]) if has_row else 0.0
                i_inner = item_to_iidx.get(next_item)
                if i_inner is not None and has_row:
                    pu_cold, bu_cold = partial_lfm_update_cold(
                        svd_base, pu_cold, bu_cold, i_inner, r_ui, local_qi, local_bi,
                        gamma1=GAMMA1, gamma2=GAMMA2, lmbda1=lmbda1, lmbda2=lmbda2
                    )

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
            qi = local_qi.get(i_inner, svd_base.qi[i_inner])
            bi = local_bi.get(i_inner, svd_base.bi[i_inner])
            est = float(np.clip(mu_base + bu_cold + bi + np.dot(pu_cold, qi), 0, 1))
            preds.append(Pred(uid=row.user_idx, iid=row.item_idx,
                               r_ui=row.interaction, est=est, details={}))
        if not preds:
            return None
        return accuracy.rmse(preds, verbose=False)

    results: Dict[Tuple[float, float], float] = {}
    log = []

    print(f"\n=== Joint (lambda1, lambda2) ablation: SHLCP, k={K_PROBE}, "
          f"{len(hp_search_users)} users ===", flush=True)

    for lmbda1 in LAMBDA1_GRID:
        for lmbda2 in LAMBDA2_GRID:
            t_cell = time.time()
            rmses = []
            for u in hp_search_users:
                pu_cold, bu_cold, local_qi, local_bi, shown = run_active_learning_session(
                    u, K_PROBE, lmbda1, lmbda2)
                val_items, _ = split_unseen_items(u, shown, val_frac=0.5)
                r = evaluate_session_rmse(pu_cold, bu_cold, local_qi, local_bi, shown, val_items, u)
                if r is not None:
                    rmses.append(r)
            avg = np.mean(rmses) if rmses else float('nan')
            results[(lmbda1, lmbda2)] = avg
            log.append((lmbda1, lmbda2, K_PROBE, avg, len(rmses), time.time() - t_cell))
            print(f"  [lambda1={lmbda1:.0e}][lambda2={lmbda2:.0e}] val RMSE={avg:.4f} "
                  f"(n={len(rmses)}, {time.time()-t_cell:.1f}s)", flush=True)

    print(f"\n{'='*70}", flush=True)
    best_pair = min(results, key=results.get)
    default_pair = (1e-7, 1e-6)
    print(f"=== Best (lambda1, lambda2) pair at k={K_PROBE} ===", flush=True)
    print(f"  best: lambda1={best_pair[0]:.0e}, lambda2={best_pair[1]:.0e}  "
          f"RMSE={results[best_pair]:.4f}", flush=True)
    print(f"  default: lambda1={default_pair[0]:.0e}, lambda2={default_pair[1]:.0e}  "
          f"RMSE={results.get(default_pair, float('nan')):.4f}", flush=True)

    df = pd.DataFrame(log, columns=['lambda1', 'lambda2', 'k', 'val_rmse', 'n_users', 'seconds'])
    df.to_csv(OUT_JOINT_REG, index=False)
    print(f"\nSaved to {OUT_JOINT_REG}", flush=True)
    print(f"TOTAL TIME: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
