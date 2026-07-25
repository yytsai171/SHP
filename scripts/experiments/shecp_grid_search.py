"""
shecp_grid_search.py
======================
Grid search over SHECP's epsilon-greedy exploration floor and decay
rate (epsilon_r = max(floor, decay ** r)), evaluated at k=50 on the
full 1,000-user evaluation population.

Deliberately sequential (not parallelised across users): a single
epsilon-greedy RNG stream is shared and continuously advanced across
all users for a given (floor, decay) cell, so that changing the number
of parallel workers can never change which random draws a given user
receives -- see README.md "Reproducibility".

Uses the validation half of each user's remaining unseen items
(disjoint from the test half used for final reporting in
personalised_strategies.py), so this grid search introduces no
leakage into the final evaluation.

Usage
-----
    python scripts/experiments/shecp_grid_search.py

Input
-----
    results/base_model_cache.pkl

Output
------
    results/shecp_grid_results.csv
        Columns: floor, decay, val_rmse, n_users

Expected runtime: ~2 hours (9 grid cells x 1,000 users, sequential).
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
OUT_CSV: str = os.path.join(RESULTS_DIR, 'shecp_grid_results.csv')

BATCH_SIZE: int = 3
# Number of partial-SGD steps per interaction; 1 is the confirmed
# winner from decaying_lr_test.py / thesis Table 3.7 -- fixed here
# rather than swept, since this grid search targets floor/decay only.
NUM_SGD_STEPS: int = 1
FLOOR_GRID: List[float] = [0.05, 0.1, 0.2]
DECAY_GRID: List[float] = [0.8, 0.9, 0.95]


def _stable_seed(u: int, shown: List[Any]) -> int:
    """Deterministic replacement for Python's built-in ``hash()`` on
    strings -- see README.md "Reproducibility"."""
    key = f"{int(u)}|" + ','.join(sorted(str(x) for x in shown))
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def main() -> None:
    """Runs the SHECP floor/decay grid search on the full 1,000-user
    population and writes ``results/shecp_grid_results.csv``.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

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
    LMBDA1, LMBDA2 = cache['LMBDA1'], cache['LMBDA2']

    eval_cold_users = cold_users[:min(1000, len(cold_users))]
    egreedy_rng = np.random.RandomState(123)

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
        pu_cold: np.ndarray, bu_cold: float, i_inner: int, r_ui: float,
        local_qi: Dict[int, np.ndarray], local_bi: Dict[int, float]
    ) -> Tuple[np.ndarray, float]:
        """One partial-SGD update, using the base model's fixed
        GAMMA1/GAMMA2/LMBDA1/LMBDA2 (thesis Eq. 3.9-3.10; no decaying
        LR or shrinkage in this ablation -- floor/decay is the only
        variable under test)."""
        if i_inner not in local_qi:
            local_qi[i_inner] = svd_base.qi[i_inner].copy()
            local_bi[i_inner] = float(svd_base.bi[i_inner])
        mu = svd_base.trainset.global_mean
        for _ in range(NUM_SGD_STEPS):
            qi, bi = local_qi[i_inner], local_bi[i_inner]
            pred = mu + bu_cold + bi + np.dot(pu_cold, qi)
            error = r_ui - pred
            bu_cold = bu_cold + GAMMA1 * (error - LMBDA1 * bu_cold)
            local_bi[i_inner] = bi + GAMMA1 * (error - LMBDA1 * bi)
            pu_new = pu_cold + GAMMA2 * (error * qi - LMBDA2 * pu_cold)
            local_qi[i_inner] = qi + GAMMA2 * (error * pu_cold - LMBDA2 * qi)
            pu_cold = pu_new
        return pu_cold, bu_cold

    def select_batch_items_cold(
        pu_cold: np.ndarray, bu_cold: float, shown: List[Any],
        local_qi: Dict[int, np.ndarray], local_bi: Dict[int, float],
        batch_size: int, round_number: int, epsilon_floor: float, epsilon_decay: float
    ) -> List[Any]:
        """SHECP: explores (SHLCP logic) with probability
        ``max(epsilon_floor, epsilon_decay ** round_number)``, else
        exploits (SHHCP logic). Draws from the single, continuously-
        advancing ``egreedy_rng`` stream shared across all users in
        this (floor, decay) cell."""
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
        epsilon = max(epsilon_floor, epsilon_decay ** round_number)
        rand_draw = egreedy_rng.random()
        if rand_draw < epsilon:
            return sorted(scores, key=scores.get)[:b]                 # explore (SHLCP logic)
        else:
            return sorted(scores, key=scores.get, reverse=True)[:b]   # exploit (SHHCP logic)

    def run_session(
        u: int, k: int, epsilon_floor: float, epsilon_decay: float
    ) -> Tuple[np.ndarray, float, Dict[int, np.ndarray], Dict[int, float], List[Any]]:
        """Runs one full SHECP session for user u under the given
        (epsilon_floor, epsilon_decay)."""
        if i_0_inner is not None:
            pu_cold = svd_base.qi[i_0_inner].copy()
        else:
            pu_cold = np.zeros(n_factors)
        bu_cold = 0.0
        local_qi, local_bi = {}, {}
        shown = [most_popular_iid]

        first_row = data[(data['user_idx'] == u) & (data['itemId'] == most_popular_iid)]
        r_first = float(first_row['interaction'].iloc[0]) if len(first_row) > 0 else 0.0
        if i_0_inner is not None:
            pu_cold, bu_cold = partial_lfm_update_cold(pu_cold, bu_cold, i_0_inner, r_first,
                                                         local_qi, local_bi)

        round_number = 0
        while len(shown) < k:
            b = min(BATCH_SIZE, k - len(shown))
            batch = select_batch_items_cold(pu_cold, bu_cold, shown, local_qi, local_bi,
                                             b, round_number, epsilon_floor, epsilon_decay)
            if not batch:
                break
            shown.extend(batch)
            for item in batch:
                row = data[(data['user_idx'] == u) & (data['itemId'] == item)]
                r_ui = float(row['interaction'].iloc[0]) if len(row) > 0 else 0.0
                i_inner = item_to_iidx.get(item)
                if i_inner is not None:
                    pu_cold, bu_cold = partial_lfm_update_cold(pu_cold, bu_cold, i_inner, r_ui,
                                                                 local_qi, local_bi)
            round_number += 1
        return pu_cold, bu_cold, local_qi, local_bi, shown

    def evaluate_val_rmse(
        pu_cold: np.ndarray, bu_cold: float, local_qi: Dict[int, np.ndarray],
        local_bi: Dict[int, float], shown: List[Any], u: int
    ) -> Optional[float]:
        """Scores validation RMSE for user u; returns None if the
        validation split has no valid items."""
        val_items, _ = split_unseen_items(u, shown, val_frac=0.5)
        shown_set = set(shown)
        test_df = data[(data['user_idx'] == u) & (data['itemId'].isin(val_items)) &
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
            preds.append(Pred(uid=row.user_idx, iid=row.item_idx, r_ui=row.interaction,
                               est=est, details={}))
        if not preds:
            return None
        return accuracy.rmse(preds, verbose=False)

    results = []
    print(f"=== SHECP floor/decay grid, {len(eval_cold_users)}-user population, k=50 ===",
          flush=True)
    t_start = time.time()
    for floor in FLOOR_GRID:
        for decay in DECAY_GRID:
            t_cell = time.time()
            rmses = []
            for u in eval_cold_users:
                pu_cold, bu_cold, local_qi, local_bi, shown = run_session(u, 50, floor, decay)
                r = evaluate_val_rmse(pu_cold, bu_cold, local_qi, local_bi, shown, u)
                if r is not None:
                    rmses.append(r)
            avg = float(np.mean(rmses)) if rmses else float('nan')
            results.append((floor, decay, avg, len(rmses)))
            print(f"  [floor={floor}, decay={decay}] val RMSE={avg:.6f} "
                  f"(n={len(rmses)}, {time.time()-t_cell:.1f}s)", flush=True)

    best = min(results, key=lambda r: r[2])
    print(f"\n-> Best: floor={best[0]}, decay={best[1]}, RMSE={best[2]:.6f}", flush=True)
    print(f"TOTAL TIME: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} min)", flush=True)

    pd.DataFrame(results, columns=['floor', 'decay', 'val_rmse', 'n_users']).to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}", flush=True)


if __name__ == '__main__':
    main()
