"""
b_ablation.py
===============
Ablation: batch size B (number of items revealed per active-learning
round before the next selection is made). Sweeps B in {1,2,3,4,5} for
all four personalised strategies, at k=50, on the 200-user
hyperparameter-tuning subset.

SHHCP is expected to be the strategy most sensitive to B: its
"exploitation trap" dynamic (repeatedly reinforcing the same
high-scoring region) means the order in which items are selected and
fed back into the score ranking can matter far more than it does for
SHLCP's simple argmin selection rule.

Usage
-----
    python scripts/experiments/b_ablation.py

Input
-----
    results/base_model_cache.pkl

Output
------
    results/b_ablation_results.csv
        Columns: strategy, B, val_rmse, n_users, seconds

Complexity
----------
O(|STRATEGIES| * |B_GRID| * 200 * 50) partial-SGD updates (each O(F));
item selection is O(|eligible_items|) per round, dict-based rather than
vectorised -- acceptable at this 200-user, single-process scale (see
README.md "Methodology" for why SECTION 3's parallel evaluation uses a
vectorised version instead).
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
OUT_B_ABL: str = os.path.join(RESULTS_DIR, 'b_ablation_results.csv')

SHECP_FLOOR: float = 0.05
SHECP_DECAY: float = 0.95
B_GRID: List[int] = [1, 2, 3, 4, 5]
STRATEGIES: List[str] = ['SHHCP', 'SHLCP', 'SHMCP', 'SHECP']


def _stable_seed(u: int, shown: List[Any]) -> int:
    """Deterministic replacement for Python's built-in ``hash()`` on
    strings -- see README.md "Reproducibility".
    """
    key = f"{int(u)}|" + ','.join(sorted(str(x) for x in shown))
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def main() -> None:
    """Runs the batch-size (B) ablation across all four strategies and
    writes ``results/b_ablation_results.csv``.
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

    eval_cold_users = cold_users[:min(1000, len(cold_users))]
    hp_search_users = eval_cold_users[:min(200, len(eval_cold_users))]

    Pred = namedtuple('Prediction', ['uid', 'iid', 'r_ui', 'est', 'details'])

    def split_unseen_items(u: int, shown: List[Any],
                            val_frac: float = 0.5) -> Tuple[List[Any], List[Any]]:
        """Validation/test split, deterministically seeded per (u, shown).
        Only the validation half is used in this ablation."""
        shown_set = set(shown)
        unseen    = [iid for iid in eligible_items if iid not in shown_set]
        rng       = np.random.RandomState(_stable_seed(u, shown))
        rng.shuffle(unseen)
        n_val = int(val_frac * len(unseen))
        return unseen[:n_val], unseen[n_val:]

    def partial_lfm_update_cold(
        svd_model: Any, pu_cold: np.ndarray, bu_cold: float, i_inner: int, r_ui: float,
        local_qi: Dict[int, np.ndarray], local_bi: Dict[int, float],
        gamma1: float = 0.005, gamma2: float = 0.005,
        lmbda1: float = 1e-7, lmbda2: float = 1e-6, num_sgd_steps: int = 1
    ) -> Tuple[np.ndarray, float]:
        """One (or num_sgd_steps) partial-SGD update(s) given a
        newly-observed interaction (thesis Eq. 3.9-3.10)."""
        if i_inner not in local_qi:
            local_qi[i_inner] = svd_model.qi[i_inner].copy()
            local_bi[i_inner] = float(svd_model.bi[i_inner])
        mu = svd_model.trainset.global_mean
        for _ in range(num_sgd_steps):
            qi  = local_qi[i_inner]
            bi  = local_bi[i_inner]
            pred  = mu + bu_cold + bi + np.dot(pu_cold, qi)
            error = r_ui - pred
            bu_cold           = bu_cold + gamma1 * (error - lmbda1 * bu_cold)
            local_bi[i_inner] = bi      + gamma1 * (error - lmbda1 * bi)
            pu_new            = pu_cold + gamma2 * (error * qi      - lmbda2 * pu_cold)
            local_qi[i_inner] = qi      + gamma2 * (error * pu_cold - lmbda2 * qi)
            pu_cold = pu_new
        return pu_cold, bu_cold

    def select_batch_items_cold(
        svd_model: Any, pu_cold: np.ndarray, bu_cold: float,
        shown: List[Any], eligible_items_pool: List[Any], item_to_iidx: Dict[Any, int],
        local_qi: Dict[int, np.ndarray], local_bi: Dict[int, float],
        strategy: str, batch_size: int = 3, round_number: int = 0,
        egreedy_rng_local: Optional[np.random.RandomState] = None
    ) -> List[Any]:
        """Selects the next batch of items under ``strategy`` (all four
        strategies supported; dict-based, non-vectorised -- see
        README.md "Methodology" for the selection rules)."""
        mu        = svd_model.trainset.global_mean
        shown_set = set(shown)
        scores    = {}
        for iid in eligible_items_pool:
            if iid in shown_set:
                continue
            i_inner = item_to_iidx.get(iid)
            if i_inner is None:
                continue
            qi          = local_qi.get(i_inner, svd_model.qi[i_inner])
            bi          = local_bi.get(i_inner, svd_model.bi[i_inner])
            scores[iid] = mu + bu_cold + bi + np.dot(pu_cold, qi)
        if not scores:
            return []
        b = min(batch_size, len(scores))
        if strategy == 'SHHCP':
            return sorted(scores, key=scores.get, reverse=True)[:b]
        elif strategy == 'SHLCP':
            return sorted(scores, key=scores.get)[:b]
        elif strategy == 'SHMCP':
            median_val = np.median(list(scores.values()))
            return sorted(scores, key=lambda i: abs(scores[i] - median_val))[:b]
        elif strategy == 'SHECP':
            epsilon = max(SHECP_FLOOR, SHECP_DECAY ** round_number)
            if egreedy_rng_local.random() < epsilon:
                return sorted(scores, key=scores.get)[:b]
            else:
                return sorted(scores, key=scores.get, reverse=True)[:b]
        raise ValueError(f"Unknown strategy: {strategy}")

    def run_active_learning_session(
        u: int, strategy: str, k: int, num_sgd_steps: int, batch_size: int,
        egreedy_rng_local: np.random.RandomState
    ) -> Tuple[np.ndarray, float, Dict[int, np.ndarray], Dict[int, float], List[Any]]:
        """Runs one full active-learning session for user u under
        ``strategy``, with the given batch size B. Returns the final
        cold-user parameters, local item copies, and shown-item list."""
        if i_0_inner is not None:
            pu_cold = svd_base.qi[i_0_inner].copy()
        else:
            pu_cold = np.zeros(n_factors)
        bu_cold  = 0.0
        local_qi = {}
        local_bi = {}
        shown    = [most_popular_iid]

        first_row = data[(data['user_idx'] == u) & (data['itemId'] == most_popular_iid)]
        r_first   = float(first_row['interaction'].iloc[0]) if len(first_row) > 0 else 0.0

        if i_0_inner is not None:
            pu_cold, bu_cold = partial_lfm_update_cold(
                svd_base, pu_cold, bu_cold, i_0_inner, r_first,
                local_qi, local_bi,
                gamma1=GAMMA1, gamma2=GAMMA2, lmbda1=LMBDA1, lmbda2=LMBDA2,
                num_sgd_steps=num_sgd_steps
            )

        round_number = 0
        while len(shown) < k:
            b     = min(batch_size, k - len(shown))
            batch = select_batch_items_cold(
                svd_base, pu_cold, bu_cold,
                shown, eligible_items, item_to_iidx,
                local_qi, local_bi, strategy, batch_size=b,
                round_number=round_number, egreedy_rng_local=egreedy_rng_local
            )
            if not batch:
                break
            shown.extend(batch)
            for next_item in batch:
                row   = data[(data['user_idx'] == u) & (data['itemId'] == next_item)]
                r_ui  = float(row['interaction'].iloc[0]) if len(row) > 0 else 0.0
                i_inner = item_to_iidx.get(next_item)
                if i_inner is not None:
                    pu_cold, bu_cold = partial_lfm_update_cold(
                        svd_base, pu_cold, bu_cold, i_inner, r_ui,
                        local_qi, local_bi,
                        gamma1=GAMMA1, gamma2=GAMMA2, lmbda1=LMBDA1, lmbda2=LMBDA2,
                        num_sgd_steps=num_sgd_steps
                    )
            round_number += 1

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
            qi  = local_qi.get(i_inner, svd_base.qi[i_inner])
            bi  = local_bi.get(i_inner, svd_base.bi[i_inner])
            est = float(np.clip(mu_base + bu_cold + bi + np.dot(pu_cold, qi), 0, 1))
            preds.append(Pred(uid=row.user_idx, iid=row.item_idx,
                               r_ui=row.interaction, est=est, details={}))
        if not preds:
            return None
        return accuracy.rmse(preds, verbose=False)

    b_results = {}
    b_log = []

    for strategy in STRATEGIES:
        print(f"\n=== Batch size (B) ablation: {strategy}, k=50, "
              f"{len(hp_search_users)} users ===", flush=True)
        egreedy_rng_local = np.random.RandomState(123)  # fresh per strategy

        for B in B_GRID:
            t_b_start = time.time()
            rmses = []
            for u in hp_search_users:
                pu_cold, bu_cold, local_qi, local_bi, shown = run_active_learning_session(
                    u, strategy, 50, num_sgd_steps=1, batch_size=B,
                    egreedy_rng_local=egreedy_rng_local
                )
                val_items, _ = split_unseen_items(u, shown, val_frac=0.5)
                r = evaluate_session_rmse(pu_cold, bu_cold, local_qi, local_bi, shown, val_items, u)
                if r is not None:
                    rmses.append(r)
            avg = np.mean(rmses) if rmses else float('nan')
            b_results[(strategy, B)] = avg
            b_time = time.time() - t_b_start
            b_log.append((strategy, B, avg, len(rmses), b_time))
            print(f"  [{strategy}][B={B}] val RMSE={avg:.4f}  (n={len(rmses)} users, {b_time:.1f}s)",
                  flush=True)

    print(f"\n{'='*70}", flush=True)
    print("=== Best B per strategy ===", flush=True)
    for strategy in STRATEGIES:
        best_B = min(B_GRID, key=lambda b: b_results[(strategy, b)])
        print(f"  {strategy}: best B = {best_B}  "
              f"(RMSE range across B: {min(b_results[(strategy,b)] for b in B_GRID):.4f} - "
              f"{max(b_results[(strategy,b)] for b in B_GRID):.4f})", flush=True)

    df_b = pd.DataFrame(b_log, columns=['strategy', 'B', 'val_rmse', 'n_users', 'seconds'])
    df_b.to_csv(OUT_B_ABL, index=False)
    print(f"\nResults saved to {OUT_B_ABL}", flush=True)
    print(f"TOTAL TIME: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
