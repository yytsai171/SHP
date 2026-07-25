"""
cold_start_init_ablation.py
==============================
Ablation: compares the default cold-user latent-vector initialisation
(p_u^c <- q_{i_0}, the single most-popular eligible item's own latent
vector) against two alternatives:

  - 'zero'         : p_u^c <- 0 (the naive cold-start default the
                    item-based initialisation is meant to improve on).
  - 'top5_mean'    : p_u^c <- mean of the 5 most-popular eligible items'
                    own latent vectors (a weighted/combined reference
                    point instead of a single item).
  - 'item' (default): p_u^c <- q_{i_0}.

In every condition, b_u^c is initialised to 0 -- only the latent
factor vector's initialisation is under test.

Reference strategy: SHLCP, on a 200-user tuning subset.

Usage
-----
    python scripts/experiments/cold_start_init_ablation.py

Input
-----
    results/base_model_cache.pkl

Output
------
    results/cold_start_init_ablation_results.csv
        Columns: init, k, val_rmse, n_users, seconds
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
OUT_INIT_ABL: str = os.path.join(RESULTS_DIR, 'cold_start_init_ablation_results.csv')

INIT_STRATEGIES: List[str] = ['zero', 'item', 'top5_mean']
K_PROBES: List[int] = [10, 100]
TOP_N_FOR_MEAN: int = 5


def _stable_seed(u: int, shown: List[Any]) -> int:
    """Deterministic replacement for Python's built-in ``hash()`` on
    strings, which is randomised per-process unless PYTHONHASHSEED is
    fixed.
    """
    key = f"{int(u)}|" + ','.join(sorted(str(x) for x in shown))
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def main() -> None:
    """Runs the cold-start initialisation ablation for SHLCP at
    k in {10, 100} and writes
    ``results/cold_start_init_ablation_results.csv``.
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
    # Final re-tuned regularisation, matching personalised_strategies.py.
    LMBDA1, LMBDA2 = 1e-5, 1e-4

    print(f"[{time.time()-t_start:6.1f}s] Cache loaded (setup skipped).", flush=True)

    eval_cold_users = cold_users[:min(1000, len(cold_users))]
    hp_search_users = eval_cold_users[:min(200, len(eval_cold_users))]

    # Top-N popular items among eligible items, ranked by warm-user
    # interaction count -- same ranking that selected most_popular_iid.
    warm_data = data[~data['user_idx'].isin(cold_users)]
    item_counts = warm_data['itemId'].value_counts()
    eligible_set = set(eligible_items)
    top_n_iids = [iid for iid in item_counts.index if iid in eligible_set][:TOP_N_FOR_MEAN]
    top_n_inner = [item_to_iidx[iid] for iid in top_n_iids if iid in item_to_iidx]
    top5_mean_vector = (np.mean([svd_base.qi[i] for i in top_n_inner], axis=0)
                         if top_n_inner else np.zeros(n_factors))
    print(f"[{time.time()-t_start:6.1f}s] Top-{TOP_N_FOR_MEAN} popular items for "
          f"'top5_mean' init: {top_n_iids}", flush=True)

    Pred = namedtuple('Prediction', ['uid', 'iid', 'r_ui', 'est', 'details'])

    def init_pu_cold(init: str) -> np.ndarray:
        if init == 'zero':
            return np.zeros(n_factors)
        elif init == 'item':
            if i_0_inner is not None:
                return svd_base.qi[i_0_inner].copy()
            return np.zeros(n_factors)
        elif init == 'top5_mean':
            return top5_mean_vector.copy()
        else:
            raise ValueError(f"Unknown init: {init}")

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
    ) -> Tuple[np.ndarray, float]:
        """One partial-SGD update -- only the initialisation of pu_cold
        (before any update is applied) is under test in this ablation."""
        if i_inner not in local_qi:
            local_qi[i_inner] = svd_model.qi[i_inner].copy()
            local_bi[i_inner] = float(svd_model.bi[i_inner])
        mu = svd_model.trainset.global_mean
        qi = local_qi[i_inner]
        bi = local_bi[i_inner]
        pred = mu + bu_cold + bi + np.dot(pu_cold, qi)
        error = r_ui - pred
        bu_cold = bu_cold + GAMMA1 * (error - LMBDA1 * bu_cold)
        local_bi[i_inner] = bi + GAMMA1 * (error - LMBDA1 * bi)
        pu_new = pu_cold + GAMMA2 * (error * qi - LMBDA2 * pu_cold)
        local_qi[i_inner] = qi + GAMMA2 * (error * pu_cold - LMBDA2 * qi)
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
        u: int, k: int, init: str, batch_size: int = 3
    ) -> Tuple[np.ndarray, float, Dict[int, np.ndarray], Dict[int, float], List[Any]]:
        """Runs one full SHLCP session under the given ``init`` strategy."""
        pu_cold = init_pu_cold(init)
        bu_cold = 0.0
        local_qi: Dict[int, np.ndarray] = {}
        local_bi: Dict[int, float] = {}
        shown = [most_popular_iid]

        first_row = data[(data['user_idx'] == u) & (data['itemId'] == most_popular_iid)]
        has_first = len(first_row) > 0
        r_first = float(first_row['interaction'].iloc[0]) if has_first else 0.0
        i_first = item_to_iidx.get(most_popular_iid)
        if i_first is not None and has_first:
            pu_cold, bu_cold = partial_lfm_update_cold(
                svd_base, pu_cold, bu_cold, i_first, r_first, local_qi, local_bi)

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
                        svd_base, pu_cold, bu_cold, i_inner, r_ui, local_qi, local_bi)

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

    results: Dict[Tuple[str, int], float] = {}
    log = []

    print(f"\n=== Cold-start initialisation ablation: SHLCP, "
          f"{len(hp_search_users)} users ===", flush=True)

    for init in INIT_STRATEGIES:
        for k in K_PROBES:
            t_cell = time.time()
            rmses = []
            for u in hp_search_users:
                pu_cold, bu_cold, local_qi, local_bi, shown = run_active_learning_session(u, k, init)
                val_items, _ = split_unseen_items(u, shown, val_frac=0.5)
                r = evaluate_session_rmse(pu_cold, bu_cold, local_qi, local_bi, shown, val_items, u)
                if r is not None:
                    rmses.append(r)
            avg = np.mean(rmses) if rmses else float('nan')
            results[(init, k)] = avg
            log.append((init, k, avg, len(rmses), time.time() - t_cell))
            print(f"  [init={init}][k={k}] val RMSE={avg:.4f} "
                  f"(n={len(rmses)}, {time.time()-t_cell:.1f}s)", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("=== Best init per k ===", flush=True)
    for k in K_PROBES:
        best = min(INIT_STRATEGIES, key=lambda i: results[(i, k)])
        print(f"  k={k}: best init = {best}  RMSE={results[(best,k)]:.4f}  "
              f"(default 'item': RMSE={results[('item',k)]:.4f})", flush=True)

    df = pd.DataFrame(log, columns=['init', 'k', 'val_rmse', 'n_users', 'seconds'])
    df.to_csv(OUT_INIT_ABL, index=False)
    print(f"\nSaved to {OUT_INIT_ABL}", flush=True)
    print(f"TOTAL TIME: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
