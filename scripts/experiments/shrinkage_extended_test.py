"""
shrinkage_extended_test.py
=============================
Follow-up to shrinkage_test.py: the original shrinkage constant c sweep
(c in {5, 10, 20, 50, 100}) found validation RMSE decreasing
monotonically across the entire tested range, with c=100 winning at
every elicitation budget k -- meaning the sweep never found a turning
point, only ran out of grid. This script extends the same sweep to
much larger c (up to 1e6) to find where (or whether) the improvement
actually plateaus or reverses: if personalised RMSE keeps improving as
c grows, note where it plateaus; if it eventually gets worse
(over-shrinking erases the personalisation signal entirely, converging
to the item-level baseline as c -> infinity), report that turning
point.

As c -> infinity, alpha(k) = k/(k+c) -> 0 for any finite k, so the
shrunk prediction converges to mu + b_i (i.e. RMSE converges to the
item-level baseline's own RMSE) -- this script's own de-facto upper
bound on how much shrinkage alone can ever help.

Reference strategy: SHLCP, same 200-user tuning subset and k grid as
shrinkage_test.py, so results are directly comparable.

Usage
-----
    python scripts/experiments/shrinkage_extended_test.py

Input
-----
    results/base_model_cache.pkl

Output
------
    results/shrinkage_extended_test_results.csv
        Columns: k, c, alpha_at_this_k, val_rmse, n_users
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
OUT_SHRINK_EXT: str = os.path.join(RESULTS_DIR, 'shrinkage_extended_test_results.csv')

K_VALUES: List[int] = [10, 25, 50, 100]
# Extends shrinkage_test.py's grid (which stopped at 100) upward.
C_GRID: List[Optional[int]] = [100, 200, 500, 1000, 10000, 1000000]


def _stable_seed(u: int, shown: List[Any]) -> int:
    """Deterministic replacement for Python's built-in ``hash()`` on
    strings, which is randomised per-process unless PYTHONHASHSEED is
    fixed.
    """
    key = f"{int(u)}|" + ','.join(sorted(str(x) for x in shown))
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def main() -> None:
    """Runs the extended SHLCP shrinkage-constant sweep (on top of
    decaying LR) across all four elicitation budgets and writes
    ``results/shrinkage_extended_test_results.csv``.
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
    # Final re-tuned regularisation, matching personalised_strategies.py.
    LMBDA1, LMBDA2   = 1e-5, 1e-4

    print(f"[{time.time()-t_start:6.1f}s] Cache loaded (setup skipped).", flush=True)

    eval_cold_users = cold_users[:min(1000, len(cold_users))]
    hp_search_users = eval_cold_users[:min(200, len(eval_cold_users))]

    Pred = namedtuple('Prediction', ['uid', 'iid', 'r_ui', 'est', 'details'])

    def split_unseen_items(u: int, shown: List[Any],
                            val_frac: float = 0.5) -> Tuple[List[Any], List[Any]]:
        shown_set = set(shown)
        unseen    = [iid for iid in eligible_items if iid not in shown_set]
        rng       = np.random.RandomState(_stable_seed(u, shown))
        rng.shuffle(unseen)
        n_val = int(val_frac * len(unseen))
        return unseen[:n_val], unseen[n_val:]

    def partial_lfm_update_cold_decaying(
        svd_model: Any, pu_cold: np.ndarray, bu_cold: float, i_inner: int, r_ui: float,
        local_qi: Dict[int, np.ndarray], local_bi: Dict[int, float],
        gamma1_eff: float, gamma2_eff: float, lmbda1: float, lmbda2: float
    ) -> Tuple[np.ndarray, float]:
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
        pu_new            = pu_cold + gamma2_eff * (error * qi      - lmbda2 * pu_cold)
        local_qi[i_inner] = qi      + gamma2_eff * (error * pu_cold - lmbda2 * qi)
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

    def run_session_decaying(u, k):
        if i_0_inner is not None:
            pu_cold = svd_base.qi[i_0_inner].copy()
        else:
            pu_cold = np.zeros(n_factors)
        bu_cold  = 0.0
        local_qi, local_bi = {}, {}
        shown = [most_popular_iid]
        n = 0

        first_row = data[(data['user_idx'] == u) & (data['itemId'] == most_popular_iid)]
        has_first = len(first_row) > 0
        r_first   = float(first_row['interaction'].iloc[0]) if has_first else 0.0
        if i_0_inner is not None and has_first:
            decay = 1.0 / np.sqrt(1.0 + n)
            pu_cold, bu_cold = partial_lfm_update_cold_decaying(
                svd_base, pu_cold, bu_cold, i_0_inner, r_first, local_qi, local_bi,
                GAMMA1 * decay, GAMMA2 * decay, LMBDA1, LMBDA2
            )
            n += 1

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
                if i_inner is not None and has_row:
                    decay = 1.0 / np.sqrt(1.0 + n)
                    pu_cold, bu_cold = partial_lfm_update_cold_decaying(
                        svd_base, pu_cold, bu_cold, i_inner, r_ui, local_qi, local_bi,
                        GAMMA1 * decay, GAMMA2 * decay, LMBDA1, LMBDA2
                    )
                    n += 1
        return pu_cold, bu_cold, local_qi, local_bi, shown

    def score_with_shrinkage(pu_cold, bu_cold, local_qi, local_bi, i_inner, alpha):
        qi = local_qi.get(i_inner, svd_base.qi[i_inner])
        bi = local_bi.get(i_inner, svd_base.bi[i_inner])
        personalisation_delta = bu_cold + np.dot(pu_cold, qi)
        return float(np.clip(mu_base + bi + alpha * personalisation_delta, 0, 1))

    rows = []
    for k in K_VALUES:
        print(f"\n=== k={k}: running {len(hp_search_users)} decaying-LR sessions once, "
              f"re-scoring under {len(C_GRID)} extended shrinkage settings ===", flush=True)
        t_k = time.time()

        per_c_rmses = {c: [] for c in C_GRID}

        for u in hp_search_users:
            pu_cold, bu_cold, local_qi, local_bi, shown = run_session_decaying(u, k)
            val_items, _ = split_unseen_items(u, shown, val_frac=0.5)
            shown_set = set(shown)
            test_df = data[(data['user_idx'] == u) &
                           (data['itemId'].isin(val_items)) &
                           (~data['itemId'].isin(shown_set))]
            if len(test_df) == 0:
                continue

            for c in C_GRID:
                alpha = k / (k + c)
                preds = []
                for row in test_df.itertuples():
                    i_inner = item_to_iidx.get(row.itemId)
                    if i_inner is None:
                        continue
                    est = score_with_shrinkage(pu_cold, bu_cold, local_qi, local_bi, i_inner, alpha)
                    preds.append(Pred(uid=row.user_idx, iid=row.item_idx,
                                       r_ui=row.interaction, est=est, details={}))
                if preds:
                    per_c_rmses[c].append(accuracy.rmse(preds, verbose=False))

        for c in C_GRID:
            vals = per_c_rmses[c]
            avg  = np.mean(vals) if vals else float('nan')
            alpha_at_k = k / (k + c)
            rows.append({
                'k': k, 'c': c,
                'alpha_at_this_k': round(alpha_at_k, 5),
                'val_rmse': round(avg, 4), 'n_users': len(vals),
            })
            print(f"  [c={c}] alpha={alpha_at_k:.5f}  val RMSE={avg:.4f}  (n={len(vals)})",
                  flush=True)

        print(f"  k={k} total time: {time.time()-t_k:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_SHRINK_EXT, index=False)

    print(f"\n{'='*70}", flush=True)
    print("=== Best c per k (lowest val RMSE), extended grid ===", flush=True)
    for k in K_VALUES:
        dfk = df[df['k'] == k]
        best_row = dfk.loc[dfk['val_rmse'].idxmin()]
        largest_c_row = dfk.loc[dfk['c'].idxmax()]
        print(f"  k={k}: best c={best_row['c']} RMSE={best_row['val_rmse']:.4f}  "
              f"(c={largest_c_row['c']} (near item-baseline limit): "
              f"RMSE={largest_c_row['val_rmse']:.4f})", flush=True)

    print(f"\nSaved to {OUT_SHRINK_EXT}", flush=True)
    print(f"TOTAL TIME: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
