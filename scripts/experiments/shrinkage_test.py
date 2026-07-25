"""
shrinkage_test.py
===================
Ablation: sweeps the confidence-weighted shrinkage constant c, run on
top of the decaying learning rate schedule (see decaying_lr_test.py,
which confirms decaying LR helps before this sweep is run).

Current prediction (no shrinkage): a_hat = mu + b_u^c + b_i + (p_u^c)^T q_i
    -- always fully trusts the personalisation term, however little
       evidence supports it.

Shrinkage prediction: a_hat = mu + b_i + alpha(k) * (b_u^c + (p_u^c)^T q_i)
    -- alpha(k) = k / (k + c), a classic empirical-Bayes-style shrinkage
       weight (Efron & Morris, 1975). Small k (few interactions
       revealed) -> alpha near 0, trust the stable item-level baseline.
       Large k -> alpha near 1, trust the personalisation term fully.
       c controls how many interactions are needed before
       personalisation is "half-trusted".

Since alpha depends only on k (the total session length at evaluation
time), shrinkage only changes SCORING, not the active-learning loop
itself. That means each user's session can be run ONCE per k
(identical to the decaying-LR reference simulation), then the held-out
test items are re-scored under every candidate c value without
re-running the expensive active-learning loop.

c=None reproduces the no-shrinkage prediction exactly (alpha=1 always)
-- included as the baseline row for direct comparison.

Reference strategy: SHLCP, same 200-user tuning subset and k grid as
decaying_lr_test.py, so results are directly comparable.

Usage
-----
    python scripts/experiments/shrinkage_test.py

Input
-----
    results/base_model_cache.pkl

Output
------
    results/shrinkage_test_results.csv
        Columns: k, c, alpha_at_this_k, val_rmse, n_users

Complexity
----------
O(|K_VALUES| * 200) active-learning sessions (each O(k*F)), each then
re-scored O(|C_GRID|) times at O(1) per candidate (only the scoring
formula changes, not the trained parameters) -- this is what makes the
"run once, re-score many times" design cheaper than a naive re-run per
c value.
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
OUT_SHRINK: str = os.path.join(RESULTS_DIR, 'shrinkage_test_results.csv')

K_VALUES: List[int] = [10, 25, 50, 100]
# None = no shrinkage (alpha=1 always); other values are candidate c's.
C_GRID: List[Optional[int]] = [None, 5, 10, 20, 50, 100]


def _stable_seed(u: int, shown: List[Any]) -> int:
    """Deterministic replacement for Python's built-in ``hash()`` on
    strings -- see README.md "Reproducibility".
    """
    key = f"{int(u)}|" + ','.join(sorted(str(x) for x in shown))
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def main() -> None:
    """Runs the SHLCP shrinkage-constant sweep (on top of decaying LR)
    across all four elicitation budgets and writes
    ``results/shrinkage_test_results.csv``.
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
        """Splits remaining unseen items into validation/test halves,
        deterministically seeded per (u, shown). Only the validation
        half is used in this ablation."""
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
        """One partial-SGD update under the (already-confirmed-helpful)
        decaying learning-rate schedule -- gamma1_eff/gamma2_eff are
        pre-scaled by the caller before being passed in."""
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
        # SHLCP: select the batch_size lowest-scoring unseen items.
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
        r_first   = float(first_row['interaction'].iloc[0]) if len(first_row) > 0 else 0.0
        if i_0_inner is not None:
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
                r_ui = float(row['interaction'].iloc[0]) if len(row) > 0 else 0.0
                i_inner = item_to_iidx.get(item)
                if i_inner is not None:
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
              f"re-scoring under {len(C_GRID)} shrinkage settings ===", flush=True)
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
                alpha = 1.0 if c is None else k / (k + c)
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
            alpha_at_k = 1.0 if c is None else k / (k + c)
            rows.append({
                'k': k, 'c': c if c is not None else 'none',
                'alpha_at_this_k': round(alpha_at_k, 3),
                'val_rmse': round(avg, 4), 'n_users': len(vals),
            })
            label = 'no shrinkage' if c is None else f'c={c}'
            print(f"  [{label}] alpha={alpha_at_k:.3f}  val RMSE={avg:.4f}  (n={len(vals)})",
                  flush=True)

        print(f"  k={k} total time: {time.time()-t_k:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_SHRINK, index=False)

    print(f"\n{'='*70}", flush=True)
    print("=== Best c per k (lowest val RMSE) ===", flush=True)
    for k in K_VALUES:
        dfk = df[df['k'] == k]
        best_row = dfk.loc[dfk['val_rmse'].idxmin()]
        no_shrink_row = dfk[dfk['c'] == 'none'].iloc[0]
        improvement = no_shrink_row['val_rmse'] - best_row['val_rmse']
        print(f"  k={k}: best c={best_row['c']} RMSE={best_row['val_rmse']:.4f} "
              f"vs. no-shrinkage RMSE={no_shrink_row['val_rmse']:.4f} "
              f"(improvement: {improvement:+.4f})", flush=True)

    print(f"\nSaved to {OUT_SHRINK}", flush=True)
    print(f"TOTAL TIME: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
