"""
baseline_averaged_matched_pop_test.py
========================================
Fixes a population-mismatch flaw discovered in baseline_averaged_test.py.

baseline_averaged_test.py averages Popularity/PopError RMSE over 30
independent reshuffles of each user's val/test split (same idea as
Random's existing item-set-draw averaging), to check whether the
non-monotonic k=50 RMSE dip is single-draw noise. It confirmed the dip
is noise -- but it also silently pooled in a much larger evaluation
population than the single deterministic split uses (~795-817 users at
k=10 vs ~546-584 for the official single-draw baseline_results.csv),
because different splits of the SAME fixed item set land different
sparse users' one-or-two interactions in the test half across draws.
This is the same class of population-mismatch flaw that
baseline_ranking_metrics.py's own averaging methodology avoids, just
reintroduced at smaller scale.

This script keeps the noise-reduction (30-draw averaging) but restricts
each (strategy, k) cell to exactly the population the official
single-draw split evaluates (results/baseline_results.csv's
ColdUsersEvaluated), which is also the population Random and the
personalised strategies are evaluated against. Within that fixed
population, a user's reported RMSE is still the mean over whichever of
the 30 draws placed one of their interactions in that draw's test half
(matching baseline_averaged_test.py's per-user averaging rule) -- only
the population eligibility criterion changes.

Usage
-----
    python scripts/experiments/baseline_averaged_matched_pop_test.py

Input
-----
    results/base_model_cache.pkl

Output
------
    results/baseline_averaged_matched_pop_results.csv
        Columns: strategy, k, val_rmse, n_users, n_users_unmatched_avg
        (n_users_unmatched_avg is baseline_averaged_test.py's inflated
        count, kept alongside for direct comparison.)
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time
from collections import namedtuple
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from surprise import accuracy

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
MODEL_CACHE: str = os.path.join(RESULTS_DIR, 'base_model_cache.pkl')
OUT_CSV: str = os.path.join(RESULTS_DIR, 'baseline_averaged_matched_pop_results.csv')

K_VALUES: List[int] = [10, 25, 50, 100]
NUM_EVAL_USERS: int = 1000
N_DRAWS: int = 30


def _stable_seed(u: int, shown: List[Any], draw: int = None) -> int:
    """Matches baseline_ranking_metrics.py's single-draw seed when
    ``draw`` is None (the official, matched-population split), and
    baseline_averaged_test.py's per-draw seed otherwise.
    """
    if draw is None:
        key = f"{int(u)}|" + ','.join(sorted(str(x) for x in shown))
    else:
        key = f"{int(u)}|{draw}|" + ','.join(sorted(str(x) for x in shown))
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    with open(MODEL_CACHE, 'rb') as f:
        cache = pickle.load(f)

    data = cache['data']
    eligible_items = cache['eligible_items']
    item_to_iidx = cache['item_to_iidx']
    svd_base = cache['svd_base']
    mu_base = cache['mu_base']
    cold_users = cache['cold_users']
    ALPHA = cache['ALPHA']

    eval_cold_users = cold_users[:min(NUM_EVAL_USERS, len(cold_users))]
    eval_users_set = set(int(u) for u in eval_cold_users)
    user_data = data[data['user_idx'].isin(eval_users_set)]
    user_item_interaction: Dict[int, Dict[Any, float]] = {
        int(u): dict(zip(g['itemId'].values, g['interaction'].values))
        for u, g in user_data.groupby('user_idx')
    }

    warm_data = data[~data['user_idx'].isin(cold_users)]
    item_counts = warm_data['itemId'].value_counts()
    item_mean_interaction = warm_data.groupby('itemId')['interaction'].mean()
    error_scores: Dict[Any, float] = {}
    for item in eligible_items:
        p = float(item_mean_interaction.get(item, 0.5))
        error_scores[item] = min(p, 1.0 - p)
    poperror_scores: Dict[Any, float] = {
        item: ALPHA * np.log10(item_counts[item]) + (1 - ALPHA) * error_scores[item]
        for item in eligible_items
    }

    def select_items(strategy: str, k: int) -> List[Any]:
        if strategy == 'popularity':
            return list(item_counts.loc[eligible_items].sort_values(ascending=False).head(k).index)
        elif strategy == 'poperror':
            return sorted(poperror_scores, key=poperror_scores.get, reverse=True)[:k]
        raise ValueError(f"Unknown strategy: {strategy}")

    def baseline_predict(iid: Any) -> float:
        i_inner = item_to_iidx.get(iid)
        if i_inner is None:
            return mu_base
        return float(np.clip(mu_base + svd_base.bi[i_inner], 0, 1))

    Pred = namedtuple('Prediction', ['uid', 'iid', 'r_ui', 'est', 'details'])

    def split_unseen_items(u: int, shown: List[Any], draw: int = None, val_frac: float = 0.5):
        shown_set = set(shown)
        unseen = [iid for iid in eligible_items if iid not in shown_set]
        rng = np.random.RandomState(_stable_seed(u, shown, draw))
        rng.shuffle(unseen)
        n_val = int(val_frac * len(unseen))
        return unseen[:n_val], unseen[n_val:]

    rows = []
    for strategy in ['popularity', 'poperror']:
        for k in K_VALUES:
            t_cell = time.time()
            shown = select_items(strategy, k)

            # Step 1: the OFFICIAL single-draw matched population --
            # exactly what baseline_results.csv / personalised_results.csv
            # evaluate against (same split fn, draw=None).
            matched_population = []
            for u in eval_cold_users:
                user_dict = user_item_interaction.get(int(u))
                if not user_dict:
                    continue
                _, test_items = split_unseen_items(u, shown, draw=None)
                test_set = set(test_items)
                if any(iid in test_set for iid in user_dict):
                    matched_population.append(int(u))

            # Step 2: for exactly that population, average RMSE over
            # N_DRAWS reshuffled splits (noise reduction), same
            # per-user averaging rule as baseline_averaged_test.py.
            per_user_draw_rmse = []
            for u in matched_population:
                user_dict = user_item_interaction[u]
                draw_rmses = []
                for draw in range(N_DRAWS):
                    _, test_items = split_unseen_items(u, shown, draw=draw, val_frac=0.5)
                    test_set = set(test_items)
                    test_rows = [(iid, r) for iid, r in user_dict.items() if iid in test_set]
                    if not test_rows:
                        continue
                    preds = [Pred(uid=u, iid=iid, r_ui=r, est=baseline_predict(iid), details={})
                             for iid, r in test_rows]
                    draw_rmses.append(accuracy.rmse(preds, verbose=False))
                if draw_rmses:
                    per_user_draw_rmse.append(np.mean(draw_rmses))
                else:
                    # Matched-population member had no test-half hit in
                    # any of the 30 reshuffles (rare) -- fall back to
                    # their single official-split RMSE so they are not
                    # silently dropped from the matched population.
                    _, test_items = split_unseen_items(u, shown, draw=None)
                    test_set = set(test_items)
                    test_rows = [(iid, r) for iid, r in user_dict.items() if iid in test_set]
                    preds = [Pred(uid=u, iid=iid, r_ui=r, est=baseline_predict(iid), details={})
                             for iid, r in test_rows]
                    per_user_draw_rmse.append(accuracy.rmse(preds, verbose=False))

            avg = float(np.mean(per_user_draw_rmse)) if per_user_draw_rmse else float('nan')
            rows.append({
                'strategy': strategy, 'k': k, 'val_rmse': round(avg, 4),
                'n_users': len(matched_population),
            })
            print(f"  [{strategy}][k={k}] RMSE={avg:.4f} (matched n={len(matched_population)}, "
                  f"{N_DRAWS} draws, {time.time()-t_cell:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved to {OUT_CSV}", flush=True)
    print(f"TOTAL TIME: {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
