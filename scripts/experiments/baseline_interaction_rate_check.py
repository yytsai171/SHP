"""
baseline_interaction_rate_check.py
====================================
Companion to update_frequency_by_k_check.py, but for the three
non-personalised baselines (Random, Popularity, PopError) instead of
the four personalised strategies.

Baselines show every cold user the SAME fixed k-item set (chosen
before any responses are observed), so there is no partial-SGD update
to count -- but the same underlying question applies: of the k items
shown, how many does a given cold user actually have a recorded
interaction with? This measures that directly, using the identical
item-selection logic (select_items) as baseline_ranking_metrics.py.

Usage
-----
    python scripts/experiments/baseline_interaction_rate_check.py

Input
-----
    results/base_model_cache.pkl

Output
------
    results/baseline_interaction_rate_results.csv
        Columns: strategy, k, user, n_real_interactions, n_shown
"""

from __future__ import annotations

import os
import pickle
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
MODEL_CACHE: str = os.path.join(RESULTS_DIR, 'base_model_cache.pkl')
OUT_CSV: str = os.path.join(RESULTS_DIR, 'baseline_interaction_rate_results.csv')

K_VALUES: List[int] = [10, 25, 50, 100]
NUM_EVAL_USERS: int = 1000
N_RANDOM_DRAWS: int = 30


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    with open(MODEL_CACHE, 'rb') as f:
        cache = pickle.load(f)

    data = cache['data']
    eligible_items = cache['eligible_items']
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

    def select_items(strategy: str, k: int, draw_seed: Optional[int] = None) -> List[Any]:
        if strategy == 'random':
            return random.Random(draw_seed).sample(eligible_items, k)
        elif strategy == 'popularity':
            return list(item_counts.loc[eligible_items].sort_values(ascending=False).head(k).index)
        elif strategy == 'poperror':
            return sorted(poperror_scores, key=poperror_scores.get, reverse=True)[:k]
        raise ValueError(f"Unknown strategy: {strategy}")

    rows = []
    for strategy in ['random', 'popularity', 'poperror']:
        for k in K_VALUES:
            draws = range(N_RANDOM_DRAWS) if strategy == 'random' else [None]
            for draw_idx, draw_seed in enumerate(draws):
                shown = select_items(strategy, k, draw_seed=draw_seed)
                shown_set = set(shown)
                for u in eval_cold_users:
                    user_dict = user_item_interaction.get(int(u), {})
                    n_real = sum(1 for iid in shown_set if iid in user_dict)
                    rows.append({'strategy': strategy, 'k': k, 'draw': draw_idx,
                                 'user': int(u), 'n_real_interactions': n_real,
                                 'n_shown': len(shown)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved to {OUT_CSV} ({time.time()-t0:.1f}s)", flush=True)

    print("\n=== Fraction of users with >=1 real interaction among shown items, "
          "by (strategy, k), averaged over draws ===", flush=True)
    per_draw = (df.assign(has_real=(df['n_real_interactions'] >= 1))
                  .groupby(['strategy', 'k', 'draw'])['has_real'].mean().reset_index())
    summary = (per_draw.groupby(['strategy', 'k'])['has_real']
               .agg(['mean', 'std', 'count']).reset_index())
    print(summary.to_string(index=False), flush=True)

    print("\n=== Mean fraction of the k shown items that are a real interaction "
          "(per user, averaged over draws), by (strategy, k) ===", flush=True)
    df['frac_shown_real'] = df['n_real_interactions'] / df['n_shown']
    per_draw2 = df.groupby(['strategy', 'k', 'draw'])['frac_shown_real'].mean().reset_index()
    summary2 = (per_draw2.groupby(['strategy', 'k'])['frac_shown_real']
                .agg(['mean', 'std']).reset_index())
    print(summary2.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
