"""
unconstrained_ceiling_test.py
================================
Computes the "unconstrained MF" ceiling used by fMF (Zhou, Yang & Zha,
2011, Section 4.6) as a reference point for how much of a cost cold-start
elicitation-budget constraints actually impose -- i.e., what RMSE would a
cold user get if there were no elicitation budget at all, and their own
real interactions were simply used to fit a proper latent vector
alongside every warm user, exactly as the base model already does for
warm users?

This is the second half of the two-part fMF follow-up (the first,
hierarchical_regularization_test.py, tests a regularization mechanism
borrowed from the same paper). It is NOT a test of any active-learning
strategy: no items are "selected", no partial-SGD update is applied, and
no elicitation budget k is enforced. Each evaluated cold user's own real
interactions with eligible items (except a held-out canonical test half)
are simply added to the training data and a fresh SVD is refit, so every
evaluated cold user gets exactly the same treatment a warm user already
gets -- a genuinely unconstrained personalised profile, fit the same way
the base model itself is fit (same hyperparameters, single fit, no CV
needed since they are already known-good from build_model_cache.py).

Canonical test split (a necessary approximation)
------------------------------------------------------
Every personalised-strategy evaluation in this repo computes its own
train/test split of each cold user's unseen items, seeded by
``_stable_seed(u, shown)`` -- and ``shown`` differs by (strategy, k), so
there is no single "the" test set for a user across every existing
result file. This script defines ONE canonical test split per user,
seeded the same way but with ``shown=[most_popular_iid]`` (i.e., the
state before any elicitation has happened) -- a fixed, reproducible
reference point, not literally identical to any one (strategy, k)
condition's own split. Treat the ceiling RMSE below as an illustrative
upper bound for "cost of personalisation", not a number directly
paired-testable against personalised_results.csv's per-user rows.

Usage
-----
    python scripts/experiments/unconstrained_ceiling_test.py

Input
-----
    results/base_model_cache.pkl

Output
------
    results/unconstrained_ceiling_results.csv
        Columns: user, rmse, n_test_items
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
from surprise import SVD, Dataset, Reader, accuracy

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
MODEL_CACHE: str = os.path.join(RESULTS_DIR, 'base_model_cache.pkl')
OUT_CEILING: str = os.path.join(RESULTS_DIR, 'unconstrained_ceiling_results.csv')

NUM_USERS: int = 1000  # same population evaluated throughout this repo


def _stable_seed(u: int, shown: List[Any]) -> int:
    """Deterministic replacement for Python's built-in ``hash()`` on
    strings -- see README.md "Reproducibility".
    """
    key = f"{int(u)}|" + ','.join(sorted(str(x) for x in shown))
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def main() -> None:
    """Refits an SVD including each evaluated cold user's own real
    interactions (minus their canonical held-out test half), scores each
    user's ceiling RMSE, and writes
    ``results/unconstrained_ceiling_results.csv``.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.time()

    with open(MODEL_CACHE, 'rb') as f:
        cache = pickle.load(f)

    data             = cache['data']
    eligible_items   = cache['eligible_items']
    most_popular_iid = cache['most_popular_iid']
    cold_users       = cache['cold_users']
    best_params      = cache['best_params']

    print(f"[{time.time()-t_start:6.1f}s] Cache loaded.", flush=True)

    eval_users = cold_users[:NUM_USERS]
    eligible_set = set(eligible_items)

    # ── Canonical held-out test split per evaluated user ───────────────────
    # Only rows with a REAL recorded interaction can ever appear in a test
    # split evaluated this way (data only contains recorded rows), so this
    # naturally restricts each user's test set to the eligible items they
    # actually interacted with.
    test_rows_idx: List[int] = []
    user_test_items: Dict[int, List[Any]] = {}
    for u in eval_users:
        user_rows = data[(data['user_idx'] == u) & (data['itemId'].isin(eligible_set))]
        items = user_rows['itemId'].tolist()
        if not items:
            continue
        rng = np.random.RandomState(_stable_seed(u, [most_popular_iid]))
        shuffled = items.copy()
        rng.shuffle(shuffled)
        n_test = max(1, int(0.5 * len(shuffled))) if len(shuffled) > 1 else 0
        test_items = shuffled[:n_test] if n_test > 0 else []
        if not test_items:
            continue
        user_test_items[u] = test_items
        test_rows_idx.extend(
            user_rows[user_rows['itemId'].isin(test_items)].index.tolist()
        )
    print(f"[{time.time()-t_start:6.1f}s] Canonical test split computed for "
          f"{len(user_test_items)}/{len(eval_users)} users "
          f"({len(test_rows_idx)} held-out rows).", flush=True)

    # ── Build the "unconstrained" training set: everyone except the
    #    held-out test rows for evaluated cold users ────────────────────────
    train_data = data.drop(index=test_rows_idx)
    print(f"[{time.time()-t_start:6.1f}s] Unconstrained training set: "
          f"{len(train_data)} rows (full dataset: {len(data)} rows).", flush=True)

    reader = Reader(rating_scale=(0, 1))
    dataset = Dataset.load_from_df(train_data[['user_idx', 'item_idx', 'interaction']], reader)
    trainset = dataset.build_full_trainset()

    svd_unconstrained = SVD(
        n_factors=best_params['n_factors'],
        reg_all=best_params['reg_all'],
        n_epochs=best_params['n_epochs'],
        biased=True,
        random_state=1,
    )
    print(f"[{time.time()-t_start:6.1f}s] Fitting unconstrained SVD "
          f"(n_factors={best_params['n_factors']}, reg_all={best_params['reg_all']})...",
          flush=True)
    svd_unconstrained.fit(trainset)
    print(f"[{time.time()-t_start:6.1f}s] Fit done.", flush=True)

    Pred = namedtuple('Prediction', ['uid', 'iid', 'r_ui', 'est', 'details'])
    mu_new = trainset.global_mean

    rows = []
    for u, test_items in user_test_items.items():
        try:
            u_inner = trainset.to_inner_uid(u)
        except ValueError:
            continue
        pu = svd_unconstrained.pu[u_inner]
        bu = svd_unconstrained.bu[u_inner]

        test_df = data[(data['user_idx'] == u) & (data['itemId'].isin(test_items))]
        preds = []
        for row in test_df.itertuples():
            try:
                i_inner = trainset.to_inner_iid(row.item_idx)
            except ValueError:
                continue
            qi = svd_unconstrained.qi[i_inner]
            bi = svd_unconstrained.bi[i_inner]
            est = float(np.clip(mu_new + bu + bi + np.dot(pu, qi), 0, 1))
            preds.append(Pred(uid=row.user_idx, iid=row.item_idx,
                               r_ui=row.interaction, est=est, details={}))
        if not preds:
            continue
        rmse = accuracy.rmse(preds, verbose=False)
        rows.append({'user': int(u), 'rmse': rmse, 'n_test_items': len(preds)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CEILING, index=False)

    print(f"\n{'='*70}", flush=True)
    print(f"=== Unconstrained ceiling: {len(df)} users scored ===", flush=True)
    print(f"  Mean RMSE = {df['rmse'].mean():.4f}  "
          f"(median n_test_items/user = {df['n_test_items'].median():.1f})", flush=True)
    print(f"\nSaved to {OUT_CEILING}", flush=True)
    print(f"TOTAL TIME: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
