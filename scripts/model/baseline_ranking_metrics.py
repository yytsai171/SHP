"""
baseline_ranking_metrics.py
=============================
Evaluates three non-personalised baseline strategies -- Random,
Popularity, and PopError -- on RMSE and sampled ranking metrics
(HR@K, NDCG@K), using the same test-item split and evaluation
population as personalised_strategies.py, so the two sets of results
are directly comparable.

Baseline prediction is ``mu + b_i`` (the item's own learned bias, from
the frozen base model) -- no personalisation term. Random, Popularity,
and PopError differ only in which fixed set of k items they present to
every cold user, chosen before any of that user's responses are
observed -- unlike the personalised strategies, which adapt their
selection as each user's feedback arrives.

All three baselines are averaged over N_RANDOM_DRAWS independent draws,
each contributing its own coherent (population, RMSE/HR/NDCG) estimate
that gets averaged into the final number -- never pooled across draws
into one larger population, which would silently change the evaluated
population size relative to every other strategy in this study. For
Random, the randomness is in which items get shown, so each draw picks
a fresh independent item set (the val/test split then follows
deterministically from that draw's shown set). Popularity and PopError
select items deterministically, so their draws instead reshuffle the
val/test split of the (fixed) unseen items, via an explicit ``draw``
index folded into the split seed.

Uses a one-time per-user interaction lookup dict, built once before
the (strategy, k, draw) loop, instead of filtering the interaction
table per user per call.

Usage
-----
    python scripts/model/baseline_ranking_metrics.py

Input
-----
    results/base_model_cache.pkl   (see build_model_cache.py)

Output
------
    results/baseline_results.csv
        Columns: Strategy, ItemsShown, ColdUsersEvaluated, RMSE,
        HR@5(sampled), HR@10(sampled), NDCG@5(sampled), NDCG@10(sampled).
        Every numeric column is the mean across N_RANDOM_DRAWS
        independent draws (item-set draws for Random; val/test-split
        draws for Popularity/PopError). ColdUsersEvaluated is the
        rounded mean of each draw's own population size.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import random
import time
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from surprise import accuracy

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
MODEL_CACHE: str = os.path.join(RESULTS_DIR, 'base_model_cache.pkl')
OUT_BASELINE: str = os.path.join(RESULTS_DIR, 'baseline_results.csv')

# Number of sampled negatives per positive item (He et al., 2017
# methodology).
N_NEG: int = 99
K_VALUES: List[int] = [10, 25, 50, 100]
NUM_EVAL_USERS: int = 1000
# Number of independent draws every baseline's RMSE/HR/NDCG are
# averaged over: item-set draws for Random, val/test-split draws for
# Popularity/PopError (see module docstring).
N_RANDOM_DRAWS: int = 30


def _stable_seed(u: int, shown: List[Any], draw: Optional[int] = None) -> int:
    """Deterministic replacement for Python's built-in ``hash()`` on
    strings, which is randomised per-process unless PYTHONHASHSEED is
    fixed.

    Parameters
    ----------
    u : int
        User index.
    shown : list
        Raw itemId values shown to this user (order-independent).
    draw : int, optional
        Distinguishes independent val/test-split draws for a strategy
        whose ``shown`` set doesn't itself vary by draw (Popularity,
        PopError). Left as ``None`` for Random, whose ``shown`` already
        differs per draw, so the split varies as a side effect without
        needing this folded into the seed too -- keeps Random's
        per-draw seeds byte-identical to before this parameter existed.

    Returns
    -------
    int
        A deterministic 32-bit seed derived from ``(u, shown[, draw])``.
    """
    key = f"{int(u)}|" + ','.join(sorted(str(x) for x in shown))
    if draw is not None:
        key = f"{key}|draw={draw}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def main() -> None:
    """Evaluates all three baselines at all four elicitation budgets
    and writes ``results/baseline_results.csv``.

    For each (strategy, k) cell: runs N_RANDOM_DRAWS independent draws
    (item-set draws for Random, val/test-split draws of the same fixed
    item set for Popularity/PopError). Each draw selects (or reuses)
    its k-item set, then for each of the first ``NUM_EVAL_USERS`` cold
    users, splits their remaining unseen items into a validation/test
    half, scores RMSE on the test half, and HR@{5,10}/NDCG@{5,10}
    against ``N_NEG`` sampled negatives per held-out positive item. The
    N_RANDOM_DRAWS per-draw results are then averaged into one row.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.time()

    with open(MODEL_CACHE, 'rb') as f:
        cache: Dict[str, Any] = pickle.load(f)

    data = cache['data']
    eligible_items = cache['eligible_items']
    item_to_iidx = cache['item_to_iidx']
    svd_base = cache['svd_base']
    mu_base = cache['mu_base']
    cold_users = cache['cold_users']
    ALPHA = cache['ALPHA']

    print(f"[{time.time()-t_start:6.1f}s] Cache loaded (setup skipped).", flush=True)

    eval_cold_users = cold_users[:min(NUM_EVAL_USERS, len(cold_users))]

    # Per-user interaction lookup, built once here instead of filtering
    # the full DataFrame per user per call.
    eval_users_set = set(int(u) for u in eval_cold_users)
    user_data = data[data['user_idx'].isin(eval_users_set)]
    user_item_interaction: Dict[int, Dict[Any, float]] = {
        int(u): dict(zip(g['itemId'].values, g['interaction'].values))
        for u, g in user_data.groupby('user_idx')
    }
    print(f"[{time.time()-t_start:6.1f}s] Fast per-user lookup built "
          f"({len(user_item_interaction)} users).", flush=True)

    warm_data = data[~data['user_idx'].isin(cold_users)]
    item_counts = warm_data['itemId'].value_counts()
    item_mean_interaction = warm_data.groupby('itemId')['interaction'].mean()
    error_scores: Dict[Any, float] = {}
    for item in eligible_items:
        p = float(item_mean_interaction.get(item, 0.5))
        error_scores[item] = min(p, 1.0 - p)
    # PopError(i) = ALPHA * log10(freq(i)) + (1-ALPHA) * MisclassError(i).
    # ALPHA was learned in build_model_cache.py.
    poperror_scores: Dict[Any, float] = {
        item: ALPHA * np.log10(item_counts[item]) + (1 - ALPHA) * error_scores[item]
        for item in eligible_items
    }

    neg_rng = np.random.RandomState(42)
    Pred = namedtuple('Prediction', ['uid', 'iid', 'r_ui', 'est', 'details'])

    def select_items(strategy: str, k: int, draw_seed: Optional[int] = None) -> List[Any]:
        """Selects a fixed set of k items to show to every cold user
        under this baseline strategy.

        Parameters
        ----------
        strategy : {'random', 'popularity', 'poperror'}
        k : int
            Number of items to select.
        draw_seed : int, optional
            For ``strategy='random'`` only: seeds an independent draw
            via a local ``random.Random`` instance, so that
            ``N_RANDOM_DRAWS`` calls with different seeds are
            genuinely independent samples. Ignored for the
            deterministic strategies.

        Returns
        -------
        list
            Raw itemId values (same set for every cold user this cell
            -- and, for 'random', this one draw).
        """
        if strategy == 'random':
            return random.Random(draw_seed).sample(eligible_items, k)
        elif strategy == 'popularity':
            return list(item_counts.loc[eligible_items].sort_values(ascending=False).head(k).index)
        elif strategy == 'poperror':
            return sorted(poperror_scores, key=poperror_scores.get, reverse=True)[:k]
        raise ValueError(f"Unknown strategy: {strategy}")

    def split_unseen_items(u: int, shown: List[Any], val_frac: float = 0.5,
                            draw: Optional[int] = None) -> Tuple[List[Any], List[Any]]:
        """Splits a cold user's remaining unseen items into a
        validation half and a test half, deterministically seeded per
        ``(u, shown[, draw])``. With ``draw=None``, matches the exact
        single-split seed used everywhere else in this codebase
        (personalised_strategies.py's ``_split_unseen_items``, and
        Random's per-draw calls here, whose ``shown`` already encodes
        the draw). Popularity/PopError pass an explicit ``draw`` so
        their fixed ``shown`` set still yields independent splits.
        """
        shown_set = set(shown)
        unseen = [iid for iid in eligible_items if iid not in shown_set]
        rng = np.random.RandomState(_stable_seed(u, shown, draw))
        rng.shuffle(unseen)
        n_val = int(val_frac * len(unseen))
        return unseen[:n_val], unseen[n_val:]

    def baseline_predict(iid: Any) -> float:
        """Predicts the interaction score for item ``iid`` under the
        non-personalised baseline formula ``mu + b_i``, clipped to
        [0, 1]. Returns ``mu_base`` alone if the item is absent from
        the trained model's item index.
        """
        i_inner = item_to_iidx.get(iid)
        if i_inner is None:
            return mu_base
        return float(np.clip(mu_base + svd_base.bi[i_inner], 0, 1))

    def sampled_metrics_at_k(pos_score: float, neg_scores: List[float],
                              k_list: List[int]) -> Dict[str, float]:
        """Computes sampled HR@K and NDCG@K for one positive item
        against sampled negatives (He et al., 2017).
        """
        rank = 1 + sum(1 for s in neg_scores if s > pos_score)
        out: Dict[str, float] = {}
        for k in k_list:
            out[f'HR@{k}'] = 1.0 if rank <= k else 0.0
            out[f'NDCG@{k}'] = (1.0 / np.log2(rank + 1)) if rank <= k else 0.0
        return out

    def evaluate_shown_set(shown: List[Any], draw: Optional[int] = None) -> Dict[str, float]:
        """Evaluates one fixed item set against every evaluated cold
        user, returning the aggregate RMSE/HR/NDCG for that one draw.
        Called once per independent draw for every strategy (see
        ``N_RANDOM_DRAWS``): for Random, ``shown`` itself differs per
        draw and ``draw`` is left ``None``; for Popularity/PopError,
        ``shown`` is fixed and ``draw`` varies the val/test split.
        """
        shown_set = set(shown)
        per_user_rmse: List[float] = []
        per_user_hr5: List[float] = []
        per_user_hr10: List[float] = []
        per_user_ndcg5: List[float] = []
        per_user_ndcg10: List[float] = []

        for u in eval_cold_users:
            user_dict = user_item_interaction.get(int(u))
            if not user_dict:
                continue
            _, test_items = split_unseen_items(u, shown, val_frac=0.5, draw=draw)
            test_set = set(test_items)
            test_rows = [(iid, r) for iid, r in user_dict.items() if iid in test_set]
            if not test_rows:
                continue

            preds = [
                Pred(uid=u, iid=iid, r_ui=r, est=baseline_predict(iid), details={})
                for iid, r in test_rows
            ]
            per_user_rmse.append(accuracy.rmse(preds, verbose=False))

            user_interacted = set(user_dict.keys())
            candidate_negs = [
                iid for iid in eligible_items
                if iid not in user_interacted and iid not in shown_set
                and iid in item_to_iidx
            ]
            pos_test_iids = [iid for iid, r in test_rows if r == 1]

            user_hr5: List[float] = []
            user_hr10: List[float] = []
            user_ndcg5: List[float] = []
            user_ndcg10: List[float] = []
            for pos_iid in pos_test_iids:
                pos_score = baseline_predict(pos_iid)
                n_sample = min(N_NEG, len(candidate_negs))
                if n_sample == 0:
                    continue
                sampled_neg_iids = neg_rng.choice(candidate_negs, size=n_sample, replace=False)
                neg_scores = [baseline_predict(nid) for nid in sampled_neg_iids]
                m = sampled_metrics_at_k(pos_score, neg_scores, k_list=[5, 10])
                user_hr5.append(m['HR@5']); user_hr10.append(m['HR@10'])
                user_ndcg5.append(m['NDCG@5']); user_ndcg10.append(m['NDCG@10'])

            if user_hr5:
                per_user_hr5.append(float(np.mean(user_hr5)))
                per_user_hr10.append(float(np.mean(user_hr10)))
                per_user_ndcg5.append(float(np.mean(user_ndcg5)))
                per_user_ndcg10.append(float(np.mean(user_ndcg10)))

        return {
            'n_users': len(per_user_rmse),
            'rmse': float(np.mean(per_user_rmse)) if per_user_rmse else float('nan'),
            'hr5': float(np.mean(per_user_hr5)) if per_user_hr5 else float('nan'),
            'hr10': float(np.mean(per_user_hr10)) if per_user_hr10 else float('nan'),
            'ndcg5': float(np.mean(per_user_ndcg5)) if per_user_ndcg5 else float('nan'),
            'ndcg10': float(np.mean(per_user_ndcg10)) if per_user_ndcg10 else float('nan'),
        }

    results: List[Dict[str, Any]] = []
    for strat in ['random', 'popularity', 'poperror']:
        for k in K_VALUES:
            if strat == 'random':
                # Average over N_RANDOM_DRAWS independent item-set
                # draws; shown differs per draw, so draw=None (the
                # split varies as a side effect of shown varying).
                draws = [evaluate_shown_set(select_items('random', k, draw_seed=s))
                         for s in range(N_RANDOM_DRAWS)]
            else:
                # Popularity/PopError: shown is fixed (deterministic
                # ranking), so average over N_RANDOM_DRAWS independent
                # val/test-split draws of that same fixed shown set.
                shown = select_items(strat, k)
                draws = [evaluate_shown_set(shown, draw=d) for d in range(N_RANDOM_DRAWS)]

            row = {
                'Strategy': strat, 'ItemsShown': k,
                'ColdUsersEvaluated': int(round(np.mean([d['n_users'] for d in draws]))),
                'RMSE': round(float(np.mean([d['rmse'] for d in draws])), 4),
                'HR@5(sampled)': round(float(np.mean([d['hr5'] for d in draws])), 4),
                'HR@10(sampled)': round(float(np.mean([d['hr10'] for d in draws])), 4),
                'NDCG@5(sampled)': round(float(np.mean([d['ndcg5'] for d in draws])), 4),
                'NDCG@10(sampled)': round(float(np.mean([d['ndcg10'] for d in draws])), 4),
            }
            results.append(row)
            print(f"[{time.time()-t_start:6.1f}s] [{strat}][k={k}] "
                  f"RMSE={row['RMSE']:.4f} HR@10={row['HR@10(sampled)']:.4f}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(OUT_BASELINE, index=False)
    print("\n=== Baseline ranking metrics ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {OUT_BASELINE}", flush=True)
    print(f"TOTAL TIME: {time.time()-t_start:.1f}s", flush=True)


if __name__ == '__main__':
    main()
