"""
baseline_ranking_metrics.py
=============================
Evaluates the three non-personalised baseline strategies -- Random,
Popularity, and PopError -- on RMSE and sampled ranking metrics
(HR@K, NDCG@K), using the identical methodology, test-item split, and
1,000-user evaluation population as personalised_strategies.py, so the
two families of results are directly comparable (see README.md
"Reproducing Thesis Results" -> Table 4.1).

Baseline prediction is ``mu + b_i`` (the item's own learned bias, from
the frozen base model) -- no personalisation term (thesis Section 3.5).
Random, Popularity, and PopError differ only in which fixed set of k
items they present to every cold user, chosen *before* any of that
user's responses are observed -- this is the defining structural
difference from the personalised strategies in
personalised_strategies.py, which adapt their selection to each user's
revealed feedback as it arrives (thesis Section 3.5, opening paragraph).

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
        HR@5(sampled), HR@10(sampled), NDCG@5(sampled), NDCG@10(sampled)

Complexity
----------
O(|strategies| * |K_VALUES| * NUM_EVAL_USERS * N_NEG) predicted-score
evaluations; each ``baseline_predict`` call is O(1) (a single array
lookup), so this is far cheaper per-cell than the personalised
strategies' O(k * F) per-user active-learning loop.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import random
import time
from collections import namedtuple
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from surprise import accuracy

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
MODEL_CACHE: str = os.path.join(RESULTS_DIR, 'base_model_cache.pkl')
OUT_BASELINE: str = os.path.join(RESULTS_DIR, 'baseline_results.csv')

# Number of sampled negatives per positive item (He et al., 2017
# methodology; thesis Section 3.9).
N_NEG: int = 99
K_VALUES: List[int] = [10, 25, 50, 100]
NUM_EVAL_USERS: int = 1000


def _stable_seed(u: int, shown: List[Any]) -> int:
    """Deterministic replacement for Python's built-in ``hash()`` on
    strings -- see README.md "Reproducibility" for why this matters
    (hash() on strings is randomised per-process unless
    PYTHONHASHSEED is fixed).

    Parameters
    ----------
    u : int
        User index.
    shown : list
        Raw itemId values shown to this user (order-independent).

    Returns
    -------
    int
        A deterministic 32-bit seed derived from ``(u, shown)``.
    """
    key = f"{int(u)}|" + ','.join(sorted(str(x) for x in shown))
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def main() -> None:
    """Evaluates all three baselines at all four elicitation budgets
    and writes ``results/baseline_results.csv``.

    For each (strategy, k) cell: selects the fixed k-item set (thesis
    Section 3.5), then for each of the first ``NUM_EVAL_USERS`` cold
    users, splits their remaining unseen items into a validation/test
    half (identical protocol to personalised_strategies.py, so the
    populations are directly comparable), scores RMSE on the test half,
    and HR@{5,10}/NDCG@{5,10} against ``N_NEG`` sampled negatives per
    held-out positive item.
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

    warm_data = data[~data['user_idx'].isin(cold_users)]
    item_counts = warm_data['itemId'].value_counts()
    item_mean_interaction = warm_data.groupby('itemId')['interaction'].mean()
    error_scores: Dict[Any, float] = {}
    for item in eligible_items:
        p = float(item_mean_interaction.get(item, 0.5))
        error_scores[item] = min(p, 1.0 - p)
    # PopError(i) = ALPHA * log10(freq(i)) + (1-ALPHA) * MisclassError(i)
    # -- thesis Eq. 3.2. ALPHA was learned in build_model_cache.py.
    poperror_scores: Dict[Any, float] = {
        item: ALPHA * np.log10(item_counts[item]) + (1 - ALPHA) * error_scores[item]
        for item in eligible_items
    }

    random.seed(1)
    neg_rng = np.random.RandomState(42)
    Pred = namedtuple('Prediction', ['uid', 'iid', 'r_ui', 'est', 'details'])

    def select_items(strategy: str, k: int) -> List[Any]:
        """Selects the fixed set of k items shown to every cold user
        under this baseline strategy (thesis Section 3.5).

        Parameters
        ----------
        strategy : {'random', 'popularity', 'poperror'}
        k : int
            Number of items to select.

        Returns
        -------
        list
            Raw itemId values (same set for every cold user this cell).
        """
        if strategy == 'random':
            return random.sample(eligible_items, k)
        elif strategy == 'popularity':
            return list(item_counts.loc[eligible_items].sort_values(ascending=False).head(k).index)
        elif strategy == 'poperror':
            return sorted(poperror_scores, key=poperror_scores.get, reverse=True)[:k]
        raise ValueError(f"Unknown strategy: {strategy}")

    def split_unseen_items(u: int, shown: List[Any],
                            val_frac: float = 0.5) -> Tuple[List[Any], List[Any]]:
        """Splits a cold user's remaining unseen items into a
        validation half and a test half, deterministically seeded per
        ``(u, shown)``. Identical protocol to
        personalised_strategies.py's ``_split_unseen_items``, so the
        two result populations are directly comparable.
        """
        shown_set = set(shown)
        unseen = [iid for iid in eligible_items if iid not in shown_set]
        rng = np.random.RandomState(_stable_seed(u, shown))
        rng.shuffle(unseen)
        n_val = int(val_frac * len(unseen))
        return unseen[:n_val], unseen[n_val:]

    def baseline_predict(iid: Any) -> float:
        """Predicts the interaction score for item ``iid`` under the
        non-personalised baseline formula ``mu + b_i`` (thesis
        Section 3.5), clipped to [0, 1]. Returns ``mu_base`` alone if
        the item is absent from the trained model's item index.
        """
        i_inner = item_to_iidx.get(iid)
        if i_inner is None:
            return mu_base
        return float(np.clip(mu_base + svd_base.bi[i_inner], 0, 1))

    def sampled_metrics_at_k(pos_score: float, neg_scores: List[float],
                              k_list: List[int]) -> Dict[str, float]:
        """Computes sampled HR@K and NDCG@K for one positive item
        against sampled negatives (He et al., 2017; thesis Eq.
        3.15-3.16). See personalised_strategies.py's
        ``_sampled_metrics_at_k`` for the identical logic used on the
        personalised side.
        """
        rank = 1 + sum(1 for s in neg_scores if s > pos_score)
        out: Dict[str, float] = {}
        for k in k_list:
            out[f'HR@{k}'] = 1.0 if rank <= k else 0.0
            out[f'NDCG@{k}'] = (1.0 / np.log2(rank + 1)) if rank <= k else 0.0
        return out

    results: List[Dict[str, Any]] = []
    for strat in ['random', 'popularity', 'poperror']:
        for k in K_VALUES:
            shown = select_items(strat, k)
            shown_set = set(shown)

            per_user_rmse: List[float] = []
            per_user_hr5: List[float] = []
            per_user_hr10: List[float] = []
            per_user_ndcg5: List[float] = []
            per_user_ndcg10: List[float] = []

            for u in eval_cold_users:
                _, test_items = split_unseen_items(u, shown, val_frac=0.5)
                test_df = data[(data['user_idx'] == u) & (data['itemId'].isin(test_items))]
                if len(test_df) == 0:
                    continue

                preds = [
                    Pred(uid=row.user_idx, iid=row.item_idx, r_ui=row.interaction,
                         est=baseline_predict(row.itemId), details={})
                    for row in test_df.itertuples()
                ]
                per_user_rmse.append(accuracy.rmse(preds, verbose=False))

                user_interacted = set(data[data['user_idx'] == u]['itemId'].tolist())
                candidate_negs = [
                    iid for iid in eligible_items
                    if iid not in user_interacted and iid not in shown_set
                    and iid in item_to_iidx
                ]
                pos_test_iids = test_df[test_df['interaction'] == 1]['itemId'].tolist()

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

            row = {
                'Strategy': strat, 'ItemsShown': k,
                'ColdUsersEvaluated': len(per_user_rmse),
                'RMSE': round(np.mean(per_user_rmse), 4) if per_user_rmse else float('nan'),
                'HR@5(sampled)': round(np.mean(per_user_hr5), 4) if per_user_hr5 else float('nan'),
                'HR@10(sampled)': round(np.mean(per_user_hr10), 4) if per_user_hr10 else float('nan'),
                'NDCG@5(sampled)': round(np.mean(per_user_ndcg5), 4) if per_user_ndcg5 else float('nan'),
                'NDCG@10(sampled)': round(np.mean(per_user_ndcg10), 4) if per_user_ndcg10 else float('nan'),
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
