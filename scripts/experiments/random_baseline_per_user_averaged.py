"""
random_baseline_per_user_averaged.py
=======================================
Builds a per-user Random-baseline dataset, averaged over N_RANDOM_DRAWS
independent item-set draws -- the same corrected methodology now used
by scripts/model/baseline_ranking_metrics.py (see that script's
docstring "Random is averaged over N_RANDOM_DRAWS independent item-set
draws"), but at PER-USER granularity instead of the aggregate-only
output in results/baseline_results.csv.

Why this is needed
--------------------
A paired significance test against Random needs one RMSE/HR@10/NDCG@10
value PER USER PER k, not just the aggregate mean in
results/baseline_results.csv. A single-draw Random evaluation has real
per-user sampling variance (see random_baseline_variance_test.py), so
this script instead builds the per-user dataset at the same
multi-draw-averaged granularity as baseline_ranking_metrics.py.

Averaging rule: for each (user, k), average that user's own RMSE (and
HR@5/HR@10/NDCG@5/NDCG@10) across whichever of the N_RANDOM_DRAWS draws
they were actually evaluated in (a user is skipped in a given draw only
if none of their few real recorded interactions land in that draw's
test half -- see split_unseen_items). ``n_draws_observed`` records how
many of the 30 draws contributed to each row, for transparency.

Usage
-----
    python scripts/experiments/random_baseline_per_user_averaged.py

Input
-----
    results/base_model_cache.pkl

Output
------
    results/random_baseline_per_user_averaged_results.csv
        Columns: family, strategy, k, user, rmse, hr5, hr10, ndcg5,
        ndcg10, n_draws_observed
"""

from __future__ import annotations

import os
import pickle
import random
import time
from collections import defaultdict, namedtuple
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from surprise import accuracy

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR: str = os.path.join(SCRIPT_DIR, '..', 'model')
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
MODEL_CACHE: str = os.path.join(RESULTS_DIR, 'base_model_cache.pkl')
OUT_CSV: str = os.path.join(RESULTS_DIR, 'random_baseline_per_user_averaged_results.csv')

N_NEG: int = 99
K_VALUES: List[int] = [10, 25, 50, 100]
NUM_EVAL_USERS: int = 1000
N_RANDOM_DRAWS: int = 30


def _stable_seed(u: int, shown: List[Any]) -> int:
    """Identical to baseline_ranking_metrics.py's own function -- see
    README.md "Reproducibility"."""
    import hashlib
    key = f"{int(u)}|" + ','.join(sorted(str(x) for x in shown))
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def main() -> None:
    """Runs N_RANDOM_DRAWS independent Random-baseline draws per k,
    records per-user metrics for each draw, averages them per (user, k),
    and writes ``results/random_baseline_per_user_averaged_results.csv``.
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

    print(f"[{time.time()-t_start:6.1f}s] Cache loaded.", flush=True)
    eval_cold_users = cold_users[:min(NUM_EVAL_USERS, len(cold_users))]

    eval_users_set = set(int(u) for u in eval_cold_users)
    user_data = data[data['user_idx'].isin(eval_users_set)]
    user_item_interaction: Dict[int, Dict[Any, float]] = {
        int(u): dict(zip(g['itemId'].values, g['interaction'].values))
        for u, g in user_data.groupby('user_idx')
    }
    print(f"[{time.time()-t_start:6.1f}s] Fast per-user lookup built "
          f"({len(user_item_interaction)} users).", flush=True)

    neg_rng = np.random.RandomState(42)
    Pred = namedtuple('Prediction', ['uid', 'iid', 'r_ui', 'est', 'details'])

    def baseline_predict(iid: Any) -> float:
        i_inner = item_to_iidx.get(iid)
        if i_inner is None:
            return mu_base
        return float(np.clip(mu_base + svd_base.bi[i_inner], 0, 1))

    def split_unseen_items(u: int, shown: List[Any],
                            val_frac: float = 0.5) -> Tuple[List[Any], List[Any]]:
        shown_set = set(shown)
        unseen = [iid for iid in eligible_items if iid not in shown_set]
        rng = np.random.RandomState(_stable_seed(u, shown))
        rng.shuffle(unseen)
        n_val = int(val_frac * len(unseen))
        return unseen[:n_val], unseen[n_val:]

    def sampled_metrics_at_k(pos_score: float, neg_scores: List[float],
                              k_list: List[int]) -> Dict[str, float]:
        rank = 1 + sum(1 for s in neg_scores if s > pos_score)
        out: Dict[str, float] = {}
        for kk in k_list:
            out[f'HR@{kk}'] = 1.0 if rank <= kk else 0.0
            out[f'NDCG@{kk}'] = (1.0 / np.log2(rank + 1)) if rank <= kk else 0.0
        return out

    # (user, k) -> list of per-draw values
    per_user_k_rmse: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    per_user_k_hr5: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    per_user_k_hr10: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    per_user_k_ndcg5: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    per_user_k_ndcg10: Dict[Tuple[int, int], List[float]] = defaultdict(list)

    for k in K_VALUES:
        print(f"\n=== k={k}: {N_RANDOM_DRAWS} independent draws, per-user ===", flush=True)
        t_k = time.time()
        for draw_seed in range(N_RANDOM_DRAWS):
            shown = random.Random(draw_seed).sample(eligible_items, k)
            shown_set = set(shown)

            for u in eval_cold_users:
                u_int = int(u)
                user_dict = user_item_interaction.get(u_int)
                if not user_dict:
                    continue
                _, test_items = split_unseen_items(u, shown, val_frac=0.5)
                test_set = set(test_items)
                test_rows = [(iid, r) for iid, r in user_dict.items() if iid in test_set]
                if not test_rows:
                    continue

                preds = [
                    Pred(uid=u, iid=iid, r_ui=r, est=baseline_predict(iid), details={})
                    for iid, r in test_rows
                ]
                per_user_k_rmse[(u_int, k)].append(accuracy.rmse(preds, verbose=False))

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
                    per_user_k_hr5[(u_int, k)].append(float(np.mean(user_hr5)))
                    per_user_k_hr10[(u_int, k)].append(float(np.mean(user_hr10)))
                    per_user_k_ndcg5[(u_int, k)].append(float(np.mean(user_ndcg5)))
                    per_user_k_ndcg10[(u_int, k)].append(float(np.mean(user_ndcg10)))

        print(f"  k={k} total time: {time.time()-t_k:.1f}s", flush=True)

    rows = []
    for (u, k), rmse_list in per_user_k_rmse.items():
        hr5_list = per_user_k_hr5.get((u, k), [])
        hr10_list = per_user_k_hr10.get((u, k), [])
        ndcg5_list = per_user_k_ndcg5.get((u, k), [])
        ndcg10_list = per_user_k_ndcg10.get((u, k), [])
        rows.append({
            'family': 'baseline', 'strategy': 'random', 'k': k, 'user': u,
            'rmse': float(np.mean(rmse_list)),
            'hr5': float(np.mean(hr5_list)) if hr5_list else float('nan'),
            'hr10': float(np.mean(hr10_list)) if hr10_list else float('nan'),
            'ndcg5': float(np.mean(ndcg5_list)) if ndcg5_list else float('nan'),
            'ndcg10': float(np.mean(ndcg10_list)) if ndcg10_list else float('nan'),
            'n_draws_observed': len(rmse_list),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n{'='*70}", flush=True)
    print("=== Sanity check: per-user-averaged mean vs. production "
          "baseline_results.csv (should be close) ===", flush=True)
    for k in K_VALUES:
        sub = df[df['k'] == k]
        print(f"  k={k}: n_users={len(sub)}  mean_rmse={sub['rmse'].mean():.4f}  "
              f"mean_hr10={sub['hr10'].mean():.4f}  "
              f"mean_n_draws_observed={sub['n_draws_observed'].mean():.1f}/{N_RANDOM_DRAWS}",
              flush=True)

    print(f"\nSaved to {OUT_CSV}", flush=True)
    print("NOTE: this dataset is not yet wired into any significance test "
          "-- built for future use only, per explicit instruction.", flush=True)
    print(f"TOTAL TIME: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
