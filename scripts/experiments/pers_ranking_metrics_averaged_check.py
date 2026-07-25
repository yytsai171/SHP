"""
pers_ranking_metrics_averaged_check.py
========================================
Companion to pers_valtest_averaged_check.py, but for HR@K/NDCG@K
instead of RMSE.

tab:pers_rank (HR@5, HR@10, NDCG@5, NDCG@10 for all four personalised
strategies at all four k) has never been checked for single-draw
sensitivity at all. It's exposed to the val/test-split issue already
confirmed to matter for RMSE (positive test items are drawn from the
test half), plus a second, separate single-draw source specific to
ranking metrics: the N_NEG=99 sampled negatives per positive test item
are also drawn once and never resampled.

Same "run once, rescore N_DRAWS times" design: the active-learning
loop runs once per user (item selection doesn't depend on the
eventual test split or negative sample), then each of the N_DRAWS
draws reshuffles BOTH the val/test split and the negative sample
together (seeded jointly per draw), so this captures the combined
effect of both currently-unaveraged noise sources on the ranking
metrics.

Usage
-----
    python scripts/experiments/pers_ranking_metrics_averaged_check.py

Input
-----
    results/base_model_cache.pkl

Output
------
    results/pers_ranking_metrics_averaged_results.csv
        Columns: strategy, k, draw, hr5, hr10, ndcg5, ndcg10, n_users
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR: str = os.path.join(SCRIPT_DIR, '..', 'model')
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
OUT_CSV: str = os.path.join(RESULTS_DIR, 'pers_ranking_metrics_averaged_results.csv')

WORK: List[Tuple[str, int]] = [
    ('SHHCP', 10), ('SHHCP', 25), ('SHHCP', 50), ('SHHCP', 100),
    ('SHLCP', 10), ('SHLCP', 25), ('SHLCP', 50), ('SHLCP', 100),
    ('SHMCP', 10), ('SHMCP', 25), ('SHMCP', 50), ('SHMCP', 100),
    ('SHECP', 10), ('SHECP', 25), ('SHECP', 50), ('SHECP', 100),
]
SINGLE_DRAW_HR10: Dict[Tuple[str, int], float] = {
    ('SHHCP', 10): 0.1909, ('SHHCP', 25): 0.2065, ('SHHCP', 50): 0.1929, ('SHHCP', 100): 0.1961,
    ('SHLCP', 10): 0.1977, ('SHLCP', 25): 0.1904, ('SHLCP', 50): 0.1857, ('SHLCP', 100): 0.1942,
    ('SHMCP', 10): 0.1873, ('SHMCP', 25): 0.1853, ('SHMCP', 50): 0.2088, ('SHMCP', 100): 0.1774,
    ('SHECP', 10): 0.2081, ('SHECP', 25): 0.2031, ('SHECP', 50): 0.1935, ('SHECP', 100): 0.1935,
}
NUM_USERS: int = 1000
N_DRAWS: int = 30
N_NEG: int = 99
N_WORKERS: int = 10


def _stable_seed(u: int, shown: List[Any], draw: int, tag: str) -> int:
    key = f"{int(u)}|{draw}|{tag}|" + ','.join(sorted(str(x) for x in shown))
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def _worker_init_local(eval_users) -> None:
    if MODEL_DIR not in sys.path:
        sys.path.insert(0, MODEL_DIR)
    import personalised_strategies as pers
    pers._worker_init(eval_users)


def run_session_and_score(work_item: Tuple[str, int, int]) -> Dict[str, Any]:
    import personalised_strategies as pers

    strategy, k, u = work_item
    cache = pers._cache
    user_dict        = pers._user_item_interaction.get(int(u), {})
    eligible_items    = cache['eligible_items']
    item_to_iidx      = cache['item_to_iidx']
    most_popular_iid  = cache['most_popular_iid']
    i_0_inner         = cache['i_0_inner']
    svd_base          = cache['svd_base']
    mu_base           = cache['mu_base']
    n_factors         = cache['n_factors']
    GAMMA1, GAMMA2    = cache['GAMMA1'], cache['GAMMA2']
    pers.mu_base_global[0] = mu_base

    alpha = pers._shrink_alpha(k)
    egreedy_rng_local = pers._seeded_rng(u, strategy, k, 'egreedy')

    if k in pers.ZERO_INIT_K_VALUES:
        pu_cold = np.zeros(n_factors)
    elif i_0_inner is not None:
        pu_cold = svd_base.qi[i_0_inner].copy()
    else:
        pu_cold = np.zeros(n_factors)
    bu_cold = 0.0
    local_qi: Dict[int, np.ndarray] = {}
    local_bi: Dict[int, float] = {}
    shown = [most_popular_iid]

    has_first = most_popular_iid in user_dict
    r_first = float(user_dict[most_popular_iid]) if has_first else 0.0
    if i_0_inner is not None and has_first:
        pu_cold, bu_cold = pers._partial_lfm_update_cold(
            svd_base, pu_cold, bu_cold, i_0_inner, r_first, local_qi, local_bi,
            GAMMA1, GAMMA2, pers.LMBDA1, pers.LMBDA2,
            num_sgd_steps=pers.NUM_SGD_STEPS
        )

    round_number = 0
    while len(shown) < k:
        b = min(pers.BATCH_SIZE, k - len(shown))
        batch = pers._select_batch(svd_base, pu_cold, bu_cold, shown, eligible_items,
                                    item_to_iidx, local_qi, local_bi, strategy, b,
                                    round_number, egreedy_rng_local)
        if not batch:
            break
        shown.extend(batch)
        for item in batch:
            has_row = item in user_dict
            r_ui = float(user_dict[item]) if has_row else 0.0
            i_inner = item_to_iidx.get(item)
            if i_inner is not None and has_row:
                pu_cold, bu_cold = pers._partial_lfm_update_cold(
                    svd_base, pu_cold, bu_cold, i_inner, r_ui, local_qi, local_bi,
                    GAMMA1, GAMMA2, pers.LMBDA1, pers.LMBDA2,
                    num_sgd_steps=pers.NUM_SGD_STEPS
                )
        round_number += 1

    shown_set = set(shown)
    unseen = [iid for iid in eligible_items if iid not in shown_set]
    user_interacted = set(user_dict.keys())
    candidate_negs = [iid for iid in eligible_items
                      if iid not in user_interacted and iid not in shown_set
                      and iid in item_to_iidx]

    draw_metrics: Dict[int, Dict[str, float]] = {}
    for draw in range(N_DRAWS):
        split_rng = np.random.RandomState(_stable_seed(u, shown, draw, 'split'))
        unseen_shuffled = unseen.copy()
        split_rng.shuffle(unseen_shuffled)
        n_val = int(0.5 * len(unseen_shuffled))
        test_items = unseen_shuffled[n_val:]
        test_set = set(test_items)
        pos_test_iids = [iid for iid, r in user_dict.items() if iid in test_set and r == 1]
        if not pos_test_iids or not candidate_negs:
            continue

        neg_rng = np.random.RandomState(_stable_seed(u, shown, draw, 'neg'))
        hr5s, hr10s, ndcg5s, ndcg10s = [], [], [], []
        for pos_iid in pos_test_iids:
            i_pos = item_to_iidx.get(pos_iid)
            if i_pos is None:
                continue
            qi_pos = local_qi.get(i_pos, svd_base.qi[i_pos])
            bi_pos = local_bi.get(i_pos, svd_base.bi[i_pos])
            pos_score = pers._score(bu_cold, bi_pos, pu_cold, qi_pos, alpha)
            n_sample = min(N_NEG, len(candidate_negs))
            sampled = neg_rng.choice(candidate_negs, size=n_sample, replace=False)
            neg_scores = []
            for nid in sampled:
                ni = item_to_iidx[nid]
                qi_n = local_qi.get(ni, svd_base.qi[ni])
                bi_n = local_bi.get(ni, svd_base.bi[ni])
                neg_scores.append(pers._score(bu_cold, bi_n, pu_cold, qi_n, alpha))
            m = pers._sampled_metrics_at_k(pos_score, neg_scores, k_list=[5, 10])
            hr5s.append(m['HR@5']); hr10s.append(m['HR@10'])
            ndcg5s.append(m['NDCG@5']); ndcg10s.append(m['NDCG@10'])
        if hr5s:
            draw_metrics[draw] = {
                'hr5': float(np.mean(hr5s)), 'hr10': float(np.mean(hr10s)),
                'ndcg5': float(np.mean(ndcg5s)), 'ndcg10': float(np.mean(ndcg10s)),
            }

    return {'strategy': strategy, 'k': k, 'user': int(u), 'draw_metrics': draw_metrics}


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_cache = os.path.join(RESULTS_DIR, 'base_model_cache.pkl')
    with open(model_cache, 'rb') as f:
        cache = pickle.load(f)
    cold_users = cache['cold_users']
    eval_users = cold_users[:NUM_USERS]

    work_items = [(s, k, u) for s, k in WORK for u in eval_users]
    print(f"=== Ranking-metrics averaging check: {len(WORK)} cells, {len(eval_users)} users, "
          f"{N_DRAWS} draws each ===", flush=True)
    t0 = time.time()
    with mp.Pool(processes=N_WORKERS, initializer=_worker_init_local,
                  initargs=(eval_users,)) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_session_and_score, work_items, chunksize=4)):
            results.append(r)
            if (i + 1) % 400 == 0:
                print(f"  {i+1}/{len(work_items)} done ({time.time()-t0:.1f}s)", flush=True)
    elapsed = time.time() - t0
    print(f"Done: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    rows = []
    for strategy, k in WORK:
        cell_results = [r for r in results if r['strategy'] == strategy and r['k'] == k]
        per_user_avg_hr10 = []
        for metric in ['hr5', 'hr10', 'ndcg5', 'ndcg10']:
            per_user_vals = [np.mean([dm[metric] for dm in r['draw_metrics'].values()])
                              for r in cell_results if r['draw_metrics']]
            overall = np.mean(per_user_vals) if per_user_vals else float('nan')
            if metric == 'hr10':
                overall_hr10 = overall
                n_users = len(per_user_vals)
            rows.append({'strategy': strategy, 'k': k, 'metric': metric,
                         'averaged_value': round(overall, 4), 'n_users': len(per_user_vals)})
        print(f"  {strategy} k={k}: HR@10 avg={overall_hr10:.4f} "
              f"(single-draw value: {SINGLE_DRAW_HR10[(strategy, k)]}), n={n_users}", flush=True)

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nSaved to {OUT_CSV}", flush=True)


if __name__ == '__main__':
    main()
