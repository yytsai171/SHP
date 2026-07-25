"""
no_fabricated_negatives_test.py
==================================
Compares two ways of handling a revealed item with no recorded
interaction row for that user:

    r_ui = actual recorded interaction, if a (user, item) row exists
           0.0 (a fabricated "dislike"), otherwise

Given this dataset's sparsity (median cold user has only 2 recorded
interactions across a 45,543-item eligible pool), this fallback would
fire for the large majority of revealed items -- including the very
first item shown in nearly every session.

Two conditions, otherwise using the model's full final methodology
(decaying LR, shrinkage C=100, item-based init):

  - 'fabricate'    : the rejected alternative -- defaults r_ui to 0.0
                     when no row exists.
  - 'no_fabricate' : the adopted default -- skips the partial-SGD
                     update (and does not advance the decaying-LR
                     update counter) whenever the revealed item has no
                     recorded row for this user. Matches
                     process_one_user's actual current behaviour, so
                     this condition reproduces personalised_results.csv
                     exactly and serves as an in-run consistency check.

All four strategies (SHHCP, SHLCP, SHMCP, SHECP), all four k values,
the full 1,000-user population, so the two conditions are directly
comparable to each other and to the model's headline results.

This reuses personalised_strategies.py's own building-block functions
(_worker_init, _select_batch, _partial_lfm_update_cold, _shrink_alpha,
_score, _split_unseen_items, _seeded_rng) via import; it does not
modify that file.

Usage
-----
    python scripts/experiments/no_fabricated_negatives_test.py

Input
-----
    results/base_model_cache.pkl

Output
------
    results/no_fabricated_negatives_test_results.csv
        Columns: condition, strategy, k, user, rmse, hr5, hr10, ndcg5,
        ndcg10, mrr5, mrr10, auc, n_real_updates, n_shown
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from surprise import accuracy

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR: str = os.path.join(SCRIPT_DIR, '..', 'model')
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
OUT_CSV: str = os.path.join(RESULTS_DIR, 'no_fabricated_negatives_test_results.csv')

STRATEGIES: List[str] = ['SHHCP', 'SHLCP', 'SHMCP', 'SHECP']
K_VALUES: List[int] = [10, 25, 50, 100]
NUM_USERS: int = 1000
N_WORKERS: int = 10
N_NEG: int = 99

Pred = namedtuple('Prediction', ['uid', 'iid', 'r_ui', 'est', 'details'])


def _extended_metrics_at_k(pos_score: float, neg_scores: List[float],
                            k_list: List[int]) -> Dict[str, float]:
    """Computes HR@K/NDCG@K/MRR@K for one positive item against sampled
    negatives (kept standalone rather than imported, so this script
    doesn't depend on any other experiment script)."""
    rank = 1 + sum(1 for s in neg_scores if s > pos_score)
    out: Dict[str, float] = {}
    for k in k_list:
        out[f'HR@{k}'] = 1.0 if rank <= k else 0.0
        out[f'NDCG@{k}'] = (1.0 / np.log2(rank + 1)) if rank <= k else 0.0
        out[f'MRR@{k}'] = (1.0 / rank) if rank <= k else 0.0
    n_neg = len(neg_scores)
    out['AUC'] = (n_neg - (rank - 1)) / n_neg if n_neg > 0 else float('nan')
    return out


def _worker_init_local() -> None:
    """Pool initializer: runs personalised_strategies's own
    ``_worker_init`` (populates the worker-local cache/eligible-item
    arrays), leaving that module entirely unmodified."""
    import personalised_strategies as pers
    pers._worker_init()


def process_one_user_variant(work_item: Tuple[str, str, int, int]) -> Optional[Dict[str, Any]]:
    """Re-simulates one (condition, strategy, k, user) session.

    Identical to personalised_strategies.py's own process_one_user
    (same item-based init, same _select_batch, same decaying-LR
    partial-SGD update, same shrinkage scoring) EXCEPT: under the
    'no_fabricate' condition, the partial-SGD update is skipped
    entirely (and the decaying-LR update counter is not advanced)
    whenever the revealed item has no recorded row for this user --
    the item is still added to ``shown`` (elicitation-budget k is
    unaffected), it just contributes no fabricated training signal.
    """
    import personalised_strategies as pers

    condition, strategy, k, u = work_item
    cache = pers._cache
    data             = cache['data']
    eligible_items   = cache['eligible_items']
    item_to_iidx     = cache['item_to_iidx']
    most_popular_iid = cache['most_popular_iid']
    i_0_inner        = cache['i_0_inner']
    svd_base         = cache['svd_base']
    mu_base          = cache['mu_base']
    n_factors        = cache['n_factors']
    GAMMA1, GAMMA2   = cache['GAMMA1'], cache['GAMMA2']
    LMBDA1, LMBDA2   = cache['LMBDA1'], cache['LMBDA2']
    pers.mu_base_global[0] = mu_base

    fabricate = (condition == 'fabricate')
    alpha = pers._shrink_alpha(k)
    # Deliberately identical seeding to process_one_user's own (no
    # 'condition' tag) -- both conditions must share the same RNG
    # stream (epsilon-greedy draws, negative sampling) for a clean,
    # controlled comparison where only the update-skipping logic
    # differs, and so that 'no_fabricate' reproduces
    # personalised_results.csv exactly.
    egreedy_rng_local = pers._seeded_rng(u, strategy, k, 'egreedy')
    neg_rng_local     = pers._seeded_rng(u, strategy, k, 'negsample')

    if i_0_inner is not None:
        pu_cold = svd_base.qi[i_0_inner].copy()
    else:
        pu_cold = np.zeros(n_factors)
    bu_cold = 0.0
    local_qi: Dict[int, np.ndarray] = {}
    local_bi: Dict[int, float] = {}
    shown = [most_popular_iid]
    n_updates = 0
    n_real_updates = 0

    first_row = data[(data['user_idx'] == u) & (data['itemId'] == most_popular_iid)]
    has_first = len(first_row) > 0
    r_first = float(first_row['interaction'].iloc[0]) if has_first else 0.0
    do_first_update = has_first or fabricate
    if i_0_inner is not None and do_first_update:
        pu_cold, bu_cold = pers._partial_lfm_update_cold(
            svd_base, pu_cold, bu_cold, i_0_inner, r_first, local_qi, local_bi,
            GAMMA1, GAMMA2, LMBDA1, LMBDA2, num_sgd_steps=pers.NUM_SGD_STEPS
        )
        n_updates += 1
        if has_first:
            n_real_updates += 1

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
            row = data[(data['user_idx'] == u) & (data['itemId'] == item)]
            has_row = len(row) > 0
            r_ui = float(row['interaction'].iloc[0]) if has_row else 0.0
            i_inner = item_to_iidx.get(item)
            do_update = has_row or fabricate
            if i_inner is not None and do_update:
                pu_cold, bu_cold = pers._partial_lfm_update_cold(
                    svd_base, pu_cold, bu_cold, i_inner, r_ui, local_qi, local_bi,
                    GAMMA1, GAMMA2, LMBDA1, LMBDA2, num_sgd_steps=pers.NUM_SGD_STEPS
                )
                n_updates += 1
                if has_row:
                    n_real_updates += 1
        round_number += 1

    _, test_items = pers._split_unseen_items(eligible_items, u, shown, val_frac=0.5)
    shown_set = set(shown)
    test_df = data[(data['user_idx'] == u) &
                   (data['itemId'].isin(test_items)) &
                   (~data['itemId'].isin(shown_set))]
    if len(test_df) == 0:
        return None

    preds_manual = []
    for row in test_df.itertuples():
        i_inner = item_to_iidx.get(row.itemId)
        if i_inner is None:
            continue
        qi = local_qi.get(i_inner, svd_base.qi[i_inner])
        bi = local_bi.get(i_inner, svd_base.bi[i_inner])
        est = pers._score(bu_cold, bi, pu_cold, qi, alpha)
        preds_manual.append(Pred(uid=row.user_idx, iid=row.item_idx,
                                  r_ui=row.interaction, est=est, details={}))
    if not preds_manual:
        return None
    rmse = accuracy.rmse(preds_manual, verbose=False)

    user_interacted = set(data[data['user_idx'] == u]['itemId'].tolist())
    candidate_negs = [iid for iid in eligible_items
                      if iid not in user_interacted and iid not in shown_set
                      and iid in item_to_iidx]
    pos_test_iids = test_df[test_df['interaction'] == 1]['itemId'].tolist()

    metric_keys = ['HR@5', 'HR@10', 'NDCG@5', 'NDCG@10', 'MRR@5', 'MRR@10', 'AUC']
    accum: Dict[str, List[float]] = {mk: [] for mk in metric_keys}
    for pos_iid in pos_test_iids:
        i_pos = item_to_iidx.get(pos_iid)
        if i_pos is None:
            continue
        qi_pos = local_qi.get(i_pos, svd_base.qi[i_pos])
        bi_pos = local_bi.get(i_pos, svd_base.bi[i_pos])
        pos_score = pers._score(bu_cold, bi_pos, pu_cold, qi_pos, alpha)
        n_sample = min(N_NEG, len(candidate_negs))
        if n_sample == 0:
            continue
        sampled_neg_iids = neg_rng_local.choice(candidate_negs, size=n_sample, replace=False)
        neg_scores = []
        for nid in sampled_neg_iids:
            ni = item_to_iidx[nid]
            qi_n = local_qi.get(ni, svd_base.qi[ni])
            bi_n = local_bi.get(ni, svd_base.bi[ni])
            neg_scores.append(pers._score(bu_cold, bi_n, pu_cold, qi_n, alpha))
        m = _extended_metrics_at_k(pos_score, neg_scores, k_list=[5, 10])
        for mk in metric_keys:
            accum[mk].append(m[mk])

    has_pos = bool(accum['HR@5'])
    return {
        'condition': condition, 'strategy': strategy, 'k': k, 'user': int(u),
        'rmse': rmse,
        'hr5': np.mean(accum['HR@5']) if has_pos else np.nan,
        'hr10': np.mean(accum['HR@10']) if has_pos else np.nan,
        'ndcg5': np.mean(accum['NDCG@5']) if has_pos else np.nan,
        'ndcg10': np.mean(accum['NDCG@10']) if has_pos else np.nan,
        'mrr5': np.mean(accum['MRR@5']) if has_pos else np.nan,
        'mrr10': np.mean(accum['MRR@10']) if has_pos else np.nan,
        'auc': np.mean(accum['AUC']) if has_pos else np.nan,
        'n_real_updates': n_real_updates,
        'n_shown': len(shown),
    }


def main() -> None:
    """Runs both conditions across all four strategies and k values at
    full 1,000-user scale, writes
    ``results/no_fabricated_negatives_test_results.csv``, and prints a
    per-(strategy, k) comparison plus a comparison against
    ``results/baseline_results.csv``.
    """
    t_start = time.time()
    if MODEL_DIR not in sys.path:
        sys.path.insert(0, MODEL_DIR)
    import personalised_strategies as pers

    with open(pers.MODEL_CACHE, 'rb') as f:
        cache = pickle.load(f)
    cold_users = cache['cold_users']
    eval_users = cold_users[:NUM_USERS]

    work_items: List[Tuple[str, str, int, int]] = [
        (cond, s, k, u)
        for cond in ['fabricate', 'no_fabricate']
        for s in STRATEGIES for k in K_VALUES for u in eval_users
    ]
    print(f"=== No-fabricated-negatives test: {len(work_items)} work items, "
          f"{N_WORKERS} workers ===", flush=True)

    mp.set_start_method('spawn', force=True)
    t0 = time.time()
    with mp.Pool(processes=N_WORKERS, initializer=_worker_init_local) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(process_one_user_variant, work_items, chunksize=4)):
            if r is not None:
                results.append(r)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(work_items)} done ({time.time()-t0:.1f}s elapsed)", flush=True)
    elapsed = time.time() - t0
    print(f"  Done: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved to {OUT_CSV}", flush=True)

    summary = df.groupby(['condition', 'strategy', 'k']).agg(
        rmse=('rmse', 'mean'), n_users=('user', 'count'),
        mean_n_real_updates=('n_real_updates', 'mean'), mean_n_shown=('n_shown', 'mean'),
    ).round(4)
    print("\n=== Summary (mean per condition/strategy/k) ===")
    print(summary.to_string())

    print(f"\n{'='*70}", flush=True)
    print("=== fabricate vs. no_fabricate RMSE, per strategy/k ===", flush=True)
    for strat in STRATEGIES:
        for k in K_VALUES:
            r_fab = df[(df.condition == 'fabricate') & (df.strategy == strat) & (df.k == k)]['rmse'].mean()
            r_nofab = df[(df.condition == 'no_fabricate') & (df.strategy == strat) & (df.k == k)]['rmse'].mean()
            print(f"  {strat} k={k:>3d}: fabricate RMSE={r_fab:.4f}  no_fabricate RMSE={r_nofab:.4f}  "
                  f"(change: {r_fab - r_nofab:+.4f}, positive = no_fabricate better)", flush=True)

    baseline_csv = os.path.join(RESULTS_DIR, 'baseline_results.csv')
    if os.path.exists(baseline_csv):
        base_df = pd.read_csv(baseline_csv)
        print(f"\n{'='*70}", flush=True)
        print("=== no_fabricate vs. best non-personalised baseline ===", flush=True)
        for strat in STRATEGIES:
            for k in K_VALUES:
                base_k = base_df[base_df['ItemsShown'] == k]
                if base_k.empty:
                    continue
                best_base_row = base_k.loc[base_k['RMSE'].idxmin()]
                r_nofab = df[(df.condition == 'no_fabricate') & (df.strategy == strat) & (df.k == k)]['rmse'].mean()
                margin = best_base_row['RMSE'] - r_nofab
                verdict = 'PERSONALISED WINS' if margin > 0 else 'baseline still wins'
                print(f"  {strat} k={k:>3d}: no_fabricate RMSE={r_nofab:.4f}  vs.  "
                      f"best baseline ({best_base_row['Strategy']}) RMSE={best_base_row['RMSE']:.4f}  "
                      f"-> margin={margin:+.4f}  [{verdict}]", flush=True)

    print(f"\nTOTAL TIME: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
