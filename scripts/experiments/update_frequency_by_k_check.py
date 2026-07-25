"""
update_frequency_by_k_check.py
================================
Checks whether the near-zero-real-update property holds equally at
every elicitation budget k, or varies with k -- e.g. "triggering" at
some k values and not others.

Re-simulates every (strategy, k, user) session using the CURRENT
production active-learning loop (personalised_strategies.py's own
_select_batch / _partial_lfm_update_cold / ZERO_INIT_K_VALUES /
LMBDA1 / LMBDA2 / NUM_SGD_STEPS, imported live so this always reflects
whatever that module currently implements), but skips RMSE/HR/NDCG
computation entirely -- only the number of real partial-SGD updates
each user's session actually received is recorded. This makes it much
faster than the full evaluation and keeps the two concerns (item
selection + update counting vs. final scoring) cleanly separated.

A "real" update is one triggered by an actual recorded (user, item)
interaction, exactly matching process_one_user's own has_first/has_row
gating; this script does not fabricate negatives.

Usage
-----
    python scripts/experiments/update_frequency_by_k_check.py

Input
-----
    results/base_model_cache.pkl

Output
------
    results/update_frequency_by_k_results.csv
        Columns: strategy, k, user, n_real_updates, n_shown
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR: str = os.path.join(SCRIPT_DIR, '..', 'model')
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
OUT_CSV: str = os.path.join(RESULTS_DIR, 'update_frequency_by_k_results.csv')

STRATEGIES: List[str] = ['SHHCP', 'SHLCP', 'SHMCP', 'SHECP']
K_VALUES: List[int] = [10, 25, 50, 100]
NUM_USERS: int = 1000
N_WORKERS: int = 10


def _worker_init_local(eval_users) -> None:
    import sys
    if MODEL_DIR not in sys.path:
        sys.path.insert(0, MODEL_DIR)
    import personalised_strategies as pers
    pers._worker_init(eval_users)


def count_updates_one_user(work_item: Tuple[str, int, int]) -> Dict[str, Any]:
    """Runs the current production item-selection + partial-update
    loop for one (strategy, k, user) and returns only the update
    count -- no RMSE/HR/NDCG scoring, since that is not needed to
    answer this question and skipping it makes the run much faster.
    """
    import personalised_strategies as pers

    strategy, k, u = work_item
    cache = pers._cache
    user_dict        = pers._user_item_interaction.get(int(u), {})
    eligible_items    = cache['eligible_items']
    item_to_iidx      = cache['item_to_iidx']
    most_popular_iid  = cache['most_popular_iid']
    i_0_inner         = cache['i_0_inner']
    svd_base          = cache['svd_base']
    n_factors         = cache['n_factors']
    GAMMA1, GAMMA2    = cache['GAMMA1'], cache['GAMMA2']

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
    n_real_updates = 0

    has_first = most_popular_iid in user_dict
    r_first = float(user_dict[most_popular_iid]) if has_first else 0.0
    if i_0_inner is not None and has_first:
        pu_cold, bu_cold = pers._partial_lfm_update_cold(
            svd_base, pu_cold, bu_cold, i_0_inner, r_first, local_qi, local_bi,
            GAMMA1, GAMMA2, pers.LMBDA1, pers.LMBDA2,
            num_sgd_steps=pers.NUM_SGD_STEPS
        )
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
            has_row = item in user_dict
            r_ui = float(user_dict[item]) if has_row else 0.0
            i_inner = item_to_iidx.get(item)
            if i_inner is not None and has_row:
                pu_cold, bu_cold = pers._partial_lfm_update_cold(
                    svd_base, pu_cold, bu_cold, i_inner, r_ui, local_qi, local_bi,
                    GAMMA1, GAMMA2, pers.LMBDA1, pers.LMBDA2,
                    num_sgd_steps=pers.NUM_SGD_STEPS
                )
                n_real_updates += 1
        round_number += 1

    return {'strategy': strategy, 'k': k, 'user': int(u),
            'n_real_updates': n_real_updates, 'n_shown': len(shown)}


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_cache = os.path.join(RESULTS_DIR, 'base_model_cache.pkl')
    with open(model_cache, 'rb') as f:
        cache = pickle.load(f)
    cold_users = cache['cold_users']
    eval_users = cold_users[:NUM_USERS]

    work_items = [(s, k, u) for s in STRATEGIES for k in K_VALUES for u in eval_users]
    print(f"=== Update-frequency-by-k check: {len(work_items)} work items, "
          f"{N_WORKERS} workers ===", flush=True)

    t0 = time.time()
    with mp.Pool(processes=N_WORKERS, initializer=_worker_init_local,
                  initargs=(eval_users,)) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(count_updates_one_user, work_items, chunksize=8)):
            results.append(r)
            if (i + 1) % 4000 == 0:
                print(f"  {i+1}/{len(work_items)} done ({time.time()-t0:.1f}s)", flush=True)
    elapsed = time.time() - t0
    print(f"Done: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved to {OUT_CSV}", flush=True)

    print("\n=== Fraction of users with >=1 real update, by (strategy, k) ===", flush=True)
    summary = (df.assign(has_update=(df['n_real_updates'] >= 1))
                 .groupby(['strategy', 'k'])
                 .agg(frac_with_update=('has_update', 'mean'),
                      mean_n_updates=('n_real_updates', 'mean'),
                      n_users=('user', 'count'))
                 .reset_index())
    print(summary.to_string(index=False), flush=True)

    print("\n=== Pooled across all 4 strategies, by k ===", flush=True)
    pooled = (df.assign(has_update=(df['n_real_updates'] >= 1))
                .groupby('k')
                .agg(frac_with_update=('has_update', 'mean'),
                     mean_n_updates=('n_real_updates', 'mean'),
                     n_sessions=('user', 'count'))
                .reset_index())
    print(pooled.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
