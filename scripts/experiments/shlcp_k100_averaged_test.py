"""
shlcp_k100_averaged_test.py
=============================
Checks whether SHLCP's k=100 RMSE (0.1616, the single lowest RMSE of
any (strategy, k) cell in the headline results) is a real result or
single-draw val/test-split noise -- the same question already checked
for the Popularity/PopError baselines.

The active-learning loop (item selection, partial-SGD updates) does
not depend on the val/test split at all -- that split only happens
afterward, over the REMAINING unseen items. So each user's session is
run once (reusing personalised_strategies.py's own production
building blocks), then rescored against N_DRAWS independent val/test
reshuffles, exactly the "run once, rescore many times" pattern already
used for the shrinkage/cold-start-init ablations and for
baseline_averaged_test.py.

Usage
-----
    python scripts/experiments/shlcp_k100_averaged_test.py

Input
-----
    results/base_model_cache.pkl

Output
------
    results/shlcp_k100_averaged_results.csv
        Columns: draw, val_rmse, n_users (per-draw RMSE, for
        inspecting draw-to-draw spread directly)
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import namedtuple
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from surprise import accuracy

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR: str = os.path.join(SCRIPT_DIR, '..', 'model')
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
OUT_CSV: str = os.path.join(RESULTS_DIR, 'shlcp_k100_averaged_results.csv')

STRATEGY: str = 'SHLCP'
K: int = 100
NUM_USERS: int = 1000
N_DRAWS: int = 30
N_WORKERS: int = 10

Pred = namedtuple('Prediction', ['uid', 'iid', 'r_ui', 'est', 'details'])


def _stable_seed(u: int, shown: List[Any], draw: int) -> int:
    key = f"{int(u)}|{draw}|" + ','.join(sorted(str(x) for x in shown))
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def _worker_init_local(eval_users) -> None:
    if MODEL_DIR not in sys.path:
        sys.path.insert(0, MODEL_DIR)
    import personalised_strategies as pers
    pers._worker_init(eval_users)


def run_session_and_score(u: int) -> Dict[str, Any]:
    """Runs one SHLCP k=100 session (identical to process_one_user's
    own logic), then scores RMSE against N_DRAWS independent val/test
    reshuffles of the remaining unseen items instead of the single
    deterministic split process_one_user itself would use.
    """
    import personalised_strategies as pers

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

    alpha = pers._shrink_alpha(K)
    egreedy_rng_local = pers._seeded_rng(u, STRATEGY, K, 'egreedy')

    if K in pers.ZERO_INIT_K_VALUES:
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
    while len(shown) < K:
        b = min(pers.BATCH_SIZE, K - len(shown))
        batch = pers._select_batch(svd_base, pu_cold, bu_cold, shown, eligible_items,
                                    item_to_iidx, local_qi, local_bi, STRATEGY, b,
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

    draw_rmses = {}
    for draw in range(N_DRAWS):
        rng = np.random.RandomState(_stable_seed(u, shown, draw))
        unseen_shuffled = unseen.copy()
        rng.shuffle(unseen_shuffled)
        n_val = int(0.5 * len(unseen_shuffled))
        test_items = unseen_shuffled[n_val:]
        test_set = set(test_items)
        test_rows = [(iid, r) for iid, r in user_dict.items() if iid in test_set]
        if not test_rows:
            continue
        preds = []
        for iid, r in test_rows:
            i_inner = item_to_iidx.get(iid)
            if i_inner is None:
                continue
            qi = local_qi.get(i_inner, svd_base.qi[i_inner])
            bi = local_bi.get(i_inner, svd_base.bi[i_inner])
            est = pers._score(bu_cold, bi, pu_cold, qi, alpha)
            preds.append(Pred(uid=u, iid=iid, r_ui=r, est=est, details={}))
        if preds:
            draw_rmses[draw] = accuracy.rmse(preds, verbose=False)

    return {'user': int(u), 'draw_rmses': draw_rmses}


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_cache = os.path.join(RESULTS_DIR, 'base_model_cache.pkl')
    with open(model_cache, 'rb') as f:
        cache = pickle.load(f)
    cold_users = cache['cold_users']
    eval_users = cold_users[:NUM_USERS]

    print(f"=== SHLCP k=100 val/test-split averaging check: {len(eval_users)} users, "
          f"{N_DRAWS} draws each ===", flush=True)
    t0 = time.time()
    with mp.Pool(processes=N_WORKERS, initializer=_worker_init_local,
                  initargs=(eval_users,)) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_session_and_score, eval_users, chunksize=4)):
            results.append(r)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(eval_users)} users done ({time.time()-t0:.1f}s)", flush=True)
    elapsed = time.time() - t0
    print(f"Done: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    # Per-draw mean RMSE across all users (draw-to-draw spread).
    per_draw_means = []
    for draw in range(N_DRAWS):
        vals = [r['draw_rmses'][draw] for r in results if draw in r['draw_rmses']]
        if vals:
            per_draw_means.append({'draw': draw, 'val_rmse': np.mean(vals), 'n_users': len(vals)})
    df_per_draw = pd.DataFrame(per_draw_means)
    df_per_draw.to_csv(OUT_CSV, index=False)
    print(f"\nSaved per-draw means to {OUT_CSV}", flush=True)
    print(df_per_draw.to_string(index=False), flush=True)

    # Overall: average each user's per-draw RMSEs first, then average across users
    # (matches baseline_averaged_test.py's own aggregation order).
    per_user_avg = []
    for r in results:
        if r['draw_rmses']:
            per_user_avg.append(np.mean(list(r['draw_rmses'].values())))
    overall = np.mean(per_user_avg) if per_user_avg else float('nan')
    print(f"\n=== SHLCP k=100: {N_DRAWS}-draw averaged RMSE = {overall:.4f} "
          f"(n_users={len(per_user_avg)}) ===", flush=True)
    print(f"    Single-draw RMSE: 0.1616", flush=True)
    print(f"    Draw-to-draw std across the {N_DRAWS} per-draw means: "
          f"{df_per_draw['val_rmse'].std():.4f}", flush=True)


if __name__ == '__main__':
    main()
