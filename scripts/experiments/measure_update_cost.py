"""
measure_update_cost.py
========================
Empirical (not just asymptotic) measurement of the partial-update
efficiency claim underlying the whole personalised-strategy pipeline
(thesis Section 3.7.1, "Motivation and Efficiency Gain"): one full SVD
retrain on the warm-user trainset (what a naive personalised-strategy
implementation would do after every single revealed cold-user
interaction) vs. one partial-SGD update restricted to the four
parameters actually touched by that interaction
(`personalised_strategies.py`'s `_partial_lfm_update_cold`).

Uses the already-cached warm data and tuned hyperparameters
(`results/base_model_cache.pkl`), so this does not repeat the
~20-25 minute GridSearchCV -- only the single-fit retrain cost is
measured fresh.

Usage
-----
    python scripts/experiments/measure_update_cost.py

Input
-----
    results/base_model_cache.pkl

Output
------
    results/update_cost_results.txt
        Key=value lines: mean_retrain_seconds, mean_partial_update_seconds,
        speedup_x, n_retrain_trials, n_update_trials, warm_trainset_n_ratings
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Any, Dict, Tuple

import numpy as np
from surprise import SVD, Dataset, Reader

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
MODEL_CACHE: str = os.path.join(RESULTS_DIR, 'base_model_cache.pkl')
OUT_TXT: str = os.path.join(RESULTS_DIR, 'update_cost_results.txt')

N_RETRAIN_TRIALS: int = 3
N_UPDATE_TRIALS: int = 20000


def main() -> None:
    """Measures and reports the full-retrain-vs-partial-update speedup,
    writing ``results/update_cost_results.txt``.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(MODEL_CACHE, 'rb') as f:
        cache: Dict[str, Any] = pickle.load(f)

    data = cache['data']
    cold_users = cache['cold_users']
    best_params = cache['best_params']
    svd_base = cache['svd_base']
    GAMMA1, GAMMA2 = cache['GAMMA1'], cache['GAMMA2']
    LMBDA1, LMBDA2 = cache['LMBDA1'], cache['LMBDA2']

    warm_data = data[~data['user_idx'].isin(cold_users)]
    reader = Reader(rating_scale=(0, 1))
    warm_dataset = Dataset.load_from_df(
        warm_data[['user_idx', 'item_idx', 'interaction']], reader
    )
    warm_trainset = warm_dataset.build_full_trainset()

    print(f"Warm trainset: {warm_trainset.n_ratings:,} observations, "
          f"{warm_trainset.n_users:,} users, {warm_trainset.n_items:,} items", flush=True)

    # ── Cost 1: one full SVD retrain from scratch ──
    # (what a naive implementation would do after EVERY single revealed
    # cold-user interaction, across the full 140,873-user cold
    # population -- the cost this thesis's partial update avoids.)
    retrain_times = []
    for trial in range(N_RETRAIN_TRIALS):
        svd = SVD(n_factors=best_params['n_factors'], reg_all=best_params['reg_all'],
                  n_epochs=best_params['n_epochs'], biased=True, random_state=trial)
        t0 = time.time()
        svd.fit(warm_trainset)
        retrain_times.append(time.time() - t0)
        print(f"  [full retrain trial {trial+1}/{N_RETRAIN_TRIALS}] {retrain_times[-1]:.2f}s",
              flush=True)

    mean_retrain = float(np.mean(retrain_times))

    # ── Cost 2: one partial-SGD update ──
    # Restricted to (p_u^c, b_u^c, q_i, b_i) -- exactly the four
    # quantities thesis Section 3.7.1 argues are the only ones that need
    # to change per interaction. Timed over many repetitions since a
    # single call is too fast to measure reliably against timer
    # resolution/Python call overhead.
    n_factors = svd_base.pu.shape[1]
    i_inner = 0
    qi = svd_base.qi[i_inner].copy()
    bi = float(svd_base.bi[i_inner])
    pu_cold = np.random.RandomState(0).normal(size=n_factors)
    bu_cold = 0.0
    mu = svd_base.trainset.global_mean

    def partial_update(
        pu_cold: np.ndarray, bu_cold: float, qi: np.ndarray, bi: float, r_ui: float
    ) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """One partial-SGD update step (thesis Eq. 3.9-3.10), identical
        arithmetic to ``personalised_strategies.py``'s
        ``_partial_lfm_update_cold`` with ``num_sgd_steps=1`` and no
        learning-rate decay."""
        pred = mu + bu_cold + bi + np.dot(pu_cold, qi)
        error = r_ui - pred
        bu_cold_new = bu_cold + GAMMA1 * (error - LMBDA1 * bu_cold)
        bi_new = bi + GAMMA1 * (error - LMBDA1 * bi)
        pu_new = pu_cold + GAMMA2 * (error * qi - LMBDA2 * pu_cold)
        qi_new = qi + GAMMA2 * (error * pu_cold - LMBDA2 * qi)
        return pu_new, bu_cold_new, qi_new, bi_new

    t0 = time.time()
    for _ in range(N_UPDATE_TRIALS):
        pu_cold, bu_cold, qi, bi = partial_update(pu_cold, bu_cold, qi, bi, 1.0)
    total_update_time = time.time() - t0
    mean_update = total_update_time / N_UPDATE_TRIALS

    speedup = mean_retrain / mean_update

    print(f"\n{'='*60}")
    print(f"Mean full retrain time  : {mean_retrain:.3f}s  (n={N_RETRAIN_TRIALS} trials)")
    print(f"Mean partial update time: {mean_update*1e6:.2f} microseconds  (n={N_UPDATE_TRIALS} calls)")
    print(f"Speedup (retrain / partial update): {speedup:,.0f}x")
    print(f"{'='*60}")

    with open(OUT_TXT, 'w') as f:
        f.write(f"mean_retrain_seconds={mean_retrain:.4f}\n")
        f.write(f"mean_partial_update_seconds={mean_update:.8f}\n")
        f.write(f"speedup_x={speedup:.1f}\n")
        f.write(f"n_retrain_trials={N_RETRAIN_TRIALS}\n")
        f.write(f"n_update_trials={N_UPDATE_TRIALS}\n")
        f.write(f"warm_trainset_n_ratings={warm_trainset.n_ratings}\n")
    print(f"Saved {OUT_TXT}")


if __name__ == '__main__':
    main()
