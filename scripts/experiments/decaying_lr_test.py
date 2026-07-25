"""
decaying_lr_test.py
=====================
Ablation: does a decaying learning rate for the incremental partial-SGD
update improve validation RMSE over a constant learning rate?

Motivation (Robbins & Monro, 1951): decreasing step sizes are a
classical requirement for SGD-type stochastic approximation to converge
rather than oscillate around the true value. Here, gamma_eff(n) =
gamma_0 / sqrt(1+n), where n counts the local SGD updates already
applied to a cold user's own parameters (pu_cold, bu_cold) in the
current session -- the first interaction sees n=0 (gamma_eff=gamma_0),
and the effective learning rate shrinks as more evidence accumulates.

This is a different mechanism from confidence shrinkage
(shrinkage_test.py): shrinkage re-weights an already-trained,
potentially over-confident personalised prediction at scoring time;
decaying LR aims to prevent the personalised estimate from becoming
over-confident during training in the first place. Since the decay
affects the training trajectory itself (not just final scoring), each
mode requires a full session re-run -- a single run cannot be re-scored
under both settings after the fact (contrast with shrinkage_test.py,
where only the final scoring step changes).

Reference strategy: SHLCP, evaluated on the 200-user hyperparameter-
tuning subset, across all four elicitation budgets k in {10,25,50,100}.

Usage
-----
    python scripts/experiments/decaying_lr_test.py

Input
-----
    results/base_model_cache.pkl

Output
------
    results/decaying_lr_test_results.csv
        Columns: k, mode ('fixed'/'decaying'), val_rmse, n_users, seconds

Complexity
----------
O(|K_VALUES| * |MODES| * 200 * k) partial-SGD updates (each O(F)) plus
O(|eligible_items|) scoring per active-learning round -- see thesis
Section 3.7.1 for the per-update cost argument.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from surprise import accuracy

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
MODEL_CACHE: str = os.path.join(RESULTS_DIR, 'base_model_cache.pkl')
OUT_DECAY: str = os.path.join(RESULTS_DIR, 'decaying_lr_test_results.csv')

K_VALUES: List[int] = [10, 25, 50, 100]
# (mode_name, decaying_flag) pairs to compare.
MODES: List[Tuple[str, bool]] = [('fixed', False), ('decaying', True)]


def _stable_seed(u: int, shown: List[Any]) -> int:
    """Deterministic replacement for Python's built-in ``hash()`` on
    strings -- see README.md "Reproducibility".

    Parameters
    ----------
    u : int
        User index.
    shown : list
        Raw itemId values shown to this user.

    Returns
    -------
    int
        A deterministic 32-bit seed derived from ``(u, shown)``.
    """
    key = f"{int(u)}|" + ','.join(sorted(str(x) for x in shown))
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def main() -> None:
    """Runs the SHLCP fixed-vs-decaying-LR comparison across all four
    elicitation budgets and writes ``results/decaying_lr_test_results.csv``.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.time()

    with open(MODEL_CACHE, 'rb') as f:
        cache = pickle.load(f)

    data             = cache['data']
    eligible_items   = cache['eligible_items']
    item_to_iidx     = cache['item_to_iidx']
    most_popular_iid = cache['most_popular_iid']
    i_0_inner        = cache['i_0_inner']
    svd_base         = cache['svd_base']
    mu_base          = cache['mu_base']
    n_factors        = cache['n_factors']
    cold_users       = cache['cold_users']
    GAMMA1, GAMMA2   = cache['GAMMA1'], cache['GAMMA2']
    LMBDA1, LMBDA2   = cache['LMBDA1'], cache['LMBDA2']

    print(f"[{time.time()-t_start:6.1f}s] Cache loaded (setup skipped).", flush=True)

    eval_cold_users = cold_users[:min(1000, len(cold_users))]
    hp_search_users = eval_cold_users[:min(200, len(eval_cold_users))]

    Pred = namedtuple('Prediction', ['uid', 'iid', 'r_ui', 'est', 'details'])

    def split_unseen_items(u: int, shown: List[Any],
                            val_frac: float = 0.5) -> Tuple[List[Any], List[Any]]:
        """Splits a cold user's remaining unseen items into a validation
        half and a test half, deterministically seeded per (u, shown).
        Only the validation half is used in this ablation."""
        shown_set = set(shown)
        unseen    = [iid for iid in eligible_items if iid not in shown_set]
        rng       = np.random.RandomState(_stable_seed(u, shown))
        rng.shuffle(unseen)
        n_val = int(val_frac * len(unseen))
        return unseen[:n_val], unseen[n_val:]

    def partial_lfm_update_cold(
        svd_model: Any, pu_cold: np.ndarray, bu_cold: float, i_inner: int, r_ui: float,
        local_qi: Dict[int, np.ndarray], local_bi: Dict[int, float],
        gamma1: float, gamma2: float, lmbda1: float, lmbda2: float
    ) -> Tuple[np.ndarray, float]:
        """One partial-SGD update given a newly-observed interaction
        (thesis Eq. 3.9-3.10). Identical update rule to
        personalised_strategies.py's ``_partial_lfm_update_cold`` with
        ``num_sgd_steps=1``, except ``gamma1``/``gamma2`` are already
        pre-scaled by the caller (``gammas_for``) before being passed in
        here, rather than scaled internally by an ``update_index``."""
        if i_inner not in local_qi:
            local_qi[i_inner] = svd_model.qi[i_inner].copy()
            local_bi[i_inner] = float(svd_model.bi[i_inner])
        mu  = svd_model.trainset.global_mean
        qi  = local_qi[i_inner]
        bi  = local_bi[i_inner]
        pred  = mu + bu_cold + bi + np.dot(pu_cold, qi)
        error = r_ui - pred
        bu_cold           = bu_cold + gamma1 * (error - lmbda1 * bu_cold)
        local_bi[i_inner] = bi      + gamma1 * (error - lmbda1 * bi)
        pu_new            = pu_cold + gamma2 * (error * qi      - lmbda2 * pu_cold)
        local_qi[i_inner] = qi      + gamma2 * (error * pu_cold - lmbda2 * qi)
        return pu_new, bu_cold

    def select_batch_items_cold(
        pu_cold: np.ndarray, bu_cold: float, shown: List[Any],
        local_qi: Dict[int, np.ndarray], local_bi: Dict[int, float],
        batch_size: int = 3
    ) -> List[Any]:
        """SHLCP (pure exploration): selects the ``batch_size``
        lowest-scoring not-yet-shown eligible items."""
        mu        = svd_base.trainset.global_mean
        shown_set = set(shown)
        scores    = {}
        for iid in eligible_items:
            if iid in shown_set:
                continue
            i_inner = item_to_iidx.get(iid)
            if i_inner is None:
                continue
            qi          = local_qi.get(i_inner, svd_base.qi[i_inner])
            bi          = local_bi.get(i_inner, svd_base.bi[i_inner])
            scores[iid] = mu + bu_cold + bi + np.dot(pu_cold, qi)
        if not scores:
            return []
        b = min(batch_size, len(scores))
        return sorted(scores, key=scores.get)[:b]

    def run_active_learning_session(
        u: int, k: int, decaying: bool, batch_size: int = 3
    ) -> Tuple[np.ndarray, float, Dict[int, np.ndarray], Dict[int, float], List[Any]]:
        """Runs one full SHLCP active-learning session for user ``u`` up
        to ``k`` revealed items, under either a constant (``decaying=
        False``) or decaying (``decaying=True``) learning-rate schedule.

        Returns
        -------
        pu_cold, bu_cold : the final cold-user parameters.
        local_qi, local_bi : the per-user local item vector/bias copies.
        shown : the list of raw itemId values revealed this session.
        """
        if i_0_inner is not None:
            pu_cold = svd_base.qi[i_0_inner].copy()
        else:
            pu_cold = np.zeros(n_factors)
        bu_cold  = 0.0
        local_qi, local_bi = {}, {}
        shown = [most_popular_iid]
        n = 0  # count of local SGD updates applied so far this session

        def gammas_for(n: int) -> Tuple[float, float]:
            """Returns (gamma1, gamma2) for update index n: unscaled if
            not decaying, else scaled by 1/sqrt(1+n) (thesis
            gamma_eff(n) schedule)."""
            if not decaying:
                return GAMMA1, GAMMA2
            decay = 1.0 / np.sqrt(1.0 + n)
            return GAMMA1 * decay, GAMMA2 * decay

        first_row = data[(data['user_idx'] == u) & (data['itemId'] == most_popular_iid)]
        r_first   = float(first_row['interaction'].iloc[0]) if len(first_row) > 0 else 0.0

        if i_0_inner is not None:
            g1, g2 = gammas_for(n)
            pu_cold, bu_cold = partial_lfm_update_cold(
                svd_base, pu_cold, bu_cold, i_0_inner, r_first, local_qi, local_bi,
                gamma1=g1, gamma2=g2, lmbda1=LMBDA1, lmbda2=LMBDA2
            )
            n += 1

        while len(shown) < k:
            b     = min(batch_size, k - len(shown))
            batch = select_batch_items_cold(pu_cold, bu_cold, shown, local_qi, local_bi, b)
            if not batch:
                break
            shown.extend(batch)
            for next_item in batch:
                row   = data[(data['user_idx'] == u) & (data['itemId'] == next_item)]
                r_ui  = float(row['interaction'].iloc[0]) if len(row) > 0 else 0.0
                i_inner = item_to_iidx.get(next_item)
                if i_inner is not None:
                    g1, g2 = gammas_for(n)
                    pu_cold, bu_cold = partial_lfm_update_cold(
                        svd_base, pu_cold, bu_cold, i_inner, r_ui, local_qi, local_bi,
                        gamma1=g1, gamma2=g2, lmbda1=LMBDA1, lmbda2=LMBDA2
                    )
                    n += 1

        return pu_cold, bu_cold, local_qi, local_bi, shown

    def evaluate_session_rmse(
        pu_cold: np.ndarray, bu_cold: float, local_qi: Dict[int, np.ndarray],
        local_bi: Dict[int, float], shown: List[Any], test_items: List[Any], u: int
    ) -> Optional[float]:
        """Scores RMSE on ``test_items`` using the (possibly locally-
        updated) item vectors/biases and the final cold-user parameters.
        Returns None if no test items have a valid interaction record
        for this user."""
        shown_set = set(shown)
        test_df = data[(data['user_idx'] == u) &
                       (data['itemId'].isin(test_items)) &
                       (~data['itemId'].isin(shown_set))]
        if len(test_df) == 0:
            return None
        preds = []
        for row in test_df.itertuples():
            i_inner = item_to_iidx.get(row.itemId)
            if i_inner is None:
                continue
            qi  = local_qi.get(i_inner, svd_base.qi[i_inner])
            bi  = local_bi.get(i_inner, svd_base.bi[i_inner])
            est = float(np.clip(mu_base + bu_cold + bi + np.dot(pu_cold, qi), 0, 1))
            preds.append(Pred(uid=row.user_idx, iid=row.item_idx,
                               r_ui=row.interaction, est=est, details={}))
        if not preds:
            return None
        return accuracy.rmse(preds, verbose=False)

    results = {}
    log = []

    print(f"\n=== Decaying LR test: SHLCP, {len(hp_search_users)} users, "
          f"gamma_eff(n)=gamma_0/sqrt(1+n) ===", flush=True)

    for k in K_VALUES:
        for mode_name, decaying in MODES:
            t_cell = time.time()
            rmses = []
            for u in hp_search_users:
                pu_cold, bu_cold, local_qi, local_bi, shown = run_active_learning_session(u, k, decaying)
                val_items, _ = split_unseen_items(u, shown, val_frac=0.5)
                r = evaluate_session_rmse(pu_cold, bu_cold, local_qi, local_bi, shown, val_items, u)
                if r is not None:
                    rmses.append(r)
            avg = np.mean(rmses) if rmses else float('nan')
            results[(k, mode_name)] = avg
            log.append((k, mode_name, avg, len(rmses), time.time() - t_cell))
            print(f"  [k={k}][{mode_name}] val RMSE={avg:.4f} "
                  f"(n={len(rmses)}, {time.time()-t_cell:.1f}s)", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("=== Fixed vs. decaying per k ===", flush=True)
    for k in K_VALUES:
        fixed_rmse = results[(k, 'fixed')]
        decay_rmse = results[(k, 'decaying')]
        print(f"  k={k}: fixed RMSE={fixed_rmse:.4f}  decaying RMSE={decay_rmse:.4f}  "
              f"(improvement: {fixed_rmse - decay_rmse:+.4f})", flush=True)

    df = pd.DataFrame(log, columns=['k', 'mode', 'val_rmse', 'n_users', 'seconds'])
    df.to_csv(OUT_DECAY, index=False)
    print(f"\nSaved to {OUT_DECAY}", flush=True)
    print(f"TOTAL TIME: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
