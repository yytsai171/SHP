"""
cold_start_init_averaged_test.py
===================================
Re-checks item-init vs. zero-init at k=100, averaged over N_DRAWS
independent validation-split reshuffles per user, instead of a single
split.

Why this is needed
-------------------
With almost no cold user receiving any real training update during
elicitation (confirmed separately: 0% of users get a second real
update, even at k=100), both zero-init and item-init reduce to a
near-fixed prediction formula for the vast majority of users. Whether
one beats the other on a single evaluation is then largely determined
by which of a user's few real interactions happen to land in that
run's validation half -- the same single-draw sampling variance
already found and fixed for the Random baseline. This script fixes it
the same way: run each (user, strategy, init) active-learning session
once (the item-selection sequence is deterministic, or seeded, and
does not need to be redrawn), then average RMSE/HR/NDCG over N_DRAWS
independent reshuffles of the remaining unseen items into
validation/test halves.

This design choice (zero- vs. item-init at k=100) is scored on the
validation half only, in every one of the 30 reshuffles -- never on
the test half -- so that the canonical test split used for the final
reported numbers plays no part in selecting the initialisation.

Usage
-----
    python scripts/experiments/cold_start_init_averaged_test.py

Input
-----
    results/base_model_cache.pkl

Output
------
    results/cold_start_init_averaged_results.csv
        Columns: strategy, init, mean_rmse, mean_hr10, mean_ndcg10, n_users
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
from surprise import accuracy

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
MODEL_CACHE: str = os.path.join(RESULTS_DIR, 'base_model_cache.pkl')
OUT_CSV: str = os.path.join(RESULTS_DIR, 'cold_start_init_averaged_results.csv')

K_PROBE: int = 100
BATCH_SIZE: int = 3
N_NEG: int = 99
N_DRAWS: int = 30
STRATEGIES: List[str] = ['SHHCP', 'SHLCP', 'SHMCP', 'SHECP']
INITS: List[str] = ['zero', 'item']
SHECP_FLOOR: float = 0.05
SHECP_DECAY: float = 0.95
LMBDA1: float = 1e-7
LMBDA2: float = 1e-6


def _stable_seed(*parts: Any) -> int:
    key = '|'.join(str(p) for p in parts)
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def main() -> None:
    t0 = time.time()
    with open(MODEL_CACHE, 'rb') as f:
        cache = pickle.load(f)

    data = cache['data']
    eligible_items = cache['eligible_items']
    item_to_iidx = cache['item_to_iidx']
    most_popular_iid = cache['most_popular_iid']
    i_0_inner = cache['i_0_inner']
    svd_base = cache['svd_base']
    mu_base = cache['mu_base']
    n_factors = cache['n_factors']
    cold_users = cache['cold_users']
    GAMMA1, GAMMA2 = cache['GAMMA1'], cache['GAMMA2']

    eval_cold_users = cold_users[:1000]
    eval_set = set(int(u) for u in eval_cold_users)
    user_data = data[data['user_idx'].isin(eval_set)]
    user_dict_all: Dict[int, Dict[Any, float]] = {
        int(u): dict(zip(g['itemId'].values, g['interaction'].values))
        for u, g in user_data.groupby('user_idx')
    }
    print(f"[{time.time()-t0:6.1f}s] Lookup built for {len(user_dict_all)} users.", flush=True)

    eligible_inner = np.array([item_to_iidx[iid] for iid in eligible_items])
    eligible_arr = np.array(eligible_items)
    inner_to_pos = {inner: pos for pos, inner in enumerate(eligible_inner)}
    base_qi_elig = svd_base.qi[eligible_inner].copy()
    base_bi_elig = svd_base.bi[eligible_inner].copy()

    Pred = namedtuple('Prediction', ['uid', 'iid', 'r_ui', 'est', 'details'])

    def partial_update(pu_cold, bu_cold, i_inner, r_ui, local_qi, local_bi):
        if i_inner not in local_qi:
            local_qi[i_inner] = svd_base.qi[i_inner].copy()
            local_bi[i_inner] = float(svd_base.bi[i_inner])
        mu = svd_base.trainset.global_mean
        qi, bi = local_qi[i_inner], local_bi[i_inner]
        pred = mu + bu_cold + bi + np.dot(pu_cold, qi)
        error = r_ui - pred
        bu_cold = bu_cold + GAMMA1 * (error - LMBDA1 * bu_cold)
        local_bi[i_inner] = bi + GAMMA1 * (error - LMBDA1 * bi)
        pu_new = pu_cold + GAMMA2 * (error * qi - LMBDA2 * pu_cold)
        local_qi[i_inner] = qi + GAMMA2 * (error * pu_cold - LMBDA2 * qi)
        return pu_new, bu_cold

    def select_batch(pu_cold, bu_cold, shown_mask, local_overrides, batch_size, strategy,
                      round_number, egreedy_rng):
        mu = svd_base.trainset.global_mean
        scores = mu + bu_cold + base_bi_elig + base_qi_elig @ pu_cold
        for pos, (qi, bi) in local_overrides.items():
            scores[pos] = mu + bu_cold + bi + np.dot(pu_cold, qi)
        valid = ~shown_mask
        valid_scores = scores[valid]
        valid_iids = eligible_arr[valid]
        if len(valid_scores) == 0:
            return []
        b = min(batch_size, len(valid_scores))
        if strategy == 'SHHCP':
            idx = np.argpartition(-valid_scores, b - 1)[:b]
        elif strategy == 'SHLCP':
            idx = np.argpartition(valid_scores, b - 1)[:b]
        elif strategy == 'SHMCP':
            median_val = np.median(valid_scores)
            dist = np.abs(valid_scores - median_val)
            idx = np.argpartition(dist, b - 1)[:b]
        elif strategy == 'SHECP':
            epsilon = max(SHECP_FLOOR, SHECP_DECAY ** round_number)
            if egreedy_rng.random() < epsilon:
                idx = np.argpartition(valid_scores, b - 1)[:b]
            else:
                idx = np.argpartition(-valid_scores, b - 1)[:b]
        else:
            raise ValueError(strategy)
        return valid_iids[idx].tolist()

    def run_session(u: int, strategy: str, init: str, user_dict: Dict[Any, float]):
        pu_cold = np.zeros(n_factors) if init == 'zero' else (
            svd_base.qi[i_0_inner].copy() if i_0_inner is not None else np.zeros(n_factors))
        bu_cold = 0.0
        local_overrides: Dict[int, Any] = {}
        local_qi_raw: Dict[int, np.ndarray] = {}
        local_bi_raw: Dict[int, float] = {}
        shown_mask = np.zeros(len(eligible_items), dtype=bool)
        shown_set = {most_popular_iid}
        if i_0_inner is not None and i_0_inner in inner_to_pos:
            shown_mask[inner_to_pos[i_0_inner]] = True

        has_first = most_popular_iid in user_dict
        if i_0_inner is not None and has_first:
            r_first = float(user_dict[most_popular_iid])
            pu_cold, bu_cold = partial_update(pu_cold, bu_cold, i_0_inner, r_first,
                                               local_qi_raw, local_bi_raw)
            if i_0_inner in inner_to_pos:
                local_overrides[inner_to_pos[i_0_inner]] = (local_qi_raw[i_0_inner], local_bi_raw[i_0_inner])

        egreedy_rng = np.random.RandomState(_stable_seed(u, strategy, init, 'egreedy'))
        round_number = 0
        n_shown = 1
        while n_shown < K_PROBE:
            b = min(BATCH_SIZE, K_PROBE - n_shown)
            batch = select_batch(pu_cold, bu_cold, shown_mask, local_overrides, b, strategy,
                                  round_number, egreedy_rng)
            if not batch:
                break
            for item in batch:
                i_inner = item_to_iidx.get(item)
                if i_inner is not None and i_inner in inner_to_pos:
                    shown_mask[inner_to_pos[i_inner]] = True
                shown_set.add(item)
                n_shown += 1
                has_row = item in user_dict
                if has_row and i_inner is not None:
                    r_ui = float(user_dict[item])
                    pu_cold, bu_cold = partial_update(pu_cold, bu_cold, i_inner, r_ui,
                                                       local_qi_raw, local_bi_raw)
                    if i_inner in inner_to_pos:
                        local_overrides[inner_to_pos[i_inner]] = (local_qi_raw[i_inner], local_bi_raw[i_inner])
            round_number += 1

        return pu_cold, bu_cold, local_qi_raw, local_bi_raw, shown_set

    def score(bu_cold, bi, pu_cold, qi):
        return float(np.clip(mu_base + bi + bu_cold + np.dot(pu_cold, qi), 0, 1))

    neg_rng = np.random.RandomState(42)

    def evaluate_draws(u, user_dict, pu_cold, bu_cold, local_qi_raw, local_bi_raw, shown_set):
        unseen = [iid for iid in eligible_items if iid not in shown_set]
        rmses, hr10s, ndcg10s = [], [], []
        for draw in range(N_DRAWS):
            rng = np.random.RandomState(_stable_seed(u, len(shown_set), draw))
            order = unseen.copy()
            rng.shuffle(order)
            n_val = int(0.5 * len(order))
            val_items = order[:n_val]
            val_set = set(val_items)
            val_rows = [(iid, r) for iid, r in user_dict.items() if iid in val_set]
            if not val_rows:
                continue
            preds = []
            pos_iids = []
            for iid, r in val_rows:
                i_inner = item_to_iidx.get(iid)
                if i_inner is None:
                    continue
                qi = local_qi_raw.get(i_inner, svd_base.qi[i_inner])
                bi = local_bi_raw.get(i_inner, svd_base.bi[i_inner])
                est = score(bu_cold, bi, pu_cold, qi)
                preds.append(Pred(uid=u, iid=iid, r_ui=r, est=est, details={}))
                if r == 1:
                    pos_iids.append(iid)
            if not preds:
                continue
            rmses.append(accuracy.rmse(preds, verbose=False))

            if pos_iids:
                user_interacted = set(user_dict.keys())
                candidate_negs = [iid for iid in eligible_items
                                   if iid not in user_interacted and iid not in shown_set
                                   and iid in item_to_iidx]
                hr_vals, ndcg_vals = [], []
                for pos_iid in pos_iids:
                    i_pos = item_to_iidx.get(pos_iid)
                    if i_pos is None:
                        continue
                    qi_pos = local_qi_raw.get(i_pos, svd_base.qi[i_pos])
                    bi_pos = local_bi_raw.get(i_pos, svd_base.bi[i_pos])
                    pos_score = score(bu_cold, bi_pos, pu_cold, qi_pos)
                    n_sample = min(N_NEG, len(candidate_negs))
                    if n_sample == 0:
                        continue
                    sampled = neg_rng.choice(candidate_negs, size=n_sample, replace=False)
                    neg_scores = [score(bu_cold, local_bi_raw.get(item_to_iidx[nid], svd_base.bi[item_to_iidx[nid]]),
                                         pu_cold, local_qi_raw.get(item_to_iidx[nid], svd_base.qi[item_to_iidx[nid]]))
                                  for nid in sampled]
                    rank = 1 + sum(1 for s in neg_scores if s > pos_score)
                    hr_vals.append(1.0 if rank <= 10 else 0.0)
                    ndcg_vals.append((1.0 / np.log2(rank + 1)) if rank <= 10 else 0.0)
                if hr_vals:
                    hr10s.append(float(np.mean(hr_vals)))
                    ndcg10s.append(float(np.mean(ndcg_vals)))
        if not rmses:
            return None
        return {
            'rmse': float(np.mean(rmses)),
            'hr10': float(np.mean(hr10s)) if hr10s else float('nan'),
            'ndcg10': float(np.mean(ndcg10s)) if ndcg10s else float('nan'),
        }

    rows = []
    for strategy in STRATEGIES:
        for init in INITS:
            ts = time.time()
            per_user = []
            for u in eval_cold_users:
                u_int = int(u)
                user_dict = user_dict_all.get(u_int, {})
                pu_cold, bu_cold, local_qi_raw, local_bi_raw, shown_set = run_session(
                    u_int, strategy, init, user_dict)
                res = evaluate_draws(u_int, user_dict, pu_cold, bu_cold, local_qi_raw, local_bi_raw, shown_set)
                if res is not None:
                    per_user.append(res)
            mean_rmse = float(np.mean([r['rmse'] for r in per_user])) if per_user else float('nan')
            mean_hr10 = float(np.nanmean([r['hr10'] for r in per_user])) if per_user else float('nan')
            mean_ndcg10 = float(np.nanmean([r['ndcg10'] for r in per_user])) if per_user else float('nan')
            rows.append({'strategy': strategy, 'init': init, 'mean_rmse': mean_rmse,
                         'mean_hr10': mean_hr10, 'mean_ndcg10': mean_ndcg10, 'n_users': len(per_user)})
            print(f"[{time.time()-t0:6.1f}s] [{strategy}][{init}] mean RMSE={mean_rmse:.4f} "
                  f"HR@10={mean_hr10:.4f} (n={len(per_user)}, {time.time()-ts:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n{'='*70}", flush=True)
    print("=== item vs. zero, averaged over N_DRAWS validation-split reshuffles ===", flush=True)
    for strategy in STRATEGIES:
        z = df[(df['strategy'] == strategy) & (df['init'] == 'zero')]['mean_rmse'].iloc[0]
        i = df[(df['strategy'] == strategy) & (df['init'] == 'item')]['mean_rmse'].iloc[0]
        winner = 'item' if i < z else 'zero'
        print(f"  {strategy}: zero={z:.4f}  item={i:.4f}  diff={i-z:+.4f}  winner={winner}", flush=True)
    print(f"\nSaved to {OUT_CSV}", flush=True)
    print(f"TOTAL TIME: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
