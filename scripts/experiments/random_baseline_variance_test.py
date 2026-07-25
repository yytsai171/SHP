"""
random_baseline_variance_test.py
===================================
Investigates why the Random baseline uniquely dips to its best RMSE at
k=50 (0.1611), making it the strongest baseline at k=50 and k=100 --
while Popularity sits at 0.1695-0.1747 (monotonically WORSE with k)
and the personalised strategies' best RMSE at k=50 is SHECP=0.1670,
just above Random.

This is NOT a hunt for a way to make personalised win at k=50 -- it is
a check of whether the reported Random-baseline number at k=50 is even
a *reliable* estimate in the first place, before treating it as
something that needs to be "solved".

The mechanism under test: baseline_ranking_metrics.py's Random strategy
draws exactly ONE fixed set of k items via a single call to
``random.sample(eligible_items, k)``, seeded once globally
(``random.seed(1)``) at the top of the script, and then evaluates every
one of the ~550-585 cold users against that SAME one-shot item set.
There is no averaging over independent random draws -- the reported
"Random RMSE at k=50" is the outcome of a single arbitrary sample of 50
items out of 45,543 eligible ones. Popularity and PopError, by
contrast, select items deterministically (sorted by a fixed score), so
they have no equivalent single-draw sampling variance.

This script redraws the random item-set independently many times (fresh
seed each draw, same 1,000-user population, same baseline_predict
formula and val/test split protocol as baseline_ranking_metrics.py) and
reports the distribution of resulting RMSE at each k. If the originally
reported k=50 value (0.1611) falls comfortably inside this resampling
distribution, that confirms the "k=50 anomaly" is sampling noise from
evaluating against a single random draw, not a real, reproducible
property of k=50 itself -- and the fix is to average Random's RMSE over
many draws (a legitimate methodological correction), not to keep tuning
personalised hyperparameters until something beats one lucky number.

Result: k=10 mean=0.1687, k=25 mean=0.1712, k=50 mean=0.1689 across 30
draws -- the originally reported values (0.1783/0.1719/0.1611) sit at
z=+1.69/+0.16/-1.42 respectively, i.e. within normal single-draw
sampling variance, not outliers. Uses a one-time O(n) per-user lookup
dict instead of an O(n) pandas DataFrame filter per user per draw, to
keep the 30-draw resampling tractable.

Usage
-----
    python scripts/experiments/random_baseline_variance_test.py

Input
-----
    results/base_model_cache.pkl

Output
------
    results/random_baseline_variance_test_results.csv
        Columns: k, draw_seed, n_users, rmse
"""

from __future__ import annotations

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
OUT_CSV: str = os.path.join(RESULTS_DIR, 'random_baseline_variance_test_results.csv')

K_VALUES: List[int] = [10, 25, 50, 100]
N_DRAWS: int = 30
NUM_EVAL_USERS: int = 1000

# The originally reported baseline_results.csv Random RMSE at each k,
# for comparison against this script's resampling distribution.
ORIGINAL_RANDOM_RMSE: Dict[int, float] = {10: 0.1783, 25: 0.1719, 50: 0.1611, 100: 0.1678}


def _stable_seed(u: int, shown: List[Any]) -> int:
    """Identical to baseline_ranking_metrics.py's own function -- see
    README.md "Reproducibility"."""
    import hashlib
    key = f"{int(u)}|" + ','.join(sorted(str(x) for x in shown))
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def main() -> None:
    """Redraws the Random baseline's item set N_DRAWS times per k,
    re-evaluates RMSE each time on the same 1,000-user population, and
    writes ``results/random_baseline_variance_test_results.csv``.
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

    # Fast per-user interaction lookup (CODE_REVIEW.md 1.1 pattern, verified
    # byte-identical to the O(n) pandas-scan approach in
    # fast_lookup_comparison.py) -- built ONCE, not per draw. This is what
    # made the first version of this script take ~19 min/k (a full
    # 2.56M-row pandas filter per user per draw, x30 draws): here it is a
    # single one-time O(n) filter+groupby, then O(1) dict lookups per user.
    eval_users_set = set(int(u) for u in eval_cold_users)
    user_data = data[data['user_idx'].isin(eval_users_set)]
    user_item_interaction: Dict[int, Dict[Any, float]] = {
        int(u): dict(zip(g['itemId'].values, g['interaction'].values))
        for u, g in user_data.groupby('user_idx')
    }
    print(f"[{time.time()-t_start:6.1f}s] Fast per-user lookup built "
          f"({len(user_item_interaction)} users).", flush=True)

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

    def evaluate_random_draw(k: int, draw_seed: int) -> Tuple[float, int]:
        local_rng = random.Random(draw_seed)
        shown = local_rng.sample(eligible_items, k)

        per_user_rmse: List[float] = []
        for u in eval_cold_users:
            user_dict = user_item_interaction.get(int(u))
            if not user_dict:
                continue
            _, test_items = split_unseen_items(u, shown, val_frac=0.5)
            test_set = set(test_items)
            # Iterate the user's own (few) real interactions -- O(1) dict
            # lookups -- instead of scanning the full test-item list or
            # the full DataFrame (see docstring).
            test_rows = [(iid, r) for iid, r in user_dict.items() if iid in test_set]
            if not test_rows:
                continue
            preds = [
                Pred(uid=u, iid=iid, r_ui=r, est=baseline_predict(iid), details={})
                for iid, r in test_rows
            ]
            per_user_rmse.append(accuracy.rmse(preds, verbose=False))

        return (float(np.mean(per_user_rmse)) if per_user_rmse else float('nan'),
                len(per_user_rmse))

    rows = []
    for k in K_VALUES:
        print(f"\n=== k={k}: {N_DRAWS} independent random item-set draws ===", flush=True)
        t_k = time.time()
        for seed in range(N_DRAWS):
            rmse, n_users = evaluate_random_draw(k, seed)
            rows.append({'k': k, 'draw_seed': seed, 'n_users': n_users, 'rmse': rmse})
        draw_rmses = [r['rmse'] for r in rows if r['k'] == k]
        orig = ORIGINAL_RANDOM_RMSE[k]
        mean_r, std_r = np.mean(draw_rmses), np.std(draw_rmses)
        z = (orig - mean_r) / std_r if std_r > 0 else float('nan')
        print(f"  k={k}: resampled RMSE mean={mean_r:.4f} std={std_r:.4f} "
              f"min={min(draw_rmses):.4f} max={max(draw_rmses):.4f}  |  "
              f"originally reported={orig:.4f}  (z={z:+.2f})", flush=True)
        print(f"  k={k} total time: {time.time()-t_k:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n{'='*70}", flush=True)
    print("=== Summary: is the originally reported Random RMSE typical or an outlier? ===",
          flush=True)
    for k in K_VALUES:
        draw_rmses = df[df['k'] == k]['rmse'].tolist()
        orig = ORIGINAL_RANDOM_RMSE[k]
        pct_lower = 100 * np.mean([r <= orig for r in draw_rmses])
        print(f"  k={k}: originally reported RMSE={orig:.4f} is at the "
              f"{pct_lower:.0f}th percentile of {N_DRAWS} independent resampled draws "
              f"(range {min(draw_rmses):.4f}-{max(draw_rmses):.4f})", flush=True)

    print(f"\nSaved to {OUT_CSV}", flush=True)
    print(f"TOTAL TIME: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
