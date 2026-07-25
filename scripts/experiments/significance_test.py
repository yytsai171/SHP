"""
significance_test.py
======================
Paired significance testing: for every (personalised strategy, k,
baseline) combination, tests whether the personalised strategy's RMSE
and HR@10 differ significantly from that baseline's, matched
individual-cold-user-by-individual-cold-user (Wilcoxon signed-rank
test, since RMSE/HR distributions are not assumed normal). A raw
difference in mean RMSE between two strategies can be a sampling
artefact of which specific users and held-out items happened to be
evaluated -- this is why the per-user paired test matters more than
comparing the two aggregate means directly (see README.md
"Reproducing Thesis Results" -> Table 4.4).

Baseline per-user RMSE/HR@10 are not saved by baseline_ranking_metrics.py
(which only reports the aggregated mean per cell), so they are
recomputed here, reusing that script's exact scoring logic and the same
deterministic _stable_seed val/test split.

Usage
-----
    python scripts/experiments/significance_test.py

Input
-----
    results/base_model_cache.pkl
    results/personalised_results.csv   (see personalised_strategies.py
                                         / run_complete_pipeline.py)

Output
------
    results/significance_results.csv
        Columns: strategy, k, baseline, n_paired, mean_rmse_personalised,
        mean_rmse_baseline, rmse_margin (baseline - personalised; positive
        means personalised is more accurate), rmse_p_value,
        rmse_significant_p05, hr10_p_value

Note on multiple comparisons: this script reports 48 simultaneous
hypothesis tests (4 strategies x 3 baselines x 4 k-values). See
ranking_significance_and_correction.py for the Holm-Bonferroni and
Benjamini-Hochberg corrections applied on top of these raw p-values
before any claim is reported as robust (see README.md "Methodology").
"""

from __future__ import annotations

import hashlib
import os
import pickle
import random
import time
from collections import namedtuple
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from surprise import accuracy

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
MODEL_CACHE: str = os.path.join(RESULTS_DIR, 'base_model_cache.pkl')
PERS_CSV: str = os.path.join(RESULTS_DIR, 'personalised_results.csv')
OUT_SIG: str = os.path.join(RESULTS_DIR, 'significance_results.csv')

N_NEG: int = 99
K_VALUES: List[int] = [10, 25, 50, 100]


def _stable_seed(u: int, shown: List[Any]) -> int:
    """Deterministic replacement for Python's built-in ``hash()`` on
    strings -- see README.md "Reproducibility"."""
    key = f"{int(u)}|" + ','.join(sorted(str(x) for x in shown))
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def main() -> None:
    """Runs the paired significance test (personalised vs. each
    baseline, all 48 (strategy, baseline, k) combinations) and writes
    ``results/significance_results.csv``.
    """
    t_start = time.time()

    with open(MODEL_CACHE, 'rb') as f:
        cache = pickle.load(f)

    data           = cache['data']
    eligible_items = cache['eligible_items']
    item_to_iidx   = cache['item_to_iidx']
    svd_base       = cache['svd_base']
    mu_base        = cache['mu_base']
    cold_users     = cache['cold_users']
    ALPHA          = cache['ALPHA']

    print(f"[{time.time()-t_start:6.1f}s] Cache loaded (setup skipped).", flush=True)

    eval_cold_users = cold_users[:min(1000, len(cold_users))]

    warm_data = data[~data['user_idx'].isin(cold_users)]
    item_counts = warm_data['itemId'].value_counts()
    item_mean_interaction = warm_data.groupby('itemId')['interaction'].mean()
    error_scores = {}
    for item in eligible_items:
        p = float(item_mean_interaction.get(item, 0.5))
        error_scores[item] = min(p, 1.0 - p)
    poperror_scores = {
        item: ALPHA * np.log10(item_counts[item]) + (1 - ALPHA) * error_scores[item]
        for item in eligible_items
    }

    random.seed(1)
    neg_rng = np.random.RandomState(42)
    Pred = namedtuple('Prediction', ['uid', 'iid', 'r_ui', 'est', 'details'])

    def split_unseen_items(u: int, shown: List[Any],
                            val_frac: float = 0.5):
        """Validation/test split, deterministically seeded per (u, shown)."""
        shown_set = set(shown)
        unseen    = [iid for iid in eligible_items if iid not in shown_set]
        rng       = np.random.RandomState(_stable_seed(u, shown))
        rng.shuffle(unseen)
        n_val = int(val_frac * len(unseen))
        return unseen[:n_val], unseen[n_val:]

    def select_items_nonpersonalised(strategy: str, k: int) -> List[Any]:
        """Selects the fixed k-item baseline set (thesis Section 3.5)."""
        if strategy == 'random':
            return random.sample(eligible_items, k)
        elif strategy == 'popularity':
            return list(item_counts.loc[eligible_items].sort_values(ascending=False).head(k).index)
        elif strategy == 'poperror':
            return sorted(poperror_scores, key=poperror_scores.get, reverse=True)[:k]
        raise ValueError(f"Unknown strategy: {strategy}")

    def baseline_predict(iid: Any) -> float:
        """Baseline prediction mu + b_i, clipped to [0, 1]."""
        i_inner = item_to_iidx.get(iid)
        if i_inner is None:
            return mu_base
        return float(np.clip(mu_base + svd_base.bi[i_inner], 0, 1))

    def sampled_hr_at_k(pos_score: float, neg_scores: List[float],
                         k_list: List[int]) -> Dict[int, float]:
        """HR@K for one positive item against sampled negatives."""
        rank = 1 + sum(1 for s in neg_scores if s > pos_score)
        return {k: (1.0 if rank <= k else 0.0) for k in k_list}

    # ── Recompute per-user baseline RMSE/HR@10 (not saved by baseline_ranking_metrics.py) ──
    print("\n=== Computing per-user baseline RMSE/HR@10 ===", flush=True)
    baseline_rows = []
    for strat in ['random', 'popularity', 'poperror']:
        for k in K_VALUES:
            t_cell = time.time()
            shown     = select_items_nonpersonalised(strat, k)
            shown_set = set(shown)
            n_done = 0
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
                u_rmse = accuracy.rmse(preds, verbose=False)

                user_interacted = set(data[data['user_idx'] == u]['itemId'].tolist())
                candidate_negs  = [
                    iid for iid in eligible_items
                    if iid not in user_interacted and iid not in shown_set
                    and iid in item_to_iidx
                ]
                pos_test_iids = test_df[test_df['interaction'] == 1]['itemId'].tolist()
                user_hr10 = []
                for pos_iid in pos_test_iids:
                    pos_score = baseline_predict(pos_iid)
                    n_sample = min(N_NEG, len(candidate_negs))
                    if n_sample == 0:
                        continue
                    sampled_neg_iids = neg_rng.choice(candidate_negs, size=n_sample, replace=False)
                    neg_scores = [baseline_predict(nid) for nid in sampled_neg_iids]
                    m = sampled_hr_at_k(pos_score, neg_scores, k_list=[10])
                    user_hr10.append(m[10])

                baseline_rows.append({
                    'strategy': strat, 'k': k, 'user': int(u),
                    'rmse': u_rmse,
                    'hr10': np.mean(user_hr10) if user_hr10 else np.nan,
                })
                n_done += 1
            print(f"  [{strat}][k={k}] {n_done} users done ({time.time()-t_cell:.1f}s)", flush=True)

    baseline_df = pd.DataFrame(baseline_rows)
    print(f"\nBaseline per-user computation done: {time.time()-t_start:.1f}s total", flush=True)

    # ── Paired significance test: personalised vs. each baseline ──
    pers = pd.read_csv(PERS_CSV)
    results = []

    print(f"\n{'='*70}")
    print("=== Personalised vs. each baseline, paired by user ===")
    for strat in ['SHHCP', 'SHLCP', 'SHMCP', 'SHECP']:
        for k in K_VALUES:
            p_df = pers[(pers['strategy'] == strat) & (pers['k'] == k)][['user', 'rmse', 'hr10']]
            for base_strat in ['random', 'popularity', 'poperror']:
                b_df = baseline_df[(baseline_df['strategy'] == base_strat) & (baseline_df['k'] == k)][['user', 'rmse', 'hr10']]
                merged = p_df.merge(b_df, on='user', suffixes=('_pers', '_base'))
                if len(merged) < 10:
                    continue
                diff_rmse = merged['rmse_pers'] - merged['rmse_base']
                if (diff_rmse != 0).sum() >= 1:
                    _, p_rmse = wilcoxon(merged['rmse_pers'], merged['rmse_base'])
                else:
                    p_rmse = 1.0
                merged_hr = merged.dropna(subset=['hr10_pers', 'hr10_base'])
                diff_hr = merged_hr['hr10_pers'] - merged_hr['hr10_base']
                if len(merged_hr) >= 10 and (diff_hr != 0).sum() >= 1:
                    _, p_hr = wilcoxon(merged_hr['hr10_pers'], merged_hr['hr10_base'])
                else:
                    p_hr = float('nan')
                mean_pers_rmse = merged['rmse_pers'].mean()
                mean_base_rmse = merged['rmse_base'].mean()
                row = {
                    'strategy': strat, 'k': k, 'baseline': base_strat,
                    'n_paired': len(merged),
                    'mean_rmse_personalised': round(mean_pers_rmse, 4),
                    'mean_rmse_baseline': round(mean_base_rmse, 4),
                    # positive margin = personalised RMSE lower (better)
                    'rmse_margin': round(mean_base_rmse - mean_pers_rmse, 4),
                    'rmse_p_value': round(p_rmse, 6),
                    'rmse_significant_p05': p_rmse < 0.05,
                    'hr10_p_value': round(p_hr, 6) if not np.isnan(p_hr) else None,
                }
                results.append(row)
                print(f"  {strat}@k={k} vs {base_strat}: RMSE margin={row['rmse_margin']:+.4f} "
                      f"p={p_rmse:.4f} {'SIGNIFICANT' if p_rmse < 0.05 else 'not significant'}",
                      flush=True)

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_SIG, index=False)
    print(f"\nSaved to {OUT_SIG}", flush=True)
    print(f"TOTAL TIME: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
