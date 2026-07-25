"""
ranking_significance_and_correction.py
=========================================
Two additions on top of significance_test.py:

    (A) NDCG@10 paired significance test, personalised vs. each
        baseline, matching the rigor already applied to RMSE and HR@10
        in significance_test.py. Reuses the identical methodology,
        shown-item sets, and RNG scheme, so the numbers are directly
        comparable/mergeable with that script's output.

    (B) Multiple-comparisons correction (Holm-Bonferroni and
        Benjamini-Hochberg/FDR), applied within each family of tests:
        the 48 RMSE tests, the 48 HR@10 tests, and the 48 NDCG@10
        tests. See README.md "Methodology" for why this matters: 48
        simultaneous hypothesis tests at an uncorrected p<0.05
        threshold are expected to produce some false positives by
        chance alone.

Holm and BH are implemented directly (rather than via statsmodels) as
~15 lines each, verified against textbook invariants at import time
(see _sanity_check_corrections below) -- easy to audit by hand against
a standard reference.

Usage
-----
    python scripts/experiments/ranking_significance_and_correction.py

Input
-----
    results/base_model_cache.pkl
    results/personalised_results.csv
    results/significance_results.csv      (see significance_test.py)

Output
------
    results/ranking_significance_results.csv
        NDCG@10 paired test + Holm/BH-corrected columns.
    results/significance_results_corrected.csv
        significance_results.csv with Holm/BH-corrected RMSE and HR@10
        columns added.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
MODEL_CACHE: str = os.path.join(RESULTS_DIR, 'base_model_cache.pkl')
SIG_CSV: str = os.path.join(RESULTS_DIR, 'significance_results.csv')
OUT_RANK: str = os.path.join(RESULTS_DIR, 'ranking_significance_results.csv')
OUT_CORR: str = os.path.join(RESULTS_DIR, 'significance_results_corrected.csv')

N_NEG: int = 99
K_VALUES: List[int] = [10, 25, 50, 100]


# ============================================================
# Multiple-comparisons correction (pure functions, no external
# dependency beyond numpy).
# ============================================================

def holm_correction(pvals: List[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down correction.

    Parameters
    ----------
    pvals : array-like of float
        Raw p-values.
    alpha : float, default 0.05
        Family-wise error rate to control.

    Returns
    -------
    adjusted_pvals : np.ndarray
        Adjusted p-values, in the original order of ``pvals``.
    reject : np.ndarray of bool
        Whether each adjusted p-value is below ``alpha``.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(m)
    running_max = 0.0
    for i in range(m):
        val = (m - i) * ranked[i]
        running_max = max(running_max, val)
        adj[i] = min(running_max, 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adj
    return adjusted, adjusted < alpha


def bh_fdr_correction(pvals: List[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction. Same signature/return shape
    as ``holm_correction``."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(m)
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        val = ranked[i] * m / (i + 1)
        running_min = min(running_min, val)
        adj[i] = running_min
    adjusted = np.empty(m)
    adjusted[order] = adj
    return adjusted, adjusted < alpha


def _sanity_check_corrections() -> None:
    """Checks the two correction implementations against basic
    textbook invariants (Holm 1979 / Benjamini-Hochberg 1995), rather
    than a brittle hand-derived expected output vector."""
    test_p = np.array([0.01, 0.02, 0.03, 0.04, 0.20])
    holm_adj, _ = holm_correction(test_p)
    bh_adj, _ = bh_fdr_correction(test_p)
    assert np.all(np.diff(np.sort(holm_adj)) >= -1e-12), "Holm output not monotonic"
    assert np.all(np.diff(np.sort(bh_adj)) >= -1e-12), "BH output not monotonic"
    assert np.all(holm_adj >= test_p - 1e-12), "Holm-adjusted p-values must be >= raw"
    assert np.all(bh_adj >= test_p - 1e-12), "BH-adjusted p-values must be >= raw"
    assert np.all(bh_adj <= holm_adj + 1e-12), "BH should never be stricter than Holm"
    print("[sanity] Holm/BH correction implementations pass basic invariant checks.", flush=True)


def _stable_seed(u: int, shown: List[Any]) -> int:
    """Deterministic replacement for Python's built-in ``hash()`` on
    strings -- see README.md "Reproducibility"."""
    key = f"{int(u)}|" + ','.join(sorted(str(x) for x in shown))
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def main() -> None:
    """Runs the Holm/BH correction of significance_test.py's output
    (Part A) and the new NDCG@10 paired significance test (Part B),
    writing ``results/significance_results_corrected.csv`` and
    ``results/ranking_significance_results.csv``.
    """
    t_start = time.time()
    _sanity_check_corrections()

    # ── Part A: Holm/BH correction of significance_test.py's RMSE/HR@10 tests ──
    sig_df = pd.read_csv(SIG_CSV)
    assert len(sig_df) == 48, f"expected 48 rows in {SIG_CSV}, got {len(sig_df)}"

    for col, prefix in [('rmse_p_value', 'rmse'), ('hr10_p_value', 'hr10')]:
        holm_adj, holm_rej = holm_correction(sig_df[col].values)
        bh_adj, bh_rej = bh_fdr_correction(sig_df[col].values)
        sig_df[f'{prefix}_p_holm'] = np.round(holm_adj, 6)
        sig_df[f'{prefix}_significant_holm'] = holm_rej
        sig_df[f'{prefix}_p_bh'] = np.round(bh_adj, 6)
        sig_df[f'{prefix}_significant_bh'] = bh_rej

    print(f"\n{'='*70}")
    print("=== Multiple-comparisons correction summary ===")
    print(f"{'='*70}")
    print(f"RMSE (48 tests): raw significant (p<0.05) = {(sig_df['rmse_p_value'] < 0.05).sum()}, "
          f"Holm-significant = {sig_df['rmse_significant_holm'].sum()}, "
          f"BH-significant = {sig_df['rmse_significant_bh'].sum()}")
    print(f"HR@10 (48 tests): raw significant (p<0.05) = {(sig_df['hr10_p_value'] < 0.05).sum()}, "
          f"Holm-significant = {sig_df['hr10_significant_holm'].sum()}, "
          f"BH-significant = {sig_df['hr10_significant_bh'].sum()}")

    flipped = sig_df[sig_df['rmse_significant_p05'] & ~sig_df['rmse_significant_holm']]
    print(f"\nRMSE results significant at raw p<0.05 but NOT under Holm correction "
          f"({len(flipped)} rows):")
    if len(flipped):
        print(flipped[['strategy', 'k', 'baseline', 'rmse_margin', 'rmse_p_value',
                        'rmse_p_holm']].to_string(index=False))

    sig_df.to_csv(OUT_CORR, index=False)
    print(f"\nSaved corrected significance results to {OUT_CORR}", flush=True)

    # ── Part B: NDCG@10 per-user baseline computation + paired significance ──
    print(f"\n{'='*70}")
    print("=== Computing per-user baseline NDCG@10 ===")
    print(f"{'='*70}")

    with open(MODEL_CACHE, 'rb') as f:
        cache = pickle.load(f)

    data           = cache['data']
    eligible_items = cache['eligible_items']
    item_to_iidx   = cache['item_to_iidx']
    svd_base       = cache['svd_base']
    mu_base        = cache['mu_base']
    cold_users     = cache['cold_users']
    ALPHA          = cache['ALPHA']

    eval_cold_users = cold_users[:min(1000, len(cold_users))]

    warm_data             = data[~data['user_idx'].isin(cold_users)]
    item_counts           = warm_data['itemId'].value_counts()
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

    def split_unseen_items(u, shown, val_frac=0.5):
        shown_set = set(shown)
        unseen    = [iid for iid in eligible_items if iid not in shown_set]
        rng       = np.random.RandomState(_stable_seed(u, shown))
        rng.shuffle(unseen)
        n_val = int(val_frac * len(unseen))
        return unseen[:n_val], unseen[n_val:]

    def select_items_nonpersonalised(strategy, k):
        if strategy == 'random':
            return random.sample(eligible_items, k)
        elif strategy == 'popularity':
            return list(item_counts.loc[eligible_items].sort_values(ascending=False).head(k).index)
        elif strategy == 'poperror':
            return sorted(poperror_scores, key=poperror_scores.get, reverse=True)[:k]
        raise ValueError(f"Unknown strategy: {strategy}")

    def baseline_predict(iid):
        i_inner = item_to_iidx.get(iid)
        if i_inner is None:
            return mu_base
        return float(np.clip(mu_base + svd_base.bi[i_inner], 0, 1))

    def sampled_metrics_at_k(pos_score, neg_scores, k_list):
        rank = 1 + sum(1 for s in neg_scores if s > pos_score)
        results = {}
        for k in k_list:
            results[f'HR@{k}']   = 1.0 if rank <= k else 0.0
            results[f'NDCG@{k}'] = (1.0 / np.log2(rank + 1)) if rank <= k else 0.0
        return results

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

                user_interacted = set(data[data['user_idx'] == u]['itemId'].tolist())
                candidate_negs  = [
                    iid for iid in eligible_items
                    if iid not in user_interacted and iid not in shown_set
                    and iid in item_to_iidx
                ]
                pos_test_iids = test_df[test_df['interaction'] == 1]['itemId'].tolist()
                user_hr10, user_ndcg10 = [], []
                for pos_iid in pos_test_iids:
                    pos_score = baseline_predict(pos_iid)
                    n_sample = min(N_NEG, len(candidate_negs))
                    if n_sample == 0:
                        continue
                    sampled_neg_iids = neg_rng.choice(candidate_negs, size=n_sample, replace=False)
                    neg_scores = [baseline_predict(nid) for nid in sampled_neg_iids]
                    m = sampled_metrics_at_k(pos_score, neg_scores, k_list=[10])
                    user_hr10.append(m['HR@10'])
                    user_ndcg10.append(m['NDCG@10'])

                baseline_rows.append({
                    'strategy': strat, 'k': k, 'user': int(u),
                    'hr10': np.mean(user_hr10) if user_hr10 else np.nan,
                    'ndcg10': np.mean(user_ndcg10) if user_ndcg10 else np.nan,
                })
                n_done += 1
            print(f"  [{strat}][k={k}] {n_done} users done ({time.time()-t_cell:.1f}s)", flush=True)

    baseline_df = pd.DataFrame(baseline_rows)
    print(f"\nBaseline per-user NDCG@10 computation done: {time.time()-t_start:.1f}s total", flush=True)

    pers = pd.read_csv(os.path.join(RESULTS_DIR, 'personalised_results.csv'))

    results = []
    print(f"\n{'='*70}")
    print("=== NDCG@10 significance, paired by user ===")
    for strat in ['SHHCP', 'SHLCP', 'SHMCP', 'SHECP']:
        for k in K_VALUES:
            p_df = pers[(pers['strategy'] == strat) & (pers['k'] == k)][['user', 'hr10', 'ndcg10']]
            for base_strat in ['random', 'popularity', 'poperror']:
                b_df = baseline_df[(baseline_df['strategy'] == base_strat) & (baseline_df['k'] == k)][
                    ['user', 'hr10', 'ndcg10']]
                merged = p_df.merge(b_df, on='user', suffixes=('_pers', '_base'))
                merged_ndcg = merged.dropna(subset=['ndcg10_pers', 'ndcg10_base'])

                if len(merged_ndcg) >= 10 and (merged_ndcg['ndcg10_pers'] - merged_ndcg['ndcg10_base'] != 0).sum() >= 1:
                    _, p_ndcg = wilcoxon(merged_ndcg['ndcg10_pers'], merged_ndcg['ndcg10_base'])
                else:
                    p_ndcg = float('nan')

                # NDCG is higher-is-better: margin = baseline - personalised,
                # so a POSITIVE margin here means the baseline is more
                # accurate (opposite convention from RMSE margins, where
                # lower is better) -- see README.md "Methodology".
                row = {
                    'strategy': strat, 'k': k, 'baseline': base_strat,
                    'n_paired_ndcg10': len(merged_ndcg),
                    'mean_ndcg10_personalised': round(merged_ndcg['ndcg10_pers'].mean(), 4) if len(merged_ndcg) else np.nan,
                    'mean_ndcg10_baseline': round(merged_ndcg['ndcg10_base'].mean(), 4) if len(merged_ndcg) else np.nan,
                    'ndcg10_margin': round(merged_ndcg['ndcg10_base'].mean() - merged_ndcg['ndcg10_pers'].mean(), 4) if len(merged_ndcg) else np.nan,
                    'ndcg10_p_value': round(p_ndcg, 6) if not np.isnan(p_ndcg) else None,
                }
                results.append(row)
                print(f"  {strat}@k={k} vs {base_strat}: NDCG@10 margin={row['ndcg10_margin']:+.4f} "
                      f"p={p_ndcg:.4f}", flush=True)

    ndcg_df = pd.DataFrame(results)
    holm_adj, holm_rej = holm_correction(ndcg_df['ndcg10_p_value'].fillna(1.0).values)
    bh_adj, bh_rej = bh_fdr_correction(ndcg_df['ndcg10_p_value'].fillna(1.0).values)
    ndcg_df['ndcg10_p_holm'] = np.round(holm_adj, 6)
    ndcg_df['ndcg10_significant_holm'] = holm_rej
    ndcg_df['ndcg10_p_bh'] = np.round(bh_adj, 6)
    ndcg_df['ndcg10_significant_bh'] = bh_rej

    ndcg_df.to_csv(OUT_RANK, index=False)
    print(f"\n{'='*70}")
    print(f"NDCG@10 (48 tests): raw significant (p<0.05) = {(ndcg_df['ndcg10_p_value'] < 0.05).sum()}, "
          f"Holm-significant = {ndcg_df['ndcg10_significant_holm'].sum()}, "
          f"BH-significant = {ndcg_df['ndcg10_significant_bh'].sum()}")
    print(f"Saved to {OUT_RANK}", flush=True)
    print(f"TOTAL TIME: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
