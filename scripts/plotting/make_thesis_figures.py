"""
make_thesis_figures.py
========================
Generates the two main comparison figures from the pipeline's result
CSVs.

Figure 1 (fig_rmse_vs_baselines): RMSE of the four personalised
strategies vs. the non-personalised baseline range, across the four
elicitation budgets.

Figure 2 (fig_ranking_metrics): HR@10 and NDCG@10 of the four
personalised strategies vs. the best baseline, across the four
elicitation budgets.

Usage
-----
    python scripts/plotting/make_thesis_figures.py

Input
-----
    results/personalised_results.csv
    results/baseline_results.csv

Output
------
    figures/fig_rmse_vs_baselines.{pdf,png}
    figures/fig_ranking_metrics.{pdf,png}
"""

from __future__ import annotations

import os
from typing import Dict

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
FIGURES_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'figures')

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.6,
})

K_VALUES = [10, 25, 50, 100]
STRATEGY_STYLE: Dict[str, Dict[str, str]] = {
    'SHHCP': dict(color='#D55E00', marker='o'),
    'SHLCP': dict(color='#0072B2', marker='s'),
    'SHMCP': dict(color='#009E73', marker='^'),
    'SHECP': dict(color='#CC79A7', marker='D'),
}
BASELINE_COLOR = '#7F7F7F'


def main() -> None:
    """Generates both comparison figures and writes them to figures/."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    pers = pd.read_csv(os.path.join(RESULTS_DIR, 'personalised_results.csv'))
    base = pd.read_csv(os.path.join(RESULTS_DIR, 'baseline_results.csv'))

    pers_rmse = pers.groupby(['strategy', 'k'])['rmse'].mean()
    base_rmse_by_k = base.groupby('ItemsShown')['RMSE'].agg(['min', 'max'])

    # ── Figure 1: RMSE, personalised strategies vs. baseline range ──
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    base_min = [base_rmse_by_k.loc[k, 'min'] for k in K_VALUES]
    base_max = [base_rmse_by_k.loc[k, 'max'] for k in K_VALUES]
    ax.fill_between(K_VALUES, base_min, base_max, color=BASELINE_COLOR, alpha=0.25,
                     label='Non-personalised baseline range', zorder=1)
    ax.plot(K_VALUES, base_min, color=BASELINE_COLOR, linewidth=1.0, linestyle=':', zorder=2)
    ax.plot(K_VALUES, base_max, color=BASELINE_COLOR, linewidth=1.0, linestyle=':', zorder=2)

    for strat, style in STRATEGY_STYLE.items():
        y = [pers_rmse[(strat, k)] for k in K_VALUES]
        ax.plot(K_VALUES, y, label=strat, linewidth=1.8, markersize=6, zorder=3, **style)

    ax.set_xscale('log')
    ax.set_xticks(K_VALUES)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel('Elicitation budget $k$ (items shown)')
    ax.set_ylabel('RMSE (lower is better)')
    ax.set_title('Personalised strategies vs. non-personalised baselines')
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig_rmse_vs_baselines.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'fig_rmse_vs_baselines.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_rmse_vs_baselines.{pdf,png}')

    # ── Figure 2: HR@10 / NDCG@10 across k ──
    pers_hr10 = pers.groupby(['strategy', 'k'])['hr10'].mean()
    pers_ndcg10 = pers.groupby(['strategy', 'k'])['ndcg10'].mean()
    base_hr10_by_k = base.groupby('ItemsShown')['HR@10(sampled)'].max()
    base_ndcg10_by_k = base.groupby('ItemsShown')['NDCG@10(sampled)'].max()

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharex=True)

    for ax, (metric_pers, metric_base, title) in zip(
        axes,
        [(pers_hr10, base_hr10_by_k, 'HR@10'), (pers_ndcg10, base_ndcg10_by_k, 'NDCG@10')]
    ):
        ax.plot(K_VALUES, [metric_base[k] for k in K_VALUES],
                color=BASELINE_COLOR, linewidth=1.5, linestyle='--',
                marker='x', label='Best baseline')
        for strat, style in STRATEGY_STYLE.items():
            y = [metric_pers[(strat, k)] for k in K_VALUES]
            ax.plot(K_VALUES, y, label=strat, linewidth=1.8, markersize=6, **style)
        ax.set_xscale('log')
        ax.set_xticks(K_VALUES)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel('Elicitation budget $k$')
        ax.set_title(title)

    axes[0].set_ylabel('Metric value (higher is better)')
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig_ranking_metrics.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'fig_ranking_metrics.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_ranking_metrics.{pdf,png}')


if __name__ == '__main__':
    main()
