"""
make_shecp_grid_figure.py
============================
Heatmap of the SHECP epsilon floor/decay validation-RMSE grid search
(see shecp_grid_search.py). Communicates the "decay=0.95 dominates at
every floor" finding faster than the table alone.

Usage
-----
    python scripts/plotting/make_shecp_grid_figure.py

Input
-----
    results/shecp_grid_results.csv   (see shecp_grid_search.py)

Output
------
    figures/fig_shecp_grid.{pdf,png}
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
FIGURES_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'figures')

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def main() -> None:
    """Generates the SHECP floor/decay heatmap and writes it to figures/."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(RESULTS_DIR, 'shecp_grid_results.csv'))

    floors = sorted(df['floor'].unique())
    decays = sorted(df['decay'].unique())
    grid = df.pivot(index='floor', columns='decay', values='val_rmse').loc[floors, decays]

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(grid.values, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(range(len(decays)))
    ax.set_xticklabels([f'{d:.2f}' for d in decays])
    ax.set_yticks(range(len(floors)))
    ax.set_yticklabels([f'{f:.2f}' for f in floors])
    ax.set_xlabel(r'Decay rate')
    ax.set_ylabel(r'Exploration floor')
    ax.set_title('SHECP floor/decay grid search: validation RMSE at $k=50$')

    best_val = grid.values.min()
    for i in range(len(floors)):
        for j in range(len(decays)):
            val = grid.values[i, j]
            is_best = np.isclose(val, best_val)
            ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                    fontsize=10, fontweight='bold' if is_best else 'normal',
                    color='black')
            if is_best:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                            edgecolor='black', linewidth=2.2))

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Validation RMSE (lower is better)')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig_shecp_grid.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'fig_shecp_grid.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_shecp_grid.{pdf,png}')
    print(grid)


if __name__ == '__main__':
    main()
