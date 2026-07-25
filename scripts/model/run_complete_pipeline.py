"""
run_complete_pipeline.py
==========================
Runs the complete experimental pipeline end-to-end, in one command:

    1. build_model_cache.py          -- train the base SVD, tune ALPHA
                                         and (F, reg_all, epochs)
    2. baseline_ranking_metrics.py   -- Random / Popularity / PopError,
                                         RMSE + HR@K/NDCG@K, 1,000 users
    3. personalised_strategies.py    -- SHHCP / SHLCP / SHMCP / SHECP,
                                         all corrections applied,
                                         1,000 users x 4 k-values

Each stage is a separate, independently runnable script -- this one
simply calls them in the right order and stops early if any stage
fails or setup hasn't been run yet.

Usage
-----
    python scripts/model/run_complete_pipeline.py

Expected runtime
-----------------
    Setup (build_model_cache.py):        ~20-25 minutes  (one-time;
                                          skipped if
                                          results/base_model_cache.pkl
                                          already exists)
    Baselines (baseline_ranking_metrics.py): ~30 minutes (each of
                                          Random/Popularity/PopError
                                          averaged over 30 independent
                                          draws; see baseline_ranking_
                                          metrics.py's module docstring)
    Personalised (personalised_strategies.py): ~80-85 minutes
                                          (1,000 users x 4 strategies
                                          x 4 k-values = 16,000 work
                                          items, 10 workers)
    Total: ~1.8-2.3 hours on a modern multi-core machine.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import sys
from typing import List

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
MODEL_CACHE: str = os.path.join(RESULTS_DIR, 'base_model_cache.pkl')

# Full-scale evaluation parameters.
FULL_STRATEGIES: List[str] = ['SHHCP', 'SHLCP', 'SHMCP', 'SHECP']
FULL_K_VALUES: List[int] = [10, 25, 50, 100]
FULL_NUM_USERS: int = 1000
FULL_N_WORKERS: int = 10


def run_script(name: str, directory: str = SCRIPT_DIR) -> None:
    """Runs another script as a subprocess and raises if it fails.

    Each stage script is fully self-contained (own imports, own I/O
    paths resolved relative to its own location), so subprocess
    isolation is used here rather than importing every stage as a
    module -- this keeps each stage's global state independent and
    matches how a user would run any one stage manually.

    Parameters
    ----------
    name : str
        Filename of the script to run.
    directory : str, default this file's own directory (scripts/model/)
        Directory the script lives in.

    Raises
    ------
    RuntimeError
        If the subprocess exits with a non-zero return code.
    """
    print(f"\n{'='*70}\nRunning {name} ...\n{'='*70}", flush=True)
    result = subprocess.run([sys.executable, os.path.join(directory, name)],
                             cwd=directory)
    if result.returncode != 0:
        raise RuntimeError(f"{name} exited with code {result.returncode} -- stopping pipeline.")


def main() -> None:
    """Runs all three pipeline stages in sequence.

    Stage 3 (the personalised strategies) is the one stage NOT run via
    ``run_script`` -- personalised_strategies.py's own ``__main__``
    only runs a 20-user smoke test (a fast correctness check before
    committing to the ~80-minute full run), so this function imports
    that module directly and calls the same smoke-test-then-full-run
    sequence in-process, saving the result itself.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if os.path.exists(MODEL_CACHE):
        print(f"Found existing {MODEL_CACHE} -- skipping build_model_cache.py. "
              "Delete this file first if you want to rebuild the base model "
              "from scratch.", flush=True)
    else:
        run_script('build_model_cache.py')

    run_script('baseline_ranking_metrics.py')

    print(f"\n{'='*70}\nRunning personalised strategy evaluation "
          f"({FULL_NUM_USERS} users x {len(FULL_STRATEGIES)} strategies "
          f"x {len(FULL_K_VALUES)} k-values) ...\n{'='*70}", flush=True)
    mp.set_start_method('spawn', force=True)
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    import personalised_strategies as pers

    df_smoke, _ = pers.run(
        n_workers=4, strategies=['SHLCP'], k_values=[10], num_users=20,
        label='SMOKE TEST (correctness check)'
    )
    if df_smoke.empty or df_smoke['rmse'].isna().all():
        raise RuntimeError("Smoke test produced no valid results -- stopping "
                            "before the full run.")
    print("Smoke test passed. Proceeding to full-scale evaluation.", flush=True)

    df_full, elapsed = pers.run(
        n_workers=FULL_N_WORKERS, strategies=FULL_STRATEGIES,
        k_values=FULL_K_VALUES, num_users=FULL_NUM_USERS,
        label='FULL EVALUATION'
    )
    df_full.to_csv(pers.OUT_FINAL, index=False)
    print(f"Saved to {pers.OUT_FINAL}", flush=True)

    print(f"\n{'='*70}\nPIPELINE COMPLETE.\n{'='*70}", flush=True)
    print("Results written to:", flush=True)
    print(f"  {os.path.join(RESULTS_DIR, 'baseline_results.csv')}", flush=True)
    print(f"  {os.path.join(RESULTS_DIR, 'personalised_results.csv')}", flush=True)


if __name__ == '__main__':
    main()
