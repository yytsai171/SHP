"""
personalised_strategies.py
============================
Implements and evaluates the four confidence-based personalised active
learning strategies (SHHCP, SHLCP, SHMCP, SHECP) with the final,
adopted correction mechanisms:

    - USE_DECAYING_LR : decaying learning rate for the incremental
      partial-SGD update, gamma_eff(t) = gamma0 / sqrt(1+t)
      (see README.md "Methodology" -> "Partial SGD").
    - SHRINKAGE_C     : confidence-weighted shrinkage constant c=100,
      alpha(k) = k / (k + c), blending the personalised prediction
      toward the stable item-level baseline in proportion to how
      little evidence has been observed (see README.md "Methodology"
      -> "Confidence Shrinkage").
    - SHECP_FLOOR / SHECP_DECAY : SHECP's epsilon-greedy exploration
      floor and decay rate, tuned on the full 1,000-user evaluation
      population (see README.md "Configuration").

Uses a per-user local copy of the item vectors (local_qi/local_bi) so
that the partial-SGD update performed for one cold user never leaks
into another cold user's evaluation, while the shared base model
(the item vectors and biases learned once on warm users) stays frozen
throughout. See README.md "Methodology" -> "Active Learning" for the
full algorithm description.

Runs in parallel via multiprocessing.Pool (one worker process per CPU
core requested); each worker independently evaluates a subset of
(strategy, k, user) combinations.

Usage
-----
Run directly to execute a 20-user smoke test only (a quick correctness
check, not the full experiment):

    python scripts/model/personalised_strategies.py

For the full 1,000-user x 4-strategy x 4-k evaluation, use this
module's run() function via scripts/model/run_complete_pipeline.py, which
also runs the non-personalised baselines and significance tests in the
same pass:

    python scripts/model/run_complete_pipeline.py

Input
-----
    results/base_model_cache.pkl   (see build_model_cache.py)

Output
------
    results/personalised_results.csv
        Columns: strategy, k, user, rmse, hr5, hr10, ndcg5, ndcg10

Complexity
----------
Setup (build_model_cache.py) is O(n) in dataset size; per-user
per-interaction cost here is O(F) (F = number of latent factors),
independent of dataset size -- see thesis Section 3.7.1 ("Motivation
and Efficiency Gain") for the full argument and measure_update_cost.py
for the measured wall-clock comparison against full model retraining.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import pickle
import time
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from surprise import SVD, accuracy

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR: str = os.path.join(SCRIPT_DIR, '..', '..', 'results')
MODEL_CACHE: str = os.path.join(RESULTS_DIR, 'base_model_cache.pkl')
OUT_FINAL: str = os.path.join(RESULTS_DIR, 'personalised_results.csv')

# Number of sampled negatives per positive item in HR@K/NDCG@K
# evaluation (He et al., 2017 methodology; thesis Section 3.9).
N_NEG: int = 99

# Number of partial-SGD steps applied per revealed interaction. Tuned
# to 1 via the SGD-steps ablation (thesis Table 3.7) -- more steps let
# the model overfit each individual, noisy binary response.
NUM_SGD_STEPS: int = 1

# SHECP's epsilon-greedy exploration floor/decay: epsilon_r =
# max(SHECP_FLOOR, SHECP_DECAY ** r). Tuned on the full 1,000-user
# evaluation population -- see shecp_grid_search.py and thesis
# Table 3.8 / "Validating the floor choice" paragraph.
SHECP_FLOOR: float = 0.05
SHECP_DECAY: float = 0.95

# Number of items revealed per active-learning round before the next
# selection is made. Thesis Section 3.6/Chapter 5 Limitation 6.
BATCH_SIZE: int = 3

# Final, adopted correction mechanisms (see module docstring above).
USE_DECAYING_LR: bool = True
SHRINKAGE_C: Optional[int] = 100

Pred = namedtuple('Prediction', ['uid', 'iid', 'r_ui', 'est', 'details'])

# ── Globals populated once per worker process (via pool initializer) ──────
# multiprocessing.Pool with the 'spawn' start method re-imports this
# module fresh in every worker process, so each worker gets its own
# copy of these globals, populated once by _worker_init and read (never
# mutated) by every subsequent call to _select_batch in that worker --
# this is what makes it safe to use plain module-level globals here
# rather than passing the cache through every function call.
_cache: Optional[Dict[str, Any]] = None
_eligible_inner_indices: Optional[np.ndarray] = None
_eligible_iid_array: Optional[np.ndarray] = None
_inner_to_position: Optional[Dict[int, int]] = None
_base_qi_eligible: Optional[np.ndarray] = None
_base_bi_eligible: Optional[np.ndarray] = None


def _worker_init() -> None:
    """Runs once per worker process when the Pool starts.

    Loads the model cache and pre-computes the eligible-item latent-
    vector/bias arrays used by every subsequent call to ``_select_batch``
    in that worker, so per-user work does not re-load or re-index
    anything.

    Side Effects
    ------------
    Populates the module-level ``_cache``, ``_eligible_inner_indices``,
    ``_eligible_iid_array``, ``_inner_to_position``,
    ``_base_qi_eligible``, ``_base_bi_eligible`` globals in the calling
    worker process.
    """
    global _cache, _eligible_inner_indices, _eligible_iid_array
    global _inner_to_position, _base_qi_eligible, _base_bi_eligible
    with open(MODEL_CACHE, 'rb') as f:
        _cache = pickle.load(f)
    eligible_items = _cache['eligible_items']
    item_to_iidx = _cache['item_to_iidx']
    svd_base = _cache['svd_base']
    _eligible_inner_indices = np.array([item_to_iidx[iid] for iid in eligible_items])
    _eligible_iid_array = np.array(eligible_items)
    _inner_to_position = {inner: pos for pos, inner in enumerate(_eligible_inner_indices)}
    _base_qi_eligible = svd_base.qi[_eligible_inner_indices].copy()
    _base_bi_eligible = svd_base.bi[_eligible_inner_indices].copy()


def _seeded_rng(*parts: Any) -> np.random.RandomState:
    """Builds a deterministic per-work-item random number generator.

    Uses hashlib.md5 rather than Python's built-in ``hash()`` --
    ``hash()`` on strings is randomised per-process (PYTHONHASHSEED)
    unless explicitly fixed, which would otherwise give a different
    epsilon-greedy/negative-sampling draw on every run, even for the
    same work item, since each worker process spawned by mp.Pool is a
    fresh interpreter with its own random hash seed.

    Parameters
    ----------
    *parts : Any
        Values identifying the work item (e.g. user id, strategy, k,
        purpose tag); stringified and joined to form the hash key.

    Returns
    -------
    np.random.RandomState
        A generator seeded deterministically from ``parts``, identical
        across repeated runs and across worker processes.

    See Also
    --------
    README.md "Reproducibility".
    """
    key = '|'.join(str(p) for p in parts)
    seed = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)
    return np.random.RandomState(seed)


def _partial_lfm_update_cold(
    svd_model: SVD, pu_cold: np.ndarray, bu_cold: float, i_inner: int, r_ui: float,
    local_qi: Dict[int, np.ndarray], local_bi: Dict[int, float],
    gamma1: float, gamma2: float, lmbda1: float, lmbda2: float,
    num_sgd_steps: int = 1, update_index: int = 0
) -> Tuple[np.ndarray, float]:
    """Applies one (or ``num_sgd_steps``) partial-SGD update(s) to a cold
    user's own parameters given a single newly-observed interaction.

    Updates only the four quantities directly involved in predicting
    ``(u, i_inner)``: the cold user's factor vector ``pu_cold`` and bias
    ``bu_cold``, and the *local copy* of item ``i_inner``'s vector/bias
    (``local_qi``/``local_bi``, created on first touch as a copy of the
    frozen base model's values). All other model parameters -- every
    other item's vector/bias, every warm user's parameters -- are
    untouched. This is what keeps the per-interaction cost O(F)
    (independent of dataset size) instead of O(warm_interactions x F)
    for a full retrain -- see thesis Section 3.7.1 and Eq. 3.9-3.10.

    If ``USE_DECAYING_LR``, ``gamma1``/``gamma2`` are scaled by
    ``1/sqrt(1+update_index)`` before use (thesis Eq. "gamma_eff"),
    confirmed to reduce validation RMSE in decaying_lr_test.py.

    Parameters
    ----------
    svd_model : surprise.SVD
        The frozen base model (only used for its global mean and, on
        first touch of an item, its base ``qi``/``bi`` to seed the
        local copy).
    pu_cold : np.ndarray, shape (F,)
        The cold user's current latent factor vector.
    bu_cold : float
        The cold user's current bias.
    i_inner : int
        Surprise inner index of the just-revealed item.
    r_ui : float
        The observed interaction value (0.0 or 1.0) for this item.
    local_qi : dict[int, np.ndarray]
        Per-user local copy of item factor vectors touched so far this
        session; mutated in place.
    local_bi : dict[int, float]
        Per-user local copy of item biases touched so far this session;
        mutated in place.
    gamma1, gamma2 : float
        Base learning rates for the bias and factor-vector updates.
    lmbda1, lmbda2 : float
        L2 regularisation coefficients for the bias and factor-vector
        updates.
    num_sgd_steps : int, default 1
        Number of SGD steps to apply for this single revealed
        interaction (tuned to 1; see ``NUM_SGD_STEPS``).
    update_index : int, default 0
        Count of prior local updates already applied to this user's
        (pu_cold, bu_cold) this session (0-indexed); used only for the
        decaying-learning-rate schedule.

    Returns
    -------
    pu_cold : np.ndarray, shape (F,)
        The updated cold-user factor vector.
    bu_cold : float
        The updated cold-user bias.

    Notes
    -----
    ``local_qi``/``local_bi`` are mutated in place as a side effect
    (the updated item vector/bias for ``i_inner`` is written back into
    them); only ``pu_cold``/``bu_cold`` are returned, since those are
    not stored in a dict keyed by item.
    """
    if USE_DECAYING_LR:
        decay  = 1.0 / np.sqrt(1.0 + update_index)
        gamma1 = gamma1 * decay
        gamma2 = gamma2 * decay

    if i_inner not in local_qi:
        local_qi[i_inner] = svd_model.qi[i_inner].copy()
        local_bi[i_inner] = float(svd_model.bi[i_inner])
    mu = svd_model.trainset.global_mean
    for _ in range(num_sgd_steps):
        qi  = local_qi[i_inner]
        bi  = local_bi[i_inner]
        pred  = mu + bu_cold + bi + np.dot(pu_cold, qi)
        error = r_ui - pred
        bu_cold           = bu_cold + gamma1 * (error - lmbda1 * bu_cold)
        local_bi[i_inner] = bi      + gamma1 * (error - lmbda1 * bi)
        pu_new            = pu_cold + gamma2 * (error * qi      - lmbda2 * pu_cold)
        local_qi[i_inner] = qi      + gamma2 * (error * pu_cold - lmbda2 * qi)
        pu_cold = pu_new
    return pu_cold, bu_cold


def _shrink_alpha(k: int) -> float:
    """Computes the confidence-weighted shrinkage mixing weight alpha(k).

    ``alpha(k) = k / (k + SHRINKAGE_C)`` (thesis Eq. "shrinkage",
    empirical-Bayes-style; Efron & Morris, 1975). As ``k`` grows,
    ``alpha -> 1`` and the personalised term dominates the prediction;
    as ``k -> 0``, ``alpha -> 0`` and the prediction reduces to the
    stable item-level baseline ``mu + b_i``. If ``SHRINKAGE_C`` is
    None, alpha is always 1 (no shrinkage).

    Parameters
    ----------
    k : int
        Elicitation budget (total items revealed to the cold user this
        session).

    Returns
    -------
    float
        The mixing weight in [0, 1).
    """
    return 1.0 if SHRINKAGE_C is None else k / (k + SHRINKAGE_C)


# Global base-model mean, set once per (strategy, k, user) work item in
# process_one_user from the worker-local _cache. List-boxed (rather than
# a plain module-level float) so _score can read the current value
# without a `global` statement, since it never itself needs to assign
# to this name -- only mu_base_global[0] is mutated, by the caller.
mu_base_global: List[Optional[float]] = [None]


def _score(bu_cold: float, bi: float, pu_cold: np.ndarray, qi: np.ndarray,
           alpha: float) -> float:
    """Computes the shrinkage-weighted predicted interaction score.

    ``score = clip(mu + b_i + alpha * (b_u^c + p_u^c . q_i), 0, 1)``
    (thesis Eq. "shrinkage prediction").

    Parameters
    ----------
    bu_cold : float
        Cold user's current bias.
    bi : float
        Item's bias (base or locally-updated).
    pu_cold : np.ndarray, shape (F,)
        Cold user's current factor vector.
    qi : np.ndarray, shape (F,)
        Item's factor vector (base or locally-updated).
    alpha : float
        Shrinkage mixing weight from ``_shrink_alpha``.

    Returns
    -------
    float
        Predicted score, clipped to [0, 1] (the interaction label
        range).
    """
    return float(np.clip(mu_base_global[0] + bi + alpha * (bu_cold + np.dot(pu_cold, qi)), 0, 1))


def _select_batch(
    svd_base: SVD, pu_cold: np.ndarray, bu_cold: float, shown: List[Any],
    eligible_items: List[Any], item_to_iidx: Dict[Any, int],
    local_qi: Dict[int, np.ndarray], local_bi: Dict[int, float],
    strategy: str, batch_size: int, round_number: int,
    egreedy_rng_local: np.random.RandomState
) -> List[Any]:
    """Selects the next batch of items to show a cold user.

    Scores every not-yet-shown eligible item under the *raw* (non-
    shrunk) prediction, then picks ``batch_size`` of them according to
    ``strategy``:

    - SHHCP: the ``batch_size`` highest-scoring items (pure exploitation).
    - SHLCP: the ``batch_size`` lowest-scoring items (pure exploration).
    - SHMCP: the ``batch_size`` items closest to the median score
      (uncertainty sampling).
    - SHECP: with probability ``epsilon_r = max(SHECP_FLOOR,
      SHECP_DECAY ** round_number)``, behaves like SHLCP (explore);
      otherwise like SHHCP (exploit).

    Shrinkage (``_shrink_alpha``) is applied only at final scoring/
    evaluation time, never during item selection here -- alpha(k)
    depends on total session length, not per-round state, so
    "un-shrinking" the selection score keeps the item-ranking decision
    independent of how many items will ultimately be shown.

    Parameters
    ----------
    svd_base : surprise.SVD
        The frozen base model.
    pu_cold : np.ndarray, shape (F,)
        Cold user's current factor vector.
    bu_cold : float
        Cold user's current bias.
    shown : list
        Raw itemId values already shown to this user this session.
    eligible_items : list
        All raw itemId values eligible for selection.
    item_to_iidx : dict
        Mapping from raw itemId to Surprise inner item index.
    local_qi, local_bi : dict
        Per-user local copies of item vectors/biases touched so far.
    strategy : {'SHHCP', 'SHLCP', 'SHMCP', 'SHECP'}
        Which selection rule to apply.
    batch_size : int
        Number of items to select.
    round_number : int
        0-indexed active-learning round (one round = one batch); used
        only by SHECP's epsilon schedule.
    egreedy_rng_local : np.random.RandomState
        Per-work-item RNG for SHECP's explore/exploit coin flip.

    Returns
    -------
    list
        Up to ``batch_size`` raw itemId values, selected according to
        ``strategy``. May return fewer than ``batch_size`` (or an
        empty list) if fewer eligible unseen items remain.

    Raises
    ------
    RuntimeError
        If called before ``_worker_init`` has populated the module-
        level eligible-item arrays in this worker process.
    ValueError
        If ``strategy`` is not one of the four recognised strategies.

    Complexity
    ----------
    O(|eligible_items|) for the vectorised scoring pass, plus
    O(|eligible_items|) for ``top_b``'s partial sort (``np.argpartition``
    is O(n), avoiding a full O(n log n) sort since only the top/bottom
    ``batch_size`` elements are needed).
    """
    if _base_qi_eligible is None:
        raise RuntimeError("_select_batch called before _worker_init -- "
                            "Pool must use initializer=_worker_init")

    mu = svd_base.trainset.global_mean
    scores_all = mu + bu_cold + _base_bi_eligible + _base_qi_eligible @ pu_cold

    for i_inner, qi in local_qi.items():
        pos = _inner_to_position.get(i_inner)
        if pos is not None:
            scores_all[pos] = mu + bu_cold + local_bi[i_inner] + np.dot(pu_cold, qi)

    shown_set = set(shown)
    shown_positions = np.array([
        _inner_to_position[item_to_iidx[iid]]
        for iid in shown_set
        if item_to_iidx.get(iid) is not None and item_to_iidx[iid] in _inner_to_position
    ], dtype=int)

    valid_mask = np.ones(len(scores_all), dtype=bool)
    if len(shown_positions) > 0:
        valid_mask[shown_positions] = False

    valid_scores = scores_all[valid_mask]
    valid_iids   = _eligible_iid_array[valid_mask]

    if len(valid_scores) == 0:
        return []
    b = min(batch_size, len(valid_scores))

    def top_b(arr, largest):
        sign = -1 if largest else 1
        idx = np.argpartition(sign * arr, b - 1)[:b]
        return idx[np.argsort(sign * arr[idx])]

    if strategy == 'SHHCP':
        idx = top_b(valid_scores, largest=True)
    elif strategy == 'SHLCP':
        idx = top_b(valid_scores, largest=False)
    elif strategy == 'SHMCP':
        median_val = np.median(valid_scores)
        dist = np.abs(valid_scores - median_val)
        idx = top_b(dist, largest=False)
    elif strategy == 'SHECP':
        epsilon = max(SHECP_FLOOR, SHECP_DECAY ** round_number)
        if egreedy_rng_local.random() < epsilon:
            idx = top_b(valid_scores, largest=False)   # explore: SHLCP logic
        else:
            idx = top_b(valid_scores, largest=True)    # exploit: SHHCP logic
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return valid_iids[idx].tolist()


def _stable_seed(u: int, shown: List[Any]) -> int:
    """Deterministic replacement for Python's built-in ``hash()`` on
    strings -- see ``_seeded_rng``'s docstring above for why this
    matters under multiprocessing.

    Parameters
    ----------
    u : int
        User index.
    shown : list
        Raw itemId values shown to this user (order-independent: sorted
        before hashing).

    Returns
    -------
    int
        A deterministic 32-bit seed derived from ``(u, shown)``.
    """
    key = f"{int(u)}|" + ','.join(sorted(str(x) for x in shown))
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def _split_unseen_items(
    eligible_items: List[Any], u: int, shown: List[Any], val_frac: float = 0.5
) -> Tuple[List[Any], List[Any]]:
    """Splits a cold user's remaining unseen eligible items into a
    validation half and a test half.

    The validation half is used by hyperparameter-tuning scripts
    elsewhere in this repo (e.g. decaying_lr_test.py); the test half is
    used here for final RMSE/HR@K/NDCG@K reporting. The split is
    deterministically seeded per ``(u, shown)`` via ``_stable_seed``, so
    re-running this function with the same arguments always produces
    the same partition, and the two halves never overlap.

    Parameters
    ----------
    eligible_items : list
        All raw itemId values eligible for selection.
    u : int
        User index.
    shown : list
        Raw itemId values already shown to this user this session
        (excluded from both halves).
    val_frac : float, default 0.5
        Fraction of unseen items assigned to the validation half.

    Returns
    -------
    val_items : list
        The validation-half item ids.
    test_items : list
        The test-half item ids (the remainder).
    """
    shown_set = set(shown)
    unseen = [iid for iid in eligible_items if iid not in shown_set]
    seed = _stable_seed(u, shown)
    rng = np.random.RandomState(seed)
    rng.shuffle(unseen)
    n_val = int(val_frac * len(unseen))
    return unseen[:n_val], unseen[n_val:]


def _sampled_metrics_at_k(
    pos_score: float, neg_scores: List[float], k_list: List[int]
) -> Dict[str, float]:
    """Computes sampled HR@K and NDCG@K for one positive item.

    Following He et al. (2017): the positive item is ranked against a
    fixed set of sampled negatives (``rank(i+) = 1 + count(neg_scores >
    pos_score)``, ties resolved in the positive item's favour), and
    HR@K / NDCG@K are then read off that rank (thesis Eq. 3.15-3.16).

    Parameters
    ----------
    pos_score : float
        Predicted score for the positive test item.
    neg_scores : list of float
        Predicted scores for the sampled negative items.
    k_list : list of int
        Cutoff values K to evaluate (e.g. [5, 10]).

    Returns
    -------
    dict
        Keys ``'HR@{k}'`` and ``'NDCG@{k}'`` for each ``k`` in
        ``k_list``.
    """
    rank = 1 + sum(1 for s in neg_scores if s > pos_score)
    results = {}
    for k in k_list:
        results[f'HR@{k}'] = 1.0 if rank <= k else 0.0
        results[f'NDCG@{k}'] = (1.0 / np.log2(rank + 1)) if rank <= k else 0.0
    return results


def process_one_user(work_item: Tuple[str, int, int]) -> Optional[Dict[str, Any]]:
    """Runs the full active-learning simulation for one (strategy, k,
    user) combination.

    Algorithm (thesis Section 3.7.3, Algorithm 1)
    -----------------------------------------------
    1. Initialise the cold user's factor vector at the most popular
       item's own base vector (a content-informed starting point, not
       the zero vector) and bias at 0.
    2. Show the most popular item first; observe the response; apply
       one partial-SGD update.
    3. While fewer than ``k`` items have been shown: select the next
       batch of ``BATCH_SIZE`` items via ``_select_batch`` (strategy-
       dependent), observe each response, apply a partial-SGD update
       per revealed interaction.
    4. Split the user's remaining unseen items into validation/test
       halves (deterministic, thesis 50/50 split); score RMSE on the
       test half, and HR@{5,10}/NDCG@{5,10} against ``N_NEG`` sampled
       negatives per positive test item.

    Parameters
    ----------
    work_item : tuple of (str, int, int)
        ``(strategy, k, u)`` -- the strategy name, elicitation budget,
        and cold user index to evaluate.

    Returns
    -------
    dict or None
        ``{'strategy', 'k', 'user', 'rmse', 'hr5', 'hr10', 'ndcg5',
        'ndcg10'}`` on success. Returns ``None`` if the user has no
        held-out test items (a structural consequence of dataset
        sparsity -- see thesis Section 4.3, "~43-46% dropout").

    Notes
    -----
    Reads the worker-local ``_cache`` global (populated by
    ``_worker_init``) -- this function must only be called as a
    ``multiprocessing.Pool`` worker target with that initializer, never
    directly from the main process.
    """
    strategy, k, u = work_item
    data             = _cache['data']
    eligible_items   = _cache['eligible_items']
    item_to_iidx     = _cache['item_to_iidx']
    most_popular_iid = _cache['most_popular_iid']
    i_0_inner        = _cache['i_0_inner']
    svd_base         = _cache['svd_base']
    mu_base          = _cache['mu_base']
    n_factors        = _cache['n_factors']
    GAMMA1, GAMMA2   = _cache['GAMMA1'], _cache['GAMMA2']
    LMBDA1, LMBDA2   = _cache['LMBDA1'], _cache['LMBDA2']
    mu_base_global[0] = mu_base

    alpha = _shrink_alpha(k)

    egreedy_rng_local = _seeded_rng(u, strategy, k, 'egreedy')
    neg_rng_local     = _seeded_rng(u, strategy, k, 'negsample')

    # Cold-start initialisation: the user's latent vector starts at the
    # most popular item's own vector (a content-informed starting point),
    # not the zero vector -- see README.md "Methodology".
    if i_0_inner is not None:
        pu_cold = svd_base.qi[i_0_inner].copy()
    else:
        pu_cold = np.zeros(n_factors)
    bu_cold  = 0.0
    local_qi, local_bi = {}, {}
    shown = [most_popular_iid]
    n_updates = 0

    first_row = data[(data['user_idx'] == u) & (data['itemId'] == most_popular_iid)]
    r_first   = float(first_row['interaction'].iloc[0]) if len(first_row) > 0 else 0.0
    if i_0_inner is not None:
        pu_cold, bu_cold = _partial_lfm_update_cold(
            svd_base, pu_cold, bu_cold, i_0_inner, r_first, local_qi, local_bi,
            GAMMA1, GAMMA2, LMBDA1, LMBDA2, num_sgd_steps=NUM_SGD_STEPS,
            update_index=n_updates
        )
        n_updates += 1

    round_number = 0
    while len(shown) < k:
        b     = min(BATCH_SIZE, k - len(shown))
        batch = _select_batch(svd_base, pu_cold, bu_cold, shown, eligible_items,
                               item_to_iidx, local_qi, local_bi, strategy, b,
                               round_number, egreedy_rng_local)
        if not batch:
            break
        shown.extend(batch)
        for item in batch:
            row  = data[(data['user_idx'] == u) & (data['itemId'] == item)]
            r_ui = float(row['interaction'].iloc[0]) if len(row) > 0 else 0.0
            i_inner = item_to_iidx.get(item)
            if i_inner is not None:
                pu_cold, bu_cold = _partial_lfm_update_cold(
                    svd_base, pu_cold, bu_cold, i_inner, r_ui, local_qi, local_bi,
                    GAMMA1, GAMMA2, LMBDA1, LMBDA2, num_sgd_steps=NUM_SGD_STEPS,
                    update_index=n_updates
                )
                n_updates += 1
        round_number += 1

    _, test_items = _split_unseen_items(eligible_items, u, shown, val_frac=0.5)
    shown_set = set(shown)
    test_df = data[(data['user_idx'] == u) &
                   (data['itemId'].isin(test_items)) &
                   (~data['itemId'].isin(shown_set))]
    if len(test_df) == 0:
        return None

    preds_manual = []
    for row in test_df.itertuples():
        i_inner = item_to_iidx.get(row.itemId)
        if i_inner is None:
            continue
        qi  = local_qi.get(i_inner, svd_base.qi[i_inner])
        bi  = local_bi.get(i_inner, svd_base.bi[i_inner])
        est = _score(bu_cold, bi, pu_cold, qi, alpha)
        preds_manual.append(Pred(uid=row.user_idx, iid=row.item_idx,
                                  r_ui=row.interaction, est=est, details={}))
    if not preds_manual:
        return None
    rmse = accuracy.rmse(preds_manual, verbose=False)

    user_interacted = set(data[data['user_idx'] == u]['itemId'].tolist())
    candidate_negs  = [iid for iid in eligible_items
                       if iid not in user_interacted and iid not in shown_set
                       and iid in item_to_iidx]
    pos_test_iids = test_df[test_df['interaction'] == 1]['itemId'].tolist()

    user_hr5, user_hr10, user_ndcg5, user_ndcg10 = [], [], [], []
    for pos_iid in pos_test_iids:
        i_pos = item_to_iidx.get(pos_iid)
        if i_pos is None:
            continue
        qi_pos = local_qi.get(i_pos, svd_base.qi[i_pos])
        bi_pos = local_bi.get(i_pos, svd_base.bi[i_pos])
        pos_score = _score(bu_cold, bi_pos, pu_cold, qi_pos, alpha)
        n_sample = min(N_NEG, len(candidate_negs))
        if n_sample == 0:
            continue
        sampled_neg_iids = neg_rng_local.choice(candidate_negs, size=n_sample, replace=False)
        neg_scores = []
        for nid in sampled_neg_iids:
            ni = item_to_iidx[nid]
            qi_n = local_qi.get(ni, svd_base.qi[ni])
            bi_n = local_bi.get(ni, svd_base.bi[ni])
            neg_scores.append(_score(bu_cold, bi_n, pu_cold, qi_n, alpha))
        m = _sampled_metrics_at_k(pos_score, neg_scores, k_list=[5, 10])
        user_hr5.append(m['HR@5']);     user_hr10.append(m['HR@10'])
        user_ndcg5.append(m['NDCG@5']); user_ndcg10.append(m['NDCG@10'])

    return {
        'strategy': strategy, 'k': k, 'user': int(u),
        'rmse': rmse,
        'hr5': np.mean(user_hr5) if user_hr5 else np.nan,
        'hr10': np.mean(user_hr10) if user_hr10 else np.nan,
        'ndcg5': np.mean(user_ndcg5) if user_ndcg5 else np.nan,
        'ndcg10': np.mean(user_ndcg10) if user_ndcg10 else np.nan,
    }


def run(
    n_workers: int, strategies: List[str], k_values: List[int], num_users: int,
    label: str
) -> Tuple[pd.DataFrame, float]:
    """Evaluates every (strategy, k, user) combination in parallel.

    Builds the full cross product of ``strategies x k_values x`` the
    first ``num_users`` cold users, then dispatches each combination to
    ``process_one_user`` across a ``multiprocessing.Pool`` of
    ``n_workers`` processes.

    Parameters
    ----------
    n_workers : int
        Number of worker processes.
    strategies : list of str
        Strategy names to evaluate, e.g. ``['SHHCP', 'SHLCP', 'SHMCP',
        'SHECP']``.
    k_values : list of int
        Elicitation budgets to evaluate, e.g. ``[10, 25, 50, 100]``.
    num_users : int
        Number of cold users to evaluate (the first ``num_users`` of
        ``cache['cold_users']``, in the fixed order set by
        build_model_cache.py's cold/warm split).
    label : str
        Human-readable label for progress printouts.

    Returns
    -------
    pd.DataFrame
        One row per (strategy, k, user) combination with a valid
        result (see ``process_one_user``'s return value); combinations
        with no held-out test items are silently omitted.
    elapsed : float
        Wall-clock seconds for the parallel evaluation (excludes cache
        loading in the main process, but each worker's own cache load
        via ``_worker_init`` is included in its share of the wall
        clock).

    Complexity
    ----------
    ``len(strategies) * len(k_values) * num_users`` work items,
    processed in ``chunksize=4`` batches by ``n_workers`` processes;
    each work item is O(k * F) (k partial-SGD updates plus O(F)
    scoring per candidate item, per active-learning round).
    """
    with open(MODEL_CACHE, 'rb') as f:
        cache = pickle.load(f)
    cold_users = cache['cold_users']
    eval_users = cold_users[:num_users]

    work_items = [(s, k, u) for s in strategies for k in k_values for u in eval_users]
    print(f"\n=== {label}: {len(work_items)} work items, {n_workers} workers ===", flush=True)
    print(f"    CONFIG: USE_DECAYING_LR={USE_DECAYING_LR}  SHRINKAGE_C={SHRINKAGE_C}",
          flush=True)

    t0 = time.time()
    with mp.Pool(processes=n_workers, initializer=_worker_init) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(process_one_user, work_items, chunksize=4)):
            if r is not None:
                results.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(work_items)} work items done "
                      f"({time.time()-t0:.1f}s elapsed)", flush=True)
    elapsed = time.time() - t0
    print(f"  {label} done: {elapsed:.1f}s ({elapsed/60:.1f} min) for "
          f"{len(work_items)} work items", flush=True)
    return pd.DataFrame(results), elapsed


def main() -> None:
    """Runs a 20-user smoke test as a quick correctness check.

    For the full 1,000-user x 4-strategy x 4-k evaluation used to
    produce the thesis's Chapter 4 results, use
    ``scripts/model/run_complete_pipeline.py`` instead (see module
    docstring), which calls ``run()`` directly at full scale.

    Raises
    ------
    RuntimeError
        If the smoke test produces no valid (non-NaN RMSE) results,
        indicating a correctness problem that should be fixed before
        attempting the full-scale run.
    """
    mp.set_start_method('spawn', force=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df_smoke, t_smoke = run(
        n_workers=4, strategies=['SHLCP'], k_values=[10], num_users=20,
        label='SMOKE TEST (correctness check)'
    )
    print(f"\nSmoke test results:\n{df_smoke.describe()}", flush=True)
    if df_smoke.empty or df_smoke['rmse'].isna().all():
        raise RuntimeError("Smoke test produced no valid results -- fix before scaling up.")
    print("\nSmoke test passed. Run scripts/model/run_complete_pipeline.py for "
          "the full evaluation.", flush=True)


if __name__ == '__main__':
    main()
