"""
personalised_strategies.py
============================
Implements and evaluates four confidence-based personalised active
learning strategies for the cold-user problem: SHHCP, SHLCP, SHMCP,
and SHECP. Each strategy selects which items to show a new user next,
applying an incremental update to that user's own latent factors after
every response.

Key mechanisms:
    - Confidence-weighted shrinkage toward the item-level baseline
      score (SHRINKAGE_C).
    - SHECP's epsilon-greedy explore/exploit schedule (SHECP_FLOOR,
      SHECP_DECAY).
    - An update is skipped for any shown item with no recorded
      interaction for that user, rather than treating a missing entry
      as a dislike.
    - Cold-start initialisation uses the most popular item's factor
      vector, except at k=100 where the zero vector performs better
      (ZERO_INIT_K_VALUES).

Uses a per-user local copy of each touched item's vector/bias so that
updates for one cold user never affect another user's evaluation; the
shared base model stays frozen throughout.

Runs in parallel via multiprocessing.Pool, one worker process per CPU
core requested.

Usage
-----
Run directly for a 20-user smoke test:

    python scripts/model/personalised_strategies.py

For the full evaluation, use run_complete_pipeline.py:

    python scripts/model/run_complete_pipeline.py

Input
-----
    results/base_model_cache.pkl   (see build_model_cache.py)

Output
------
    results/personalised_results.csv
        Columns: strategy, k, user, rmse, hr5, hr10, ndcg5, ndcg10
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
# evaluation (He et al., 2017 methodology).
N_NEG: int = 99

# Number of partial-SGD steps applied per revealed interaction. More
# than 1 lets the model overfit each individual, noisy binary response.
NUM_SGD_STEPS: int = 1

# SHECP's epsilon-greedy exploration floor/decay:
# epsilon_r = max(SHECP_FLOOR, SHECP_DECAY ** r).
SHECP_FLOOR: float = 0.05
SHECP_DECAY: float = 0.95

# Number of items revealed per active-learning round before the next
# selection is made.
BATCH_SIZE: int = 3

SHRINKAGE_C: Optional[int] = 100

# Regularisation coefficients for the cold-user partial update.
LMBDA1: float = 1e-7
LMBDA2: float = 1e-6

# k values that use zero-vector cold-start initialisation instead of
# the item-based default.
ZERO_INIT_K_VALUES: frozenset = frozenset({100})

Pred = namedtuple('Prediction', ['uid', 'iid', 'r_ui', 'est', 'details'])

# Populated once per worker process by _worker_init, then only read
# (never mutated) by every subsequent call in that worker -- this is
# what makes plain module-level globals safe here.
_cache: Optional[Dict[str, Any]] = None
_eligible_inner_indices: Optional[np.ndarray] = None
_eligible_iid_array: Optional[np.ndarray] = None
_inner_to_position: Optional[Dict[int, int]] = None
_base_qi_eligible: Optional[np.ndarray] = None
_base_bi_eligible: Optional[np.ndarray] = None
_user_item_interaction: Optional[Dict[int, Dict[Any, float]]] = None


def _worker_init(eval_users: Optional[List[Any]] = None) -> None:
    """Runs once per worker process when the Pool starts.

    Loads the model cache, pre-computes the eligible-item latent-
    vector/bias arrays used by ``_select_batch``, and builds a
    per-user interaction lookup dict so ``process_one_user`` never has
    to filter the full interaction table.

    Parameters
    ----------
    eval_users : list, optional
        Cold user indices this run will evaluate. When given, the
        lookup dict is built only from these users' rows instead of
        the whole dataset.
    """
    global _cache, _eligible_inner_indices, _eligible_iid_array
    global _inner_to_position, _base_qi_eligible, _base_bi_eligible
    global _user_item_interaction
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

    data = _cache['data']
    if eval_users is not None:
        eval_users_set = set(int(u) for u in eval_users)
        user_data = data[data['user_idx'].isin(eval_users_set)]
    else:
        user_data = data
    _user_item_interaction = {
        int(u): dict(zip(g['itemId'].values, g['interaction'].values))
        for u, g in user_data.groupby('user_idx')
    }


def _seeded_rng(*parts: Any) -> np.random.RandomState:
    """Builds a deterministic RNG seeded from ``parts``.

    Uses hashlib.md5 instead of Python's built-in ``hash()``, which is
    randomised per-process and would otherwise give a different draw
    on every run under multiprocessing.

    Parameters
    ----------
    *parts : Any
        Values identifying the work item; stringified and joined to
        form the hash key.

    Returns
    -------
    np.random.RandomState
        A generator seeded deterministically from ``parts``.
    """
    key = '|'.join(str(p) for p in parts)
    seed = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)
    return np.random.RandomState(seed)


def _partial_lfm_update_cold(
    svd_model: SVD, pu_cold: np.ndarray, bu_cold: float, i_inner: int, r_ui: float,
    local_qi: Dict[int, np.ndarray], local_bi: Dict[int, float],
    gamma1: float, gamma2: float, lmbda1: float, lmbda2: float,
    num_sgd_steps: int = 1
) -> Tuple[np.ndarray, float]:
    """Applies one (or ``num_sgd_steps``) partial-SGD update(s) to a cold
    user's own parameters given a single newly-observed interaction.

    Updates only the cold user's factor vector/bias and the local copy
    of the just-shown item's vector/bias (created on first touch as a
    copy of the frozen base model's values). Every other parameter --
    every other item, every warm user -- is untouched, which is what
    keeps this update cheap regardless of dataset size.

    Parameters
    ----------
    svd_model : surprise.SVD
        The frozen base model.
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
        Number of SGD steps to apply for this interaction.

    Returns
    -------
    pu_cold : np.ndarray, shape (F,)
        The updated cold-user factor vector.
    bu_cold : float
        The updated cold-user bias.
    """
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

    ``alpha(k) = k / (k + SHRINKAGE_C)``. As ``k`` grows, alpha -> 1
    and the personalised term dominates the prediction; as k -> 0,
    alpha -> 0 and the prediction reduces to the item-level baseline
    ``mu + b_i``. If ``SHRINKAGE_C`` is None, alpha is always 1 (no
    shrinkage).

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


# Base-model mean, set once per work item in process_one_user.
# List-boxed so _score can read the current value without a `global`
# statement.
mu_base_global: List[Optional[float]] = [None]


def _score(bu_cold: float, bi: float, pu_cold: np.ndarray, qi: np.ndarray,
           alpha: float) -> float:
    """Computes the shrinkage-weighted predicted interaction score.

    ``score = clip(mu + b_i + alpha * (b_u^c + p_u^c . q_i), 0, 1)``

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
        Predicted score, clipped to [0, 1].
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

    Scores every not-yet-shown eligible item under the raw (non-shrunk)
    prediction, then picks ``batch_size`` of them according to
    ``strategy``:

    - SHHCP: the highest-scoring items (pure exploitation).
    - SHLCP: the lowest-scoring items (pure exploration).
    - SHMCP: the items closest to the median score (uncertainty
      sampling).
    - SHECP: with probability ``epsilon_r = max(SHECP_FLOOR,
      SHECP_DECAY ** round_number)``, behaves like SHLCP (explore);
      otherwise like SHHCP (exploit).

    Shrinkage is applied only at final scoring, never during item
    selection here, since alpha(k) depends on total session length,
    not per-round state.

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
        0-indexed active-learning round; used only by SHECP's epsilon
        schedule.
    egreedy_rng_local : np.random.RandomState
        Per-work-item RNG for SHECP's explore/exploit coin flip.

    Returns
    -------
    list
        Up to ``batch_size`` raw itemId values, selected according to
        ``strategy``. May return fewer (or an empty list) if fewer
        eligible unseen items remain.

    Raises
    ------
    RuntimeError
        If called before ``_worker_init``.
    ValueError
        If ``strategy`` is not one of the four recognised strategies.
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
    strings -- see ``_seeded_rng`` for why this matters under
    multiprocessing.

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

    The split is deterministically seeded per ``(u, shown)`` via
    ``_stable_seed``, so re-running this function with the same
    arguments always produces the same partition, and the two halves
    never overlap.

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
    HR@K / NDCG@K are then read off that rank.

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

    Algorithm
    ---------
    1. Initialise the cold user's factor vector at the most popular
       item's own base vector (or the zero vector, see
       ZERO_INIT_K_VALUES) and bias at 0.
    2. Show the most popular item first; observe the response; apply
       one partial-SGD update.
    3. While fewer than ``k`` items have been shown: select the next
       batch of ``BATCH_SIZE`` items via ``_select_batch``, observe
       each response, apply a partial-SGD update per revealed
       interaction.
    4. Split the user's remaining unseen items into validation/test
       halves; score RMSE on the test half, and HR@{5,10}/NDCG@{5,10}
       against ``N_NEG`` sampled negatives per positive test item.

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
        held-out test items.

    Notes
    -----
    Reads the worker-local ``_cache`` global (populated by
    ``_worker_init``) -- this function must only be called as a
    ``multiprocessing.Pool`` worker target with that initializer.
    """
    strategy, k, u = work_item
    user_dict        = _user_item_interaction.get(int(u), {})
    eligible_items   = _cache['eligible_items']
    item_to_iidx     = _cache['item_to_iidx']
    most_popular_iid = _cache['most_popular_iid']
    i_0_inner        = _cache['i_0_inner']
    svd_base         = _cache['svd_base']
    mu_base          = _cache['mu_base']
    n_factors        = _cache['n_factors']
    GAMMA1, GAMMA2   = _cache['GAMMA1'], _cache['GAMMA2']
    # LMBDA1/LMBDA2 use this module's own constants above, not the
    # cache's.
    mu_base_global[0] = mu_base

    alpha = _shrink_alpha(k)

    egreedy_rng_local = _seeded_rng(u, strategy, k, 'egreedy')
    neg_rng_local     = _seeded_rng(u, strategy, k, 'negsample')

    # Item-based init for every k except ZERO_INIT_K_VALUES, which use
    # the zero vector instead.
    if k in ZERO_INIT_K_VALUES:
        pu_cold = np.zeros(n_factors)
    elif i_0_inner is not None:
        pu_cold = svd_base.qi[i_0_inner].copy()
    else:
        pu_cold = np.zeros(n_factors)
    bu_cold  = 0.0
    local_qi, local_bi = {}, {}
    shown = [most_popular_iid]

    # Skip the update entirely if this user has no recorded interaction
    # for the item -- it's still added to `shown`, so k is unaffected.
    has_first = most_popular_iid in user_dict
    r_first   = float(user_dict[most_popular_iid]) if has_first else 0.0
    if i_0_inner is not None and has_first:
        pu_cold, bu_cold = _partial_lfm_update_cold(
            svd_base, pu_cold, bu_cold, i_0_inner, r_first, local_qi, local_bi,
            GAMMA1, GAMMA2, LMBDA1, LMBDA2, num_sgd_steps=NUM_SGD_STEPS
        )

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
            has_row = item in user_dict
            r_ui = float(user_dict[item]) if has_row else 0.0
            i_inner = item_to_iidx.get(item)
            if i_inner is not None and has_row:
                pu_cold, bu_cold = _partial_lfm_update_cold(
                    svd_base, pu_cold, bu_cold, i_inner, r_ui, local_qi, local_bi,
                    GAMMA1, GAMMA2, LMBDA1, LMBDA2, num_sgd_steps=NUM_SGD_STEPS
                )
        round_number += 1

    _, test_items = _split_unseen_items(eligible_items, u, shown, val_frac=0.5)
    shown_set = set(shown)
    # test_items already excludes shown items by construction above.
    test_items_set = set(test_items)
    test_rows = [(iid, r) for iid, r in user_dict.items() if iid in test_items_set]
    if not test_rows:
        return None

    preds_manual = []
    for iid, r in test_rows:
        i_inner = item_to_iidx.get(iid)
        if i_inner is None:
            continue
        qi  = local_qi.get(i_inner, svd_base.qi[i_inner])
        bi  = local_bi.get(i_inner, svd_base.bi[i_inner])
        est = _score(bu_cold, bi, pu_cold, qi, alpha)
        preds_manual.append(Pred(uid=u, iid=iid, r_ui=r, est=est, details={}))
    if not preds_manual:
        return None
    rmse = accuracy.rmse(preds_manual, verbose=False)

    user_interacted = set(user_dict.keys())
    candidate_negs  = [iid for iid in eligible_items
                       if iid not in user_interacted and iid not in shown_set
                       and iid in item_to_iidx]
    pos_test_iids = [iid for iid, r in test_rows if r == 1]

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
        ``cache['cold_users']``).
    label : str
        Human-readable label for progress printouts.

    Returns
    -------
    pd.DataFrame
        One row per (strategy, k, user) combination with a valid
        result; combinations with no held-out test items are silently
        omitted.
    elapsed : float
        Wall-clock seconds for the parallel evaluation.
    """
    with open(MODEL_CACHE, 'rb') as f:
        cache = pickle.load(f)
    cold_users = cache['cold_users']
    eval_users = cold_users[:num_users]

    work_items = [(s, k, u) for s in strategies for k in k_values for u in eval_users]
    print(f"\n=== {label}: {len(work_items)} work items, {n_workers} workers ===", flush=True)
    print(f"    CONFIG: SHRINKAGE_C={SHRINKAGE_C}", flush=True)

    t0 = time.time()
    with mp.Pool(processes=n_workers, initializer=_worker_init,
                  initargs=(eval_users,)) as pool:
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

    For the full evaluation, use ``scripts/model/run_complete_pipeline.py``
    instead, which calls ``run()`` directly at full scale.

    Raises
    ------
    RuntimeError
        If the smoke test produces no valid results.
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
