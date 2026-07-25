# Code Review — SHP (Single Heuristic Personalized) Recommender

Scope: everything tracked in git (i.e. the exact code published to
`github.com/yytsai171/Single-Heuristic-Personalized-SHP`), with primary
focus on `scripts/model/` (the four files that make up the actual
model + pipeline: `build_model_cache.py`, `baseline_ranking_metrics.py`,
`personalised_strategies.py`, `run_complete_pipeline.py`), and a
lighter pass over `scripts/experiments/` and `scripts/plotting/`.

This review does not re-run the code; findings are from static reading
of the source (`data/useritemmatrix.csv` was confirmed to be ~2.56M
rows, which matters for finding #1 below).

Findings are ordered roughly by impact: a leading "critical finding"
section (added after a follow-up question about *why* personalised
underperforms baselines empirically), then performance, then
correctness/risk, then structure/maintainability, then style and
nice-to-haves. A "what's already solid" section closes it out — the
reproducibility engineering here is genuinely above the academic-code
norm and worth naming explicitly.

**Status update**: finding #0 and finding #1.1 are now resolved in the
current code — see the note at the top of each section below. Every
other finding in this document is still open as originally written.

---

## 0. Critical finding: missing interaction data silently becomes a fabricated "dislike" during elicitation

**RESOLVED.** `process_one_user` now skips the partial-SGD update
entirely whenever there's no recorded interaction for a shown item,
instead of defaulting to 0.0. The rest of this section is kept as the
original finding, for reference.

This is the most consequential finding in this review, and a strong
candidate for (part of) the mechanical explanation behind the paper's
central result — that personalised strategies fail to beat the
non-personalised baselines on this dataset.

**Where**: [scripts/model/personalised_strategies.py:634](scripts/model/personalised_strategies.py:634) and
[scripts/model/personalised_strategies.py:654](scripts/model/personalised_strategies.py:654):

```python
first_row = data[(data['user_idx'] == u) & (data['itemId'] == most_popular_iid)]
r_first   = float(first_row['interaction'].iloc[0]) if len(first_row) > 0 else 0.0
...
row  = data[(data['user_idx'] == u) & (data['itemId'] == item)]
r_ui = float(row['interaction'].iloc[0]) if len(row) > 0 else 0.0
```

Whenever an active-learning strategy selects an item to "show" a cold
user, and that `(user, item)` pair has **no row at all** in
`data/useritemmatrix.csv` (i.e. we have zero ground truth about this
user's response to this item), the code falls back to `r_ui = 0.0` —
a *confirmed dislike* — and feeds it directly into
`_partial_lfm_update_cold`. This conflates two semantically different
things that the dataset itself distinguishes (per the README's own
"Dataset" section: `interaction=0` means "returned or not purchased,"
an actually-observed event; a missing row means "never observed," no
information at all).

**Why this matters at the scale of this dataset**: I checked against
the locally-cached trained model (`results/base_model_cache.pkl`,
present but gitignored) rather than guessing:

| Quantity | Value |
|---|---|
| Eligible items | 45,543 |
| Median / mean recorded interactions per cold user | 2 / 4.5 |
| Cold users with a recorded row for *their forced first shown item* (`most_popular_iid`) | 5.84% |
| Cold users with a recorded row among the top-100 most popular eligible items | 13.07% |
| Expected overlap between a uniformly-drawn 100-item batch and a user's real history | ~0.008 items |

In other words: for roughly **94% of cold users, the very first
"elicited response" in their session is fabricated**, and for
essentially every subsequent shown item across the whole session, the
"observed" label fed into the SGD update is manufactured rather than
real. The genuinely-observed case is the rare exception, not the norm.

**Why this specifically penalises the personalised strategies and not
the baselines**: `random`/`popularity`/`poperror`
([scripts/model/baseline_ranking_metrics.py](scripts/model/baseline_ranking_metrics.py))
never run any per-user update loop — they always predict the static
`mu + b_i`, so they are structurally immune to this issue. Only the
four personalised strategies run `_partial_lfm_update_cold` against
this contaminated signal. And the damage should scale *with* the
elicitation budget: larger `k` means more shown items, means more
accumulated fabricated-negative updates pulling `pu_cold`/`bu_cold`
toward "dislikes everything" — which would show up exactly as
"personalised gets *relatively worse* as `k` grows," if that pattern
appears in the results.

To be precise about what this does *not* affect: the held-out test
labels used for RMSE/HR@K/NDCG@K scoring are unaffected — `test_df` is
built by filtering `data` down to rows that already exist, so the
metrics themselves are always computed against real labels. This bug
corrupts the **training signal** consumed during the simulated
elicitation session, not the evaluation labels — but a corrupted
training signal is exactly what would produce a worse-performing
trained cold-user vector, which then legitimately scores worse on the
(real) held-out data.

**What to check next**: whether this "missing = negative" convention
was an explicit, justified design decision in the thesis's methodology
chapter (treating unobserved items as implicit negatives is a common,
if debated, convention in implicit-feedback recommender research, and
it's possible this was a deliberate, discussed choice). If it was not
explicitly intended for the *elicitation/update* step specifically
(as opposed to, say, training the base warm-user model, where the
convention is more standard), the fix is to make an elicitation step a
no-op when no ground truth exists for the sampled item (skip the
update, or exclude such items from `_select_batch`'s candidate pool
entirely) rather than injecting a fabricated label.

---

## 1. Performance

### 1.1 Per-item pandas boolean-mask scans dominate the ~80-85 min personalised run (High impact)

**RESOLVED.** `_worker_init` now builds a per-user interaction lookup
dict once per worker, and every lookup in `process_one_user` is an
O(1) dict `.get()`. Verified byte-identical output against the prior
pandas-scan version before trusting the fix. The rest of this section
is kept as the original finding, for reference.

`process_one_user` in
[scripts/model/personalised_strategies.py](scripts/model/personalised_strategies.py:562)
looks up each revealed interaction's ground-truth label with:

```python
row = data[(data['user_idx'] == u) & (data['itemId'] == item)]
```

(lines 633 and 653), and later:

```python
test_df = data[(data['user_idx'] == u) & (data['itemId'].isin(test_items)) & ...]
user_interacted = set(data[data['user_idx'] == u]['itemId'].tolist())
```

(lines 667, 687). `data` is the *full* ~2.56M-row interaction table —
not a per-user slice — so every one of these lines is a full linear
scan. Per work item (one `(strategy, k, user)` combination), this runs
roughly once per revealed item (up to `k`, so up to 100 for the
`k=100` cell) plus a couple more for the test split. Across the full
evaluation (1,000 users × 4 strategies × 4 k-values = 16,000 work
items), that's on the order of several hundred thousand full-table
scans — almost certainly the dominant cost behind the measured
~80-85 minute runtime.

**Fix**: build a per-user lookup once, in `_worker_init`
([scripts/model/personalised_strategies.py:123](scripts/model/personalised_strategies.py:123)),
e.g.:

```python
_user_item_interaction: Optional[Dict[int, Dict[Any, float]]] = None
...
_user_item_interaction = {
    u: dict(zip(g['itemId'], g['interaction']))
    for u, g in _cache['data'].groupby('user_idx')
}
```

Then every lookup in `process_one_user` becomes an O(1) dict `.get()`
instead of an O(n) mask. This is a pure win — the groupby runs once
per worker process instead of hundreds of thousands of times — and
should not change any result, only wall-clock time.

### 1.2 Baseline per-user metrics are independently recomputed three times (High impact, also a DRY problem)

`baseline_ranking_metrics.py` computes per-user RMSE/HR@K/NDCG@K for
the three non-personalised baselines and only saves the **aggregated**
mean per `(strategy, k)` cell to `results/baseline_results.csv`
([scripts/model/baseline_ranking_metrics.py:256](scripts/model/baseline_ranking_metrics.py:256)).

Because the per-user numbers aren't persisted,
`scripts/experiments/significance_test.py` recomputes the *identical*
per-user baseline RMSE/HR@10 from scratch
([scripts/experiments/significance_test.py:148](scripts/experiments/significance_test.py:148)),
and `scripts/experiments/ranking_significance_and_correction.py`
recomputes the same per-user baseline predictions a **third** time,
this time for NDCG@10
([scripts/experiments/ranking_significance_and_correction.py:186](scripts/experiments/ranking_significance_and_correction.py:186)).

All three copies use the same `_stable_seed`, `split_unseen_items`,
`select_items`/`select_items_nonpersonalised`, `baseline_predict`, and
PopError-score-construction logic, copy-pasted near-verbatim. This
costs real time (baseline metrics ≈8 min, significance test ≈16 min,
ranking significance ≈16 min — a large fraction of that ≈40 min is the
same computation three times) and is a maintainability risk: if you
ever tweak the split logic, the negative-sampling scheme, or PopError's
formula in one file, the other two silently keep the old behavior
unless updated in lockstep. There's no test or assertion anywhere that
would catch that drift.

**Fix**: have `baseline_ranking_metrics.py` also write a per-user
`results/baseline_per_user_results.csv` (or extend the existing pickle
cache) with `strategy, k, user, rmse, hr5, hr10, ndcg5, ndcg10`, and
have both significance scripts *read* that instead of recomputing it.
This eliminates ~30 minutes from the full pipeline and removes two of
the three duplicated implementations.

### 1.3 `build_model_cache.py`'s cold/warm filter is a no-op (Low impact, but worth removing)

```python
data = data.groupby('userId').filter(lambda x: len(x) > 0)
```

([scripts/model/build_model_cache.py:166](scripts/model/build_model_cache.py:166))

A `groupby` only ever produces groups for `userId` values that appear
in the dataframe, so every group already has `len(x) >= 1` by
construction — this line can never drop anything. The docstring above
`load_and_split_data` says it "drops any user with zero recorded
interactions," but such a user could never have a row in `data` to
begin with, so the described case is unreachable. As written, this is
a full `groupby.filter` pass (non-trivial on 2.56M rows) that does
nothing. Either remove it, or — if the real intent was something like
"drop users below some interaction-count threshold" — implement that
explicitly with a `value_counts()`/threshold check, since that's not
what the current code does.

---

## 2. Correctness / risk

### 2.1 Shrinkage weight uses the *target* budget `k`, not the number of interactions actually elicited (Worth double-checking intent)

`alpha = _shrink_alpha(k)` is computed once, before the elicitation
loop runs
([scripts/model/personalised_strategies.py:616](scripts/model/personalised_strategies.py:616)),
and `_shrink_alpha`'s docstring frames `k` as "elicitation budget"
rather than "items actually observed"
([scripts/model/personalised_strategies.py:275](scripts/model/personalised_strategies.py:275)).
In the loop:

```python
while len(shown) < k:
    ...
    if not batch:
        break
```

if the eligible-item pool runs dry before `len(shown)` reaches `k`
(plausible for a cold user who has already interacted with a large
fraction of eligible items, or near the tail of the ranking budgets),
the actual number of interactions folded into `pu_cold`/`bu_cold` is
less than `k`, yet the shrinkage weight still treats the estimate as
if backed by the full nominal budget — overstating confidence in the
personalised term relative to the evidence actually gathered.

This is very likely intentional given the docstring's explicit framing
of `k` as "the experimental condition" rather than "observed count,"
and the early-break is presumably rare in practice given the eligible
pool size. But it's subtle enough that a future maintainer (or a
careful reviewer of the thesis) could reasonably read it as a bug.
Worth either an explicit comment noting this is deliberate ("shrinkage
is keyed to the nominal budget k, not actual observations — see
thesis Eq. X"), or switching to `_shrink_alpha(len(shown))` if that
was in fact the intended definition.

### 2.2 `mu_base_global` list-boxed mutable global relies on single-task-per-worker execution (Low risk, already implicitly safe, but fragile if refactored)

[scripts/model/personalised_strategies.py:304](scripts/model/personalised_strategies.py:304)
uses a module-level `mu_base_global = [None]`, mutated once per work
item at the top of `process_one_user`
([scripts/model/personalised_strategies.py:614](scripts/model/personalised_strategies.py:614))
and read by `_score`. This is safe today because `multiprocessing.Pool`
with `imap_unordered` runs one task at a time per worker process — but
it's a global mutated as an implicit side channel between two
functions that could otherwise just take `mu_base` as a parameter.
If anyone ever changes the parallelism model (e.g. threads instead of
processes, or async workers), this becomes a real race condition with
no error message, just silently wrong scores. Threading it through as
an explicit parameter to `_score`/`_select_batch` would cost nothing
and remove the hazard permanently.

### 2.3 Loading `results/base_model_cache.pkl` via `pickle.load` (Low risk given current usage, worth a one-line README note)

Every downstream script unpickles `base_model_cache.pkl` (e.g.
[scripts/model/personalised_strategies.py:140](scripts/model/personalised_strategies.py:140)).
Pickle deserialization executes arbitrary code if the file is
attacker-controlled. Given the README's instructions, this file is
always generated locally by the user's own `build_model_cache.py` run,
so the practical risk today is essentially zero. The only scenario
worth guarding against: someone shares a "pre-built cache" out-of-band
(e.g. to skip the 20-25 min setup) and a user loads a `.pkl` they
didn't generate themselves. A one-line README caution ("never load a
`base_model_cache.pkl` you didn't generate yourself locally") would
close that gap cheaply.

---

## 3. Structure / maintainability

### 3.1 No shared module for logic duplicated across 3-4 scripts

Beyond the baseline-recomputation issue in §1.2, the following are
copy-pasted (not imported) across `baseline_ranking_metrics.py`,
`personalised_strategies.py`, `significance_test.py`, and
`ranking_significance_and_correction.py`:

- `_stable_seed(u, shown)` — identical in all four files.
- The unseen-item validation/test split logic (`_split_unseen_items` /
  `split_unseen_items`) — identical algorithm, four separate
  definitions.
- `_sampled_metrics_at_k` / `sampled_metrics_at_k` /
  `sampled_hr_at_k` — near-identical HR@K/NDCG@K-from-rank logic,
  three variations that could easily drift (e.g. one only returns
  HR, missing NDCG, if a future edit to one copy isn't mirrored).
- PopError score construction (`ALPHA * log10(freq) + (1-ALPHA) *
  misclass_error`) — computed fresh in `build_model_cache.py`,
  `baseline_ranking_metrics.py`, `significance_test.py`, *and*
  `ranking_significance_and_correction.py`.

A small `scripts/model/shared.py` (or `scripts/common/eval_utils.py`)
exporting `stable_seed`, `split_unseen_items`, `sampled_metrics_at_k`,
and `compute_poperror_scores` would let all four/five scripts import
one implementation. Given how central reproducibility is to this
project's stated contribution, having the seeding/splitting logic
exist in exactly one place (rather than four copies that must be kept
byte-for-byte identical by hand) directly serves that goal.

### 3.2 Output column-naming is inconsistent across the two "results" CSVs

`baseline_results.csv` uses `Strategy`, `ItemsShown`,
`HR@5(sampled)` (PascalCase / spaced / symbol-laden column names —
[scripts/model/baseline_ranking_metrics.py:256](scripts/model/baseline_ranking_metrics.py:256)),
while `personalised_results.csv` uses `strategy`, `k`, `hr5`
(lowercase, terse —
[scripts/model/personalised_strategies.py:715](scripts/model/personalised_strategies.py:715)).
This is cosmetic, but it's part of why the two result sets can't be
concatenated or joined directly and why the significance scripts had
to recompute the baseline numbers in a different, code-matching shape
(reinforcing §1.2/§3.1). Aligning the two schemas (or having one
script's output be the direct input format for the other) would make
the whole pipeline more composable.

### 3.3 No automated tests beyond the embedded 20-user "smoke test"

`personalised_strategies.py main()` runs a 20-user SHLCP smoke test as
a basic correctness gate
([scripts/model/personalised_strategies.py:796](scripts/model/personalised_strategies.py:796)),
and `ranking_significance_and_correction.py` has a nice small
self-check for the Holm/BH implementations
([scripts/experiments/ranking_significance_and_correction.py:123](scripts/experiments/ranking_significance_and_correction.py:123)).
Beyond that, there's no `pytest`/`unittest` suite. For a repo whose
central claim rests on getting several fiddly numerical details right
(the shrinkage formula, the decaying learning rate, the deterministic
seeding, the rank-based HR/NDCG computation), a handful of fast unit
tests would pay for themselves:

- `_partial_lfm_update_cold` reduces squared error on a toy 2-factor
  example after N steps.
- `_shrink_alpha(0) == 0`, `_shrink_alpha(SHRINKAGE_C) == 0.5`,
  monotonically increasing in `k`.
- `_stable_seed`/`_seeded_rng` are deterministic across two calls
  with the same arguments (guards against someone reintroducing
  Python's built-in `hash()` by accident).
- `holm_correction`/`bh_fdr_correction` against a couple of hand-
  computed textbook examples (the current sanity check only verifies
  invariants, not exact values against a known-correct case).

These would run in well under a second (no need to touch the 92MB
dataset) and would catch regressions the smoke test can't (the smoke
test only checks that *some* non-NaN RMSE comes out, not that the
underlying formulas are right).

---

## 4. Style / consistency nits (low priority)

- **Mixed typing style**: files mix `from __future__ import
  annotations` + PEP 585 lowercase generics (`tuple[pd.DataFrame,
  ...]` in `build_model_cache.py`'s `load_and_split_data` signature)
  with `typing.List`/`typing.Dict` elsewhere in the same file. Since
  the project targets Python 3.11 exclusively (per README), it's safe
  to standardize on the lowercase builtin generics everywhere and drop
  the `typing` import for `List`/`Dict`/`Tuple`.
- **Magic numbers without an inline rationale**: e.g. the `len(merged)
  < 10` minimum-paired-sample cutoff in
  [scripts/experiments/significance_test.py:210](scripts/experiments/significance_test.py:210)
  and
  [scripts/experiments/ranking_significance_and_correction.py:307](scripts/experiments/ranking_significance_and_correction.py:307)
  isn't explained (presumably a reasonable floor for a Wilcoxon test
  to be meaningful) — a one-line comment referencing why 10 was chosen
  would help a reader who wants to change it.
- **All configuration is module-level constants, no CLI args**: every
  script hardcodes its `K_VALUES`, `NUM_EVAL_USERS`, `N_NEG`, etc. as
  top-of-file constants rather than `argparse` options. This is a
  defensible choice for an exact-reproducibility-focused thesis repo
  (fewer moving parts = fewer ways to accidentally reproduce the wrong
  numbers), so this is listed as a nice-to-have, not a defect — but if
  this codebase is extended beyond the thesis, promoting the constants
  most likely to be swept (`K_VALUES`, `NUM_EVAL_USERS`, `SHRINKAGE_C`,
  `SHECP_FLOOR`/`DECAY`) to optional CLI flags (defaulting to today's
  values) would make ad hoc experimentation much faster than editing
  source and re-running.
- **Plain `print(..., flush=True)` throughout, no logging module**:
  fine for single-run scripts invoked from the CLI as documented, but
  if these scripts are ever imported and run programmatically (as
  `run_complete_pipeline.py` already does for
  `personalised_strategies.run()`), a `logging`-based approach would
  let a caller redirect/silence output without touching the source.

---

## 5. Nice-to-haves / possible follow-up work

- **Persist `error_scores`/`poperror_scores` in the model cache**
  rather than recomputing them from `warm_data` in four separate
  places (ties into §1.2/§3.1) — they're deterministic given
  `warm_data` and `ALPHA`, both already in the cache.
- **Vectorize `compute_misclassification_error_scores`**
  ([scripts/model/build_model_cache.py:183](scripts/model/build_model_cache.py:183)):
  the per-item Python loop over `eligible_items` calling
  `.get(item, 0.5)` on a Series is fine at current scale but could be
  a single vectorized `np.minimum(p, 1-p)` over
  `item_mean_interaction.reindex(eligible_items, fill_value=0.5)`.
  Minor, since this only runs once in `build_model_cache.py`, but the
  same pattern is repeated (non-vectorized) in three other files for
  the identical computation.
  scale.
- **`run_complete_pipeline.py`'s stage-3 special-casing** (it imports
  `personalised_strategies` directly instead of using `run_script`
  like every other stage,
  [scripts/model/run_complete_pipeline.py:125](scripts/model/run_complete_pipeline.py:125))
  is explained clearly in the docstring and is a reasonable trade-off,
  not a defect — flagging only because it's the one asymmetry in an
  otherwise uniform "each stage is a subprocess" design, worth a reader
  noticing on first pass.
- **`data/useritemmatrix.csv` at ~92MB committed directly to git**
  (not LFS) — the README already flags this as a consideration for
  forks; no action needed for the current repo, just confirming the
  existing caveat is accurate and worth keeping if the file ever grows.

---

## 6. What's already solid (worth keeping as-is)

To be clear about the baseline this review is holding the code to —
several choices here are better than what's typical in academic
research code, and should not be "improved" away:

- **Deterministic seeding discipline** is unusually rigorous: the
  explicit `hashlib.md5`-based `_stable_seed`/`_seeded_rng` (replacing
  Python's per-process-randomized `hash()`), the explicit `KFold(...,
  random_state=1)` instead of relying on `GridSearchCV`'s default
  integer-`cv` expansion, and `n_jobs=1` for the grid search are all
  the kind of detail that's easy to get wrong under multiprocessing
  and was clearly gotten right here, with the reasoning documented
  inline every time.
- **Leakage-free warm/cold split**, enforced structurally (cold users
  are withheld before `GridSearchCV` ever sees the data), not just
  asserted after the fact.
- **The README's explicit "methodological contributions vs.
  engineering improvements" table** is a genuinely good practice for
  a thesis repo — it tells a reader exactly which design choices are
  being defended as a research claim versus which are just
  implementation necessities, heading off a whole class of "why did
  you do X" questions.
- **`_select_batch`'s vectorized scoring pass** (`_base_qi_eligible @
  pu_cold`, `np.argpartition` for top-b selection instead of a full
  sort) is genuinely efficient, O(F) per candidate item and O(n) for
  the partial top-k selection — this part of the hot path is already
  written the right way.
- **The `_sanity_check_corrections` self-test** for the hand-rolled
  Holm/BH implementations
  ([scripts/experiments/ranking_significance_and_correction.py:123](scripts/experiments/ranking_significance_and_correction.py:123))
  is a nice, cheap piece of defensive engineering exactly where it
  matters (a silently-wrong multiple-comparisons correction would be a
  serious, hard-to-notice error in the paper's statistical claims).

---

*Generated by a static read-through of the tracked source files only;
no scripts in this repo were executed as part of this review.*
