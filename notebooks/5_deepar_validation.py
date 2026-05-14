#######################################################################################
#              Appendix D.3.1 — sample-based DeepAR value-function validation
#
# Reproduces Figure D.4 of the paper. Sweeps four sample-based entropy estimators
# across three DeepAR likelihoods and a grid of trajectory budgets N, then compares
# each to DeepAR's own factorized analytical reference (Section 4.2.2) — the
# strongest validity anchor available for sample-based estimators.
#
#   * Parametric Gaussian              (mvn)
#   * Gauss copula, true marginals     (mixture_copula)
#   * Gauss copula, KDE marginals      (kde_copula)
#   * kNN with k=5                     (knn)
#
# Outputs written to `results/D_3/` (CSV tables) and `plots/5_3/` (figures):
#
#   * mae_table.csv                  -> MAE on the value function v(S)
#   * shapley_error_table.csv        -> MAE on the Shapley contribution phi_p
#   * mae_by_cardinality_table.csv   -> per-coalition-cardinality bias
#   * combined_L2_appendix.pdf       -> 2×3 L2 figure (Figure D.4, appendix)
#   * combined_L3_appendix.pdf       -> joint-entropy L3 companion
#   * per_cardinality_L2_appendix.pdf-> bias signature per |S|
#
# A trained DeepAR checkpoint per likelihood is required — run `3_train_deepar.py`
# first if any of `models/deepar_{normal,studentt,lognormal}.ckpt` is missing.
# Missing likelihoods are skipped with a warning.
#######################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────
# ── Imports ──────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
import sys, os, warnings, logging, time
start_time = time.time()
sys.path.insert(0, os.path.join(os.path.abspath(""), ".."))

# Set the CUDA device (adjust if you have multiple GPUs and want to use a different one)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import pytorch_forecasting
from pytorch_forecasting import DeepAR

# Compatibility shim for PyTorch 2.6+ to ensure DeepAR checkpoints load correctly.
import functools
_orig_load = torch.load
@functools.wraps(_orig_load)
def _patched(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched

from entropy_shapley.utils_datasets import (
    build_long_dataframe, build_datasets, make_dataloader_factory,
)
from entropy_shapley.utils_deepar import (
    build_inputs, draw_nested_samples, compute_reference, evaluate_estimators,
    save_reference, save_estimators,
    aggregate_mae, aggregate_mae_by_cardinality, aggregate_shapley_error,
    plot_fig3_combined, plot_fig3_per_cardinality,
)
from entropy_shapley.estimators import (
    KDECopulaImputer, KNNImputer, MVNSampleImputer, MixtureCopulaImputer,
)

# Suppress warnings and reduce logging verbosity for cleaner output.
warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.ERROR)   # silence harmless head-table timestamp warnings on PDF save

# ─────────────────────────────────────────────────────────────────────────────────────
# ── Setup (data parameters identical to 3_train_deepar.py) ───────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("─" * 60)
print("⚙️  Appendix D.3.1 — DeepAR value-function validation")
print("─" * 60)

# Version stamps for paper-grade reproducibility (uv.lock is authoritative;
# this just surfaces the actual runtime versions in the log).
print(f"torch={torch.__version__} | pytorch_forecasting={pytorch_forecasting.__version__}")

# Data parameters
T                 = 12
CONTEXT           = 12 * 7
BLOCKS_PER_DAY    = 12
N_SERIES          = 100
START_DATE        = "2013-03-27 18:00:00"
VALIDATION_PERIOD = 12 * 30
DROP_SERIES       = {"182"}                                  # contains mostly zeros
DATA_PATH         = "../datasets/electricity.csv"

# Shapley players (calendar + 3 history bands)
P            = 6
PLAYER_NAMES = ["hour", "weekday", "month", "recent", "middle", "old"]

# Validation budget (tune for runtime ↔ MC noise)
N_INSTANCES   = 100                              # forecast origins evaluated
N_BACKGROUNDS = 25                              # background paths per instance
N_GRID        = [50, 100, 250, 500, 1000, 2000]  # trajectory budgets swept
BATCH_I       = 10                               # instances per inner-loop batch (RAM-bound)
SEED_BG       = 0                                # seed for background selection and trajectory sampling
N_PARALLEL    = 128                              # parallel workers for estimator eval — adjust to your machine
N_MIXTURE     = 250                              # reference L2: sub-sample the N_max trajectory parameters
                                                  # used for averaging (Jensen bias O(1/N_MIXTURE),
                                                  # largely cancels under Shapley aggregation)

# Distributions and estimators under test
DISTR = ("normal", "studentt", "lognormal")
ESTIM = {
    "mvn":            MVNSampleImputer,
    "mixture_copula": MixtureCopulaImputer,
    "kde_copula":     KDECopulaImputer,
    "knn":            KNNImputer,
}

# Path to DeepAR checkpoints (trained in `3_train_deepar.py` with the same data parameters above).
MODELS = {
    "normal":    "../models/deepar_normal.ckpt",
    "studentt":  "../models/deepar_studentt.ckpt",
    "lognormal": "../models/deepar_lognormal.ckpt",
}

# Output paths. Validation caches live in `results/D_3/` (separate from §5.3 main-
# comparison caches in `results/5_3/`, because the two pipelines call `build_inputs`
# with different `P_FEAT` values and produce incompatible `inputs.npz` files). The
# figures share `plots/5_3/` via distinct file names.
results_dir = "../results/D_3"
plot_dir    = "../plots/5_3"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

print(f"Instances I={N_INSTANCES} | Backgrounds K={N_BACKGROUNDS} | "
      f"N_grid={N_GRID} | n_jobs={N_PARALLEL}")
print(f"Results dir:   {results_dir}")
print(f"Plot dir:      {plot_dir}")

# Pre-flight check: at least one DeepAR checkpoint must be present, otherwise
# the whole sweep is a no-op.
_missing = [d for d in DISTR if not os.path.exists(MODELS[d])]
if len(_missing) == len(DISTR):
    raise SystemExit(
        f"No DeepAR checkpoints found in `{os.path.dirname(MODELS[DISTR[0]])}/`.\n"
        f"Run `3_train_deepar.py` first to train and cache the models."
    )
if _missing:
    print(f"⚠ missing checkpoints (will be skipped): {_missing}")

# ─────────────────────────────────────────────────────────────────────────────────────
# ── 1. Data setup ────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("📊  1. Data setup")
print("─" * 60)

df_long = build_long_dataframe(
    data_path=DATA_PATH, start_date=START_DATE, n_series=N_SERIES,
    drop_series=DROP_SERIES, blocks_per_day=BLOCKS_PER_DAY,
)
training, validation, test_dataset, train_cutoff, val_cutoff = build_datasets(
    df_long, context=CONTEXT, t_horizon=T,
    validation_period=VALIDATION_PERIOD,
)
make_dl = make_dataloader_factory(training, df_long, context=CONTEXT, t_horizon=T, blocks_per_day=BLOCKS_PER_DAY)
print(f"df_long: {len(df_long):,} rows | train_cutoff: {train_cutoff} | val_cutoff: {val_cutoff} | T={T}, CONTEXT={CONTEXT}, P={P}\n")

# Build inputs (instances, backgrounds, coalitions)
# We sample `I` instances from the test period (after val_cutoff) and `K` background
# rows from before val_cutoff (training + validation). All `2^P = 64` coalitions over
# the six feature groups are enumerated. Cached as `results/D_3/inputs.npz`.
inputs = build_inputs(
    df_long, val_cutoff,
    CONTEXT=CONTEXT, T=T, P_FEAT=P, BLOCKS_PER_DAY=BLOCKS_PER_DAY,
    I=N_INSTANCES, K=N_BACKGROUNDS, seed_bg=0,
    results_dir=results_dir,
)



# ─────────────────────────────────────────────────────────────────────────────────────
# ── 2. Run validation ────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🧪  2. Run validation")
print("─" * 60)

# Per (distribution, seed) we iterate over *instance batches* (size BATCH_I): each
# batch draws its own slice of (I_batch, n_coal, K, N_max, T) trajectories in RAM,
# feeds them to compute_reference + evaluate_estimators, then drops them. The
# small per-batch result tables (shape (I_batch, n_coal, T)) are accumulated and
# concatenated into the full (I, n_coal, T) tables at the end of each (dist, seed),
# then written *once* to disk. No large `samples_*.npy`/`raw_*.npy` files ever
# materialize — RAM/disk peak is bounded by BATCH_I instead of I.
N_max   = max(N_GRID)
# Fixed MC seed for trajectory sampling. The pipeline is wrapped in a single-key
# `{dist: {0: ...}}` mapping so the library aggregators (`aggregate_mae`,
# `aggregate_shapley_error`, `aggregate_mae_by_cardinality`) can be called with
# `seeds=[0]` without further changes.
SEED = 0
refs_by_dist_seed = {dist: {} for dist in DISTR}
ests_by_dist_seed = {dist: {} for dist in DISTR}


def _try_load_finished(dist: str):
    """Return (refs, ests) if all small result caches for this dist are on disk;
    otherwise (None, None) to signal we need to redraw trajectories."""
    ref_path = os.path.join(results_dir, dist, f"refs_seed{SEED}.npz")
    est_paths = {(e, N): os.path.join(results_dir, dist, f"est_{e}_seed{SEED}_N{N}.npz")
                 for e in ESTIM for N in N_GRID}
    if not os.path.exists(ref_path) or not all(os.path.exists(p) for p in est_paths.values()):
        return None, None
    rnpz = np.load(ref_path)
    refs = {"v_ref_L2": rnpz["v_ref_L2"], "v_ref_L3": rnpz["v_ref_L3"]}
    ests = {e: {N: {k: np.load(est_paths[(e, N)])[k] for k in ("L1", "L2", "L3")}
                for N in N_GRID}
            for e in ESTIM}
    return refs, ests


for dist in DISTR:
    if not os.path.exists(MODELS[dist]):
        print(f"  ⚠ {dist}: checkpoint not found, skipping")
        continue
    print(f"\n ─── {dist}")

    # Re-run shortcut: if refs + all (est, N) tables for this dist are already
    # on disk, reuse them directly and skip everything below.
    cached_refs, cached_ests = _try_load_finished(dist)
    if cached_refs is not None:
        refs_by_dist_seed[dist][SEED] = cached_refs
        ests_by_dist_seed[dist][SEED] = cached_ests
        print(f"    ✓ reused cached results")
        continue

    model = DeepAR.load_from_checkpoint(MODELS[dist]).eval()

    # Per-batch accumulators. After concat across batches the shapes match
    # what compute_reference / evaluate_estimators would produce on the full I.
    I_total    = inputs['x_explain'].shape[0]
    n_batches  = (I_total + BATCH_I - 1) // BATCH_I
    ref_L2_chunks: list = []
    ref_L3_chunks: list = []
    est_chunks = {e: {N: {"L1": [], "L2": [], "L3": []}
                      for N in N_GRID} for e in ESTIM}

    t_dist = time.time()
    with tqdm(total=I_total, desc="    drawing", unit="inst", leave=False) as pbar:
        for b_idx, i_start in enumerate(range(0, I_total, BATCH_I)):
            i_end = min(i_start + BATCH_I, I_total)

            # Slice instances; backgrounds & coalitions stay shared (so identical
            # bg/coal configuration across batches and across distributions).
            inputs_batch = {**inputs, 'x_explain': inputs['x_explain'][i_start:i_end]}

            # Distinct sub-seed per batch so each batch consumes its own RNG
            # subsequence (rather than every batch starting from the same state
            # and producing duplicated draws).
            sub_seed = SEED * 1_000_000 + i_start

            cache = draw_nested_samples(
                model, inputs_batch,
                T=T, P=P, distribution=dist, N_max=N_max, seed=sub_seed,
                make_dataloader=make_dl, predict_batch_size=1024,
                results_dir=results_dir, ckpt_path=MODELS[dist],
                save=False,                     # ← keep trajectories in RAM only
            )

            ref_batch = compute_reference(
                cache['samples'], cache['raw'],
                distribution=dist, results_dir=results_dir, seed=SEED,
                save=False, n_mixture=N_MIXTURE, n_jobs=N_PARALLEL,
            )
            est_batch = evaluate_estimators(
                cache['samples'], cache['raw'],
                distribution=dist, estimators=ESTIM,
                N_grid=N_GRID, seed=SEED,
                results_dir=results_dir, n_jobs=N_PARALLEL,
                save=False,
            )

            # Stash the small per-batch chunks (instance axis = 0).
            ref_L2_chunks.append(ref_batch['v_ref_L2'])     # (I_batch, n_coal, T)
            ref_L3_chunks.append(ref_batch['v_ref_L3'])     # (I_batch, n_coal)
            for e, by_N in est_batch.items():
                for N, d in by_N.items():
                    for k in ("L1", "L2", "L3"):
                        est_chunks[e][N][k].append(d[k])

            # Drop trajectories explicitly so the next batch gets the RAM back.
            del cache, ref_batch, est_batch
            pbar.update(i_end - i_start)

    # Concat across batches → full (I, ...) tables, identical shape to the
    # non-batched code path. Persist once per dist.
    refs_full = {
        'v_ref_L2': np.concatenate(ref_L2_chunks, axis=0),
        'v_ref_L3': np.concatenate(ref_L3_chunks, axis=0),
    }
    ests_full = {
        e: {N: {k: np.concatenate(est_chunks[e][N][k], axis=0) for k in ("L1", "L2", "L3")}
            for N in N_GRID}
        for e in ESTIM
    }
    save_reference(refs_full, distribution=dist, results_dir=results_dir, seed=SEED)
    save_estimators(ests_full, distribution=dist, results_dir=results_dir, seed=SEED)

    refs_by_dist_seed[dist][SEED] = refs_full
    ests_by_dist_seed[dist][SEED] = ests_full

    print(f"    ✓ {I_total} inst × {len(ESTIM)} est × N∈{N_GRID}  "
          f"[{(time.time() - t_dist) / 60:.1f} min]")

# Check the chain rule holds for all references and estimators across every (dist, seed, N)
print(f"\nChain-rule check (max |L3 − sum_t L2_t| across N × estimators):")
all_ok = True
for dist in refs_by_dist_seed:
    ref_res = max(
        (float(np.max(np.abs(r['v_ref_L3'] - r['v_ref_L2'].sum(axis=-1))))
         for r in refs_by_dist_seed[dist].values()),
        default=0.0,
    )
    est_res = max(
        (float(np.max(np.abs(v['L3'] - v['L2'].sum(axis=-1))))
         for by_est in ests_by_dist_seed[dist].values()
         for by_n   in by_est.values()
         for v      in by_n.values()),
        default=0.0,
    )
    ok = ref_res < 1e-8 and est_res < 1e-8
    all_ok &= ok
    print(f"    {dist:9s}: refs {ref_res:.1e}  estimators {est_res:.1e}  "
          f"({'PASS' if ok else 'FAIL'})")
    assert ok, f"chain rule broken for {dist} (refs={ref_res:.2e}, estimators={est_res:.2e})"

# ─────────────────────────────────────────────────────────────────────────────────────
# ── 3. Aggregate error tables ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("📋  3. Aggregate error tables")
print("─" * 60)

# We build two long-format tables: one for the RMSE on the value-function $v(S)$, one
# for the MAE on the Shapley contribution $\phi_p$. Both share the schema
# `(likelihood, estimator, seed, N, level, value)` and form the basis for Figures 3
# and 3-Appendix.
# 
# The Shapley vector is computed in closed form per instance: for a value-function
# table indexed by binary coalition mask $s \in \{0, ..., 2^P-1\}$,
# $$\phi_p = \sum_{S\,\not\ni\,p}\, \frac{|S|!\,(P-|S|-1)!}{P!}\,\bigl(v(S\cup\{p\}) - v(S)\bigr).$$
# Each estimator is paired with its own seed's reference (same trajectory cache feeds
# both Shapley computations), so the SEM bands across seeds reflect estimator
# behavior rather than reference noise.

mae_v_df = aggregate_mae(
    refs_by_dist_seed, ests_by_dist_seed,
    distributions=tuple(refs_by_dist_seed.keys()),
    estimators=tuple(ESTIM),
    N_grid=N_GRID, seeds=[SEED],
    out_path=os.path.join(results_dir, 'mae_table.csv'),
)
print("\nMAE on value function v(S):")
print(mae_v_df.head(5))
print("\n")


shap_df = aggregate_shapley_error(
    refs_by_dist_seed, ests_by_dist_seed,
    distributions=tuple(refs_by_dist_seed.keys()),
    estimators=tuple(ESTIM),
    N_grid=N_GRID, seeds=[SEED],
    n_players=P,
    out_path=os.path.join(results_dir, 'shapley_error_table.csv'),
)
print("\nMAE on Shapley value φ_p:")
print(shap_df.head(5))


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 4. Create appendix figure (L2) ───────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("📈  4. Create appendix figure (L2)")
print("─" * 60)

# 2×3 grid: top row is MAE on the value function v(S) ("scale"), bottom row is MAE
# on the Shapley contribution phi_p ("topology"). Columns scan likelihoods from
# Gaussian → Student-t → LogNormal. On non-Gaussian likelihoods most of the v(S)
# error is a coalition-independent constant offset shared across all four
# estimators — large in absolute nats but invisible to Shapley aggregation, so the
# bottom row is what actually propagates downstream.

DISTR_LABELS = {"normal": "Gaussian", "studentt": r"Student-$t$", "lognormal": "LogNormal"}
ESTIM_LABEL = {
    "mvn":               "Parametric Gaussian",
    "mixture_copula":    "Gauss copula (true marginals)",
    "kde_copula":        "Gauss copula (KDE marginals)",
    "knn":               r"kNN ($k=5$)",
}
ESTIM_COLORS = {
    "mvn":               "#1f77b4",
    "mixture_copula":    "#ff7f0e",
    "kde_copula":        "#d62728",
    "knn":               "#2ca02c",
}
ESTIM_MARKERS = {
    "mvn":               "o",
    "mixture_copula":    "s",
    "kde_copula":        "D",
    "knn":               "^",
}

_PLOT_KW = dict(
    distributions=DISTR,
    distribution_display=DISTR_LABELS,
    estimators=tuple(ESTIM),
    estimator_display=ESTIM_LABEL,
    estimator_colors=ESTIM_COLORS,
    estimator_markers=ESTIM_MARKERS,
)

fig_L2 = plot_fig3_combined(
    mae_v_df, shap_df,
    level="L2",
    save_path=plot_dir + "/combined_L2_appendix",
    **_PLOT_KW,
)
plt.show()

# ─────────────────────────────────────────────────────────────────────────────────────
# ── 5. Create appendix figure (L3) ───────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("📐  5. Create appendix figure (L3)")
print("─" * 60)

# For completeness we render the same 2×$N_{\text{dist}}$ figure on the
# joint-entropy level $L_3 = \sum_t L_2^{(t)}$. The bias signature of `naive_gauss`
# — its Shapley MAE *rising* monotonically with $N$ as variance shrinks and the
# coalition-dependent fraction of the bias is unmasked — is most visible here on
# the non-Gaussian columns of the bottom row.

fig_appendix = plot_fig3_combined(
    mae_v_df, shap_df,
    level="L3",
    save_path=plot_dir + "/combined_L3_appendix",
    **_PLOT_KW,
)
plt.show()


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 6. Per-coalition-cardinality bias plot ───────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🔬  6. Per-coalition-cardinality bias plot")
print("─" * 60)

# Where in the coalition lattice does each estimator's bias materialize?
# We slice at the largest N (asymptotic, MC-noise-shrunk) and aggregate the
# absolute v(S)-error per coalition cardinality |S| ∈ {0, ..., P}. On non-
# Gaussian likelihoods the parametric Gaussian MVN-fit shows a tent-shaped
# bias peaking at intermediate |S| where instance and background information
# mix most. The Gauss-copula estimators are flatter; kNN's pattern is
# distinct (driven by the joint-dimensionality of the conditional rather
# than by marginal mismatch).

card_df = aggregate_mae_by_cardinality(
    refs_by_dist_seed, ests_by_dist_seed,
    distributions=tuple(refs_by_dist_seed.keys()),
    estimators=tuple(ESTIM),
    seeds=[SEED], n_players=P,
    N=max(N_GRID),
    level="L2",
    out_path=os.path.join(results_dir, 'mae_by_cardinality_table.csv'),
)
print("\nMAE on value function v(S) by coalition cardinality |S|:")
print(card_df.head(5))

fig_card = plot_fig3_per_cardinality(
    card_df, level="L2", n_players=P,
    save_path=plot_dir + "/per_cardinality_L2_appendix",
    **_PLOT_KW,
)
plt.show()


print("\n" + "─" * 60)
print("✅  Done with DeepAR validation")
print(f"Total runtime: {(time.time() - start_time) / 60:.1f} minutes")
print("─" * 60)
