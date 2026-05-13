#######################################################################################
#                Section 5.3 main comparison — DeepAR vs. Chronos
#
# Reproduces Figure 3 of the paper: a 5-box boxplot of the cross-component share
# across 100 forecast origins on the UCI Electricity dataset.
#
#   * DeepAR-analytical  -> baseline (closed-form Gaussian L2/L3 + marginal-Gaussian-mixture L1)
#   * DeepAR-KDE         -> sample-based estimator on DeepAR trajectories
#   * DeepAR-kNN         -> sample-based estimator on DeepAR trajectories
#   * Chronos-KDE        -> sample-based estimator on Chronos trajectories
#   * Chronos-kNN        -> sample-based estimator on Chronos trajectories
#
# Each step caches under `results/5_3/` and is a no-op on re-run. A trained DeepAR
# Normal checkpoint is a prerequisite — run `3_train_deepar.py` first if
# `models/deepar_normal.ckpt` is absent. Chronos is loaded zero-shot from Hugging
# Face (`amazon/chronos-t5-base`). A single GPU is sufficient.
#
# If the script is interrupted mid-Step 2, delete both `trajectories_deepar.npy` and
# `raw_deepar.npy` from `results/5_3/` before re-running — otherwise the cache pair
# can be left in an inconsistent state.
#######################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────
# ── Imports ──────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
import sys, os, warnings, logging, time
t0 = time.time()
sys.path.insert(0, os.path.join(os.path.abspath(""), ".."))

# Set the CUDA device (adjust if you have multiple GPUs and want to use a different one)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pytorch_forecasting import DeepAR
import pytorch_forecasting
import chronos
from chronos import ChronosPipeline

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
    build_feature_groups,
)
from entropy_shapley.utils_deepar import build_inputs
from entropy_shapley.utils_paradigm import (
    DeepARTrajectoryImputer, ChronosTrajectoryImputer,
    compute_trajectories, compute_hierarchy,
    compute_hierarchy_analytical_deepar, compute_shapley,
    cross_share, summary_table, plot_fig3_main_boxplot,
    extract_forecast_test_data, compute_forecast_metrics,
    plot_forecast_appendix,
)
from entropy_shapley.estimators import KDECopulaImputer, KNNImputer

# Suppress warnings and reduce logging verbosity for cleaner output.
warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.ERROR)   # silence harmless head-table timestamp warnings on PDF save


# ─────────────────────────────────────────────────────────────────────────────────────
# ── Setup (data parameters identical to 3_train_deepar.py) ───────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("─" * 60)
print("⚙️  Section 5.3 main comparison — DeepAR vs. Chronos")
print("─" * 60)

# Version stamps for paper-grade reproducibility (uv.lock is authoritative;
# this just surfaces the actual runtime versions in the log).
print(f"torch={torch.__version__} | pytorch_forecasting={pytorch_forecasting.__version__} "
      f"| chronos={getattr(chronos, '__version__', '?')}")

# Data parameters
T                 = 12
CONTEXT           = 12 * 7
BLOCKS_PER_DAY    = 12
N_SERIES          = 100
START_DATE        = "2013-03-27 18:00:00"
VALIDATION_PERIOD = 12 * 30
DROP_SERIES       = {"182"}
DATA_PATH         = "../datasets/electricity.csv"

# Shapley players (calendar + 3 history bands)
P            = 6
PLAYER_NAMES = ["hour", "weekday", "month", "recent", "middle", "old"]
P_FEAT       = CONTEXT + 3

# Shapley estimation budget (tune for runtime ↔ MC noise)
N_INSTANCES    = 100   # forecast origins evaluated 100
N_BACKGROUNDS  = 25    # background paths per instance
N_TRAJECTORIES = 1000   # sampled paths per (instance, coalition, background) 1000
N_MIXTURE      = 100   # DeepAR-analytical L1 mixture sub-samples
SEED_BG        = 0     # seed for background selection and trajectory sampling
N_JOBS         = 128    # parallel workers for KDE/kNN hierarchies — adjust to your machine

# Output paths and model identifiers
RESULTS_DIR   = "../results/5_3"
PLOT_DIR      = "../plots/5_3"
PLOT_PATH     = f"{PLOT_DIR}/main_comparison_boxplot"
DEEPAR_CKPT   = "../models/deepar_normal.ckpt"
CHRONOS_MODEL = "amazon/chronos-t5-base"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Instances I={N_INSTANCES} | Backgrounds K={N_BACKGROUNDS} | "
      f"Trajectories N={N_TRAJECTORIES} | n_jobs={N_JOBS}")
print(f"Results dir:   {RESULTS_DIR}")
print(f"Plot dir:      {PLOT_DIR}")

# Pre-check: DeepAR checkpoint must exist before step 2 tries to load it.
if not os.path.exists(DEEPAR_CKPT):
    raise SystemExit(
        f"DeepAR checkpoint not found at {DEEPAR_CKPT}.\n"
        f"Run `3_train_deepar.py` first to train and cache the model."
    )


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 1. Load Electricity & build inputs ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("📊  1. Data setup (load Electricity + build inputs)")
print("─" * 60)

df_long = build_long_dataframe(
    data_path=DATA_PATH, start_date=START_DATE, n_series=N_SERIES,
    drop_series=DROP_SERIES, blocks_per_day=BLOCKS_PER_DAY,
)
training, validation, test_dataset, train_cutoff, val_cutoff = build_datasets(
    df_long, context=CONTEXT, t_horizon=T,
    validation_period=VALIDATION_PERIOD,
)
make_dl = make_dataloader_factory(training, df_long, context=CONTEXT,
                                   t_horizon=T, blocks_per_day=BLOCKS_PER_DAY)
feature_groups = build_feature_groups(CONTEXT, blocks_per_day=BLOCKS_PER_DAY)

# build_inputs caches `results/5_3/inputs.npz`. On re-run this is a no-op
# load and the same N_INSTANCES + N_BACKGROUNDS draws are reused, so the
# trajectory caches stay consistent across estimator sweeps.
inputs = build_inputs(
    df_long, val_cutoff,
    CONTEXT=CONTEXT, T=T, P_FEAT=P_FEAT, BLOCKS_PER_DAY=BLOCKS_PER_DAY,
    I=N_INSTANCES, K=N_BACKGROUNDS, seed_bg=SEED_BG,
    feature_groups=feature_groups,
    results_dir=RESULTS_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 2. DeepAR re-sample with raw distribution params ─────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
# The existing trajectories_deepar.npy has only samples; the analytical
# reference needs the raw projector outputs alongside. Force a re-sample
# by deleting the existing file (or the existing raw_deepar.npy if it's
# missing).

print("\n" + "─" * 60)
print("🎲  2. DeepAR trajectories (with raw distribution params)")
print("─" * 60)

_traj_path = os.path.join(RESULTS_DIR, "trajectories_deepar.npy")
_raw_path  = os.path.join(RESULTS_DIR, "raw_deepar.npy")
if os.path.exists(_traj_path) and not os.path.exists(_raw_path):
    print(f"  ↻ removing samples-only cache to force re-sample with raw: {_traj_path}")
    os.remove(_traj_path)

deepar_model = DeepAR.load_from_checkpoint(DEEPAR_CKPT)
deepar_model.eval()
deepar_imp = DeepARTrajectoryImputer(
    deepar_model, make_dl,
    T=T, feature_groups=feature_groups, predict_batch_size=1024,
)

deepar_traj = compute_trajectories(
    deepar_imp, inputs, N=N_TRAJECTORIES, seed=SEED_BG,
    results_dir=RESULTS_DIR, chunk=1024, save_raw=True,
)
deepar_raw = np.load(_raw_path, mmap_mode="r")
print(f"  trajectories_deepar shape: {deepar_traj.shape}")
print(f"  raw_deepar shape:          {deepar_raw.shape}")


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 3. Chronos trajectories ──────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
# Zero-shot inference from the pretrained Chronos-T5-base pipeline. The
# trajectory cache is written to results/5_3/trajectories_chronos_base.npy.

print("\n" + "─" * 60)
print("🎲  3. Chronos trajectories (zero-shot)")
print("─" * 60)

chronos_pipeline = ChronosPipeline.from_pretrained(
    CHRONOS_MODEL,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
chronos_imp = ChronosTrajectoryImputer(
    chronos_pipeline,
    T=T, feature_groups=feature_groups,
    context=CONTEXT, chunk=8,
)
chronos_imp.model_name = "chronos_base"

# Use a chunk size ≥ total flat-size so dedup runs globally → identical
# target histories give identical trajectories (null-player axiom holds
# exactly for inactive features).
_chronos_flat_size = inputs["x_explain"].shape[0] * inputs["coalitions"].shape[0] * inputs["bg"].shape[0]
chronos_traj = compute_trajectories(
    chronos_imp, inputs, N=N_TRAJECTORIES, seed=SEED_BG,
    results_dir=RESULTS_DIR, chunk=_chronos_flat_size + 1,
)
print(f"  trajectories_chronos_base shape: {chronos_traj.shape}")


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 4. Hierarchies × 5 (DeepAR analytical + KDE + kNN, Chronos KDE + kNN)
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🪜  4. Hierarchies × 5 (analytical / KDE / kNN)")
print("─" * 60)

# DeepAR analytical: closed-form Gaussian L2/L3 + Gaussian-mixture L1 over
# the same trajectory paths. Cache: hierarchy_deepar_analytical.npz.
deepar_hier_anal = compute_hierarchy_analytical_deepar(
    deepar_traj, deepar_raw,
    model_name="deepar", estimator_suffix="_analytical",
    results_dir=RESULTS_DIR,
    n_mixture=N_MIXTURE,
)

# DeepAR KDE-Copula. Cache: hierarchy_deepar_kde.npz.
deepar_hier_kde = compute_hierarchy(
    np.asarray(deepar_traj).astype(np.float64),
    model_name="deepar", estimator_cls=KDECopulaImputer,
    estimator_suffix="_kde", results_dir=RESULTS_DIR, n_jobs=N_JOBS,
)

# DeepAR kNN. Cache: hierarchy_deepar_knn.npz.
deepar_hier_knn = compute_hierarchy(
    np.asarray(deepar_traj).astype(np.float64),
    model_name="deepar", estimator_cls=KNNImputer,
    estimator_kwargs={"k": 5}, estimator_suffix="_knn",
    results_dir=RESULTS_DIR, n_jobs=N_JOBS,
)

# Chronos KDE-Copula. Cache: hierarchy_chronos_base_kde.npz.
chronos_hier_kde = compute_hierarchy(
    np.asarray(chronos_traj).astype(np.float64),
    model_name="chronos_base", estimator_cls=KDECopulaImputer,
    estimator_suffix="_kde", results_dir=RESULTS_DIR, n_jobs=N_JOBS,
)

# Chronos kNN. Cache: hierarchy_chronos_base_knn.npz.
chronos_hier_knn = compute_hierarchy(
    np.asarray(chronos_traj).astype(np.float64),
    model_name="chronos_base", estimator_cls=KNNImputer,
    estimator_kwargs={"k": 5}, estimator_suffix="_knn",
    results_dir=RESULTS_DIR, n_jobs=N_JOBS,
)


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 5. Shapley × 5 ───────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🎯  5. Shapley values from cached value functions")
print("─" * 60)

shapley_pairs = {
    ("deepar",       "analytical"): compute_shapley(deepar_hier_anal,
        model_name="deepar", n_players=P, results_dir=RESULTS_DIR,
        estimator_suffix="_analytical"),
    ("deepar",       "kde"):        compute_shapley(deepar_hier_kde,
        model_name="deepar", n_players=P, results_dir=RESULTS_DIR,
        estimator_suffix="_kde"),
    ("deepar",       "knn"):        compute_shapley(deepar_hier_knn,
        model_name="deepar", n_players=P, results_dir=RESULTS_DIR,
        estimator_suffix="_knn"),
    ("chronos_base", "kde"):        compute_shapley(chronos_hier_kde,
        model_name="chronos_base", n_players=P, results_dir=RESULTS_DIR,
        estimator_suffix="_kde"),
    ("chronos_base", "knn"):        compute_shapley(chronos_hier_knn,
        model_name="chronos_base", n_players=P, results_dir=RESULTS_DIR,
        estimator_suffix="_knn"),
}


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 6. Cross-share table + summary table ─────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("📋  6. Cross-share + summary tables (CSV + LaTeX)")
print("─" * 60)

cross_df = cross_share(
    shapley_pairs,
    out_path=os.path.join(RESULTS_DIR, "cross_share_main.csv"),
)

print("Cross-share DataFrame (first 10 rows):")
print(cross_df.head(10))
print("\n")

hierarchy_pairs = {
    ("deepar",       "analytical"): deepar_hier_anal,
    ("deepar",       "kde"):        deepar_hier_kde,
    ("deepar",       "knn"):        deepar_hier_knn,
    ("chronos_base", "kde"):        chronos_hier_kde,
    ("chronos_base", "knn"):        chronos_hier_knn,
}
arch = {
    ("deepar",       "analytical"): "AR (trained), analytical",
    ("deepar",       "kde"):        "AR (trained), KDE-Copula",
    ("deepar",       "knn"):        "AR (trained), kNN",
    ("chronos_base", "kde"):        "AR-token FM, KDE-Copula",
    ("chronos_base", "knn"):        "AR-token FM, kNN",
}
summary_df = summary_table(
    hierarchy_pairs, cross_df,
    out_path=os.path.join(RESULTS_DIR, "summary_table_main.csv"),
    latex_path=os.path.join(RESULTS_DIR, "summary_table_main.tex"),
    architecture=arch,
)
print(summary_df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 7. Main figure: 5-box boxplot ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("📈  7. Main figure (5-box cross-share boxplot)")
print("─" * 60)

# Headline metric: cross_pct ∈ [0, 100%], the bounded share of attribution
# mass that goes into cross-step coupling. No y-clip needed (cross_pct is
# bounded by definition). The DeepAR cluster sits low, Chronos high.
fig = plot_fig3_main_boxplot(
    cross_df,
    model_estimator_pairs=[
        ("deepar",       "analytical"),
        ("deepar",       "kde"),
        ("deepar",       "knn"),
        ("chronos_base", "kde"),
        ("chronos_base", "knn"),
    ],
    model_colors={
        "deepar":       "#1f77b4",   # blue
        "chronos_base": "#d62728",   # red/orange
    },
    model_display={
        "deepar":       "DeepAR",
        "chronos_base": "Chronos",
    },
    save_path=PLOT_PATH,
    metric="cross_pct",
)
plt.show()


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 8. Sanity checks ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🔍  8. Verification (sanity checks vs paper claims)")
print("─" * 60)

# Sanity-check thresholds (paper claims; printed as PASS/WARN against these)
CORR_ANAL_KDE_MIN         = 0.8
KDE_KNN_GAP_MAX_PP        = 5.0
DEEPAR_CHRONOS_GAP_MIN_PP = 25.0

def _cp(model: str, est: str) -> np.ndarray:
    return cross_df[(cross_df["model"] == model) &
                    (cross_df["estimator"] == est)]["cross_pct"].to_numpy()

cp_anal = _cp("deepar", "analytical")
cp_kde  = _cp("deepar", "kde")
cp_knn  = _cp("deepar", "knn")
cp_chr_kde = _cp("chronos_base", "kde")
cp_chr_knn = _cp("chronos_base", "knn")

# 1) Analytical vs KDE on DeepAR — validity anchor for the sample-based estimator.
r_anal_kde = float(np.corrcoef(cp_anal, cp_kde)[0, 1])
print(f"  (1) corr(DeepAR-analytical, DeepAR-KDE)        = {r_anal_kde:+.3f}  "
      f"({'PASS' if r_anal_kde > CORR_ANAL_KDE_MIN else f'WARN  expected > {CORR_ANAL_KDE_MIN}'})")

# 2) KDE vs kNN agreement on DeepAR — estimator robustness.
diff_kde_knn = abs(float(np.median(cp_knn) - np.median(cp_kde)))
print(f"  (2) |median(kNN) - median(KDE)| on DeepAR       = {diff_kde_knn:.1f} pp  "
      f"({'PASS' if diff_kde_knn <= KDE_KNN_GAP_MAX_PP else f'WARN  expected ≤ {KDE_KNN_GAP_MAX_PP:.0f} pp'})")

# 3) DeepAR vs Chronos median gap — the headline diagnostic.
gap = float(np.median(cp_chr_kde) - np.median(cp_kde))
print(f"  (3) median(Chronos-KDE) - median(DeepAR-KDE)    = {gap:+.1f} pp  "
      f"({'PASS' if gap >= DEEPAR_CHRONOS_GAP_MIN_PP else f'WARN  expected ≥ {DEEPAR_CHRONOS_GAP_MIN_PP:.0f} pp'})")


# ─────────────────────────────────────────────────────────────────────────────────────
# ── 9. Appendix: forecast quality + qualitative joint-structure ──────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
# Anchors the cross-component diagnostic against standard accuracy
# metrics (CRPS / MAE) and shows the joint structure visually so the
# Section 5.3 claim "similar accuracy, different dependencies" is concrete.
print("\n" + "─" * 60)
print("📐  9. Forecast-quality companion (appendix)")
print("─" * 60)

full_idx = int(np.where(inputs["coalitions"].all(axis=-1))[0][0])
print(f"  full coalition index = {full_idx}")

samples_full = {}
for name, traj in [("deepar", deepar_traj), ("chronos_base", chronos_traj)]:
    # (I, n_coal, K, N, T) → (I, K*N, T)  — flatten the K backgrounds
    # into one bag of joint paths at the full coalition. Need .copy() because
    # slicing along axis=1 of a memmap'd npy gives a non-contiguous view that
    # cannot be reshaped in-place.
    a = np.ascontiguousarray(traj[:, full_idx]).astype(np.float64)
    samples_full[name] = a.reshape(a.shape[0], -1, a.shape[-1])
    print(f"  samples_full[{name}] shape: {samples_full[name].shape}")

test_data = extract_forecast_test_data(df_long, inputs, CONTEXT=CONTEXT, T=T)
print(f"  y_true shape: {test_data['y_true'].shape}")

print("\n  Forecast metrics (CRPS / MAE on test instances):")
metrics = {}
for name, s in samples_full.items():
    m = compute_forecast_metrics(s, test_data["y_true"])
    metrics[name] = m
    print(f"    {name:<14}  CRPS median = {m['crps_overall']:.4f}   "
          f"MAE median = {m['mae_overall']:.4f}")

# Side-by-side relative gap (the headline accuracy claim).
crps_gap = (metrics["chronos_base"]["crps_overall"] - metrics["deepar"]["crps_overall"]) \
           / metrics["deepar"]["crps_overall"]
mae_gap  = (metrics["chronos_base"]["mae_overall"]  - metrics["deepar"]["mae_overall"]) \
           / metrics["deepar"]["mae_overall"]
print(f"  → relative gap Chronos vs DeepAR:  CRPS {100*crps_gap:+.1f}%  "
      f"MAE {100*mae_gap:+.1f}%\n")

# Pick two forecast origins to display (q25/q75 of DeepAR-KDE cross_pct):
#   - q25: low-cross_pct case — marginal-dominated, "looks easy"
#   - q75: high-cross_pct case — strong cross-step dependence
deepar_kde_cp = cross_df[(cross_df["model"] == "deepar") &
                         (cross_df["estimator"] == "kde")] \
                .sort_values("cross_pct").reset_index(drop=True)
n_cp = len(deepar_kde_cp)
rep_instances = [
    int(deepar_kde_cp.iloc[int(0.25 * n_cp)]["instance"]),
    int(deepar_kde_cp.iloc[int(0.75 * n_cp)]["instance"]),
]
print(f"  representative origins (q25/q75 of DeepAR-KDE cross_pct): "
      f"{rep_instances}")

# Save metrics to a small CSV for paper-text quoting.
metrics_rows = []
for name, m in metrics.items():
    for t in range(T):
        metrics_rows.append(dict(model=name, horizon=t + 1,
                                  crps_median=float(m["crps_per_t"][t]),
                                  mae_median=float(m["mae_per_t"][t])))
    metrics_rows.append(dict(model=name, horizon=-1,
                              crps_median=m["crps_overall"],
                              mae_median=m["mae_overall"]))
pd.DataFrame(metrics_rows).to_csv(
    os.path.join(RESULTS_DIR, "forecast_metrics_main.csv"), index=False)
print(f"  metrics CSV → {RESULTS_DIR}/forecast_metrics_main.csv")

# Render the 2×2 appendix figure. The helper expects the raw 5D
# trajectory arrays (I, n_coal, K, N, T) — slices full_idx internally.
fig_appx = plot_forecast_appendix(
    df_long, inputs,
    {"deepar": deepar_traj, "chronos_base": chronos_traj},
    CONTEXT=CONTEXT, T=T,
    instance_indices=rep_instances, full_coal_idx=full_idx,
    n_overlay_paths=80,
    model_colors={"deepar": "#1f77b4", "chronos_base": "#d62728"},
    model_display={"deepar": "DeepAR", "chronos_base": "Chronos-T5"},
    save_path=f"{PLOT_DIR}/forecast_appendix",
)
plt.show()


print("\n" + "─" * 60)
print("✅  Done with main comparison")
print(f"Total runtime: {(time.time() - t0) / 60:.1f} minutes")
print("─" * 60)
