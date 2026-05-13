#######################################################################################
#                   DeepAR training on UCI Electricity dataset
#
# This script trains three DeepAR models on the UCI Electricity dataset, one per 
# supported likelihood (Gaussian, Student-t, LogNormal). The trained checkpoints are
# saved to disk and used by the main comparison script (4_main_comparison.py) and
# the DeepAR validation script (5_deepar_validation.py).
#
#   * Normal       -> models/deepar_normal.ckpt (used by Section 5.3 main)
#   * Student-t    -> models/deepar_studentt.ckpt (used by Appendix D.3.1)
#   * Log-Normal   -> models/deepar_lognormal.ckpt (used by Appendix D.3.1)
#
# By default, the script checks for existing checkpoints and skips training if they 
# are found; set RETRAIN = True to force re-training. A single GPU is sufficient, 
# and training all three models takes around two hours on an RTX 6000 Ada (50 epochs,
# batch size 256, 100 series). 
#######################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────
# ── Imports ──────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
import sys, os, warnings, logging, time
start_time = time.time()
sys.path.insert(0, os.path.join(os.path.abspath(""), ".."))

# Set the CUDA device
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0") # adjust if you have multiple GPUs and want to use a different one

import pytorch_forecasting
import torch
import numpy as np
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import DeepAR, Baseline
from pytorch_forecasting.metrics import NormalDistributionLoss
from torchmetrics.functional.regression import continuous_ranked_probability_score as crps_fn

# Bugfix for PyTorch 2.6+ to ensure DeepAR checkpoints load correctly.
import functools
_orig = torch.load
@functools.wraps(_orig)
def _patched(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig(*args, **kwargs)
torch.load = _patched

from entropy_shapley.losses import StudentTDistributionLoss, LogNormalDistributionLoss
from entropy_shapley.utils_datasets import build_long_dataframe, build_datasets

# Suppress warnings and reduce logging verbosity for cleaner output.
warnings.filterwarnings('ignore')
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# Print library versions for reproducibility.
print("\nLibrary versions:")
print(f"torch={torch.__version__} | lightning={pl.__version__} | pf={pytorch_forecasting.__version__}\n")

# ─────────────────────────────────────────────────────────────────────────────────────
# ── Setup ────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("─" * 60)
print("⚙️  DeepAR training setup")
print("─" * 60)


# Set parameters
T                 = 12
CONTEXT           = 12 * 7
BLOCKS_PER_DAY    = 12
VALIDATION_PERIOD = 12 * 30
N_SERIES          = 100
START_DATE        = "2013-03-27 18:00:00"           # "2013-01-01" to use the full dataset; later start trains faster
N_HIDDEN          = 64
EPOCHS            = 50
BATCH_SIZE        = 256
N_LAYERS          = 2
LR                = 1e-3
RETRAIN           = False                            # True -> ignore cached checkpoints
DROP_SERIES       = {"182"}                          # contains mostly zeros

# Set paths and checkpoint dir
DATA_PATH         = "../datasets/electricity.csv"   # notebook-relative default
MODEL_DIR         = "../models"
PLOT_DIR          = "../plots/5_3"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Set random seeds for reproducibility
SEED = 2026
np.random.seed(SEED)
pl.seed_everything(SEED, verbose=False)
print(f"{'T':<15}: {T}\n{'CONTEXT':<15}: {CONTEXT}\n{'BLOCKS_PER_DAY':<15}: {BLOCKS_PER_DAY}\n{'N_SERIES':<15}: {N_SERIES}")
print(f"{'HIDDEN':<15}: {N_HIDDEN}\n{'LAYERS':<15}: {N_LAYERS}\n{'EPOCHS':<15}: {EPOCHS}\n{'BATCH_SIZE':<15}: {BATCH_SIZE}\n")
print(f"DEVICE: '{torch.device('cuda' if torch.cuda.is_available() else 'cpu')}'")
print(f"Retraining: '{RETRAIN}'\n")


# ─────────────────────────────────────────────────────────────────────────────────────
# ── Data preprocessing ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("─" * 60)
print("📊  Data preprocessing (electricity dataset)")
print("─" * 60)

df_long = build_long_dataframe(
    data_path=DATA_PATH,
    start_date=START_DATE,
    n_series=N_SERIES,
    drop_series=DROP_SERIES,
    blocks_per_day=BLOCKS_PER_DAY,
)
print("Dataset date range:")
print(f"Start date: {df_long['index'].min()}  |  End date: {df_long['index'].max()}\n")

training, validation, test_dataset, train_cutoff, val_cutoff = build_datasets(
    df_long, context=CONTEXT, t_horizon=T,
    validation_period=VALIDATION_PERIOD,
)
train_dl = training.to_dataloader(train=True,  batch_size=BATCH_SIZE, num_workers=0)
val_dl   = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
test_dl  = test_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

print(f"Dataset splits:")
print(f"training:   {len(training):,} samples")
print(f"validation: {len(validation):,} samples  (rolling-origin, stride=T={T})")
print(f"test:       {len(test_dataset):,} samples  (rolling-origin, stride=T={T})")
print(f"\nDate cutoff (training/validation split):")
ts = df_long.drop_duplicates("time_idx").set_index("time_idx")["index"]
for name, ds in [("Training", training), ("Validation", validation), ("Test", test_dataset)]:
    i0, i1 = ds.index["time"].min(), ds.index["time"].max() + ds.max_prediction_length - 1
    print(f"{name:<11} start: {ts[i0]}  |  end: {ts[i1]}")

# ─────────────────────────────────────────────────────────────────────────────────────
# ── Baseline sanity check ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("📏  Baseline sanity check (last-value persistence)")
print("─" * 60)

baseline = Baseline()
preds = baseline.predict(test_dl,
                          trainer_kwargs=dict(accelerator="auto", devices=1,
                                              enable_progress_bar=False, logger=False),
                          return_y=True)
mae_baseline = float((preds.output - preds.y[0].cpu()).abs().mean())
print(f"MAE (last-value baseline): {mae_baseline:.4f}")


# ─────────────────────────────────────────────────────────────────────────────────────
# ── Train or load DeepAR models ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🤖  Train/load DeepAR models (Gaussian, Student-t, LogNormal)")
print("─" * 60 + "\n")

DISTRIBUTIONS = {
    "Normal":     (NormalDistributionLoss,    f"{MODEL_DIR}/deepar_normal.ckpt"),
    "Student-t":  (StudentTDistributionLoss,  f"{MODEL_DIR}/deepar_studentt.ckpt"),
    "Log-Normal": (LogNormalDistributionLoss, f"{MODEL_DIR}/deepar_lognormal.ckpt"),
}

# Helper function to train or load the DeepAR models
def train_or_load(label, LossCls, ckpt_path):
    if os.path.exists(ckpt_path) and not RETRAIN:
        print(f"✓ {label}: cached → {ckpt_path}")
        map_location = None if torch.cuda.is_available() else "cpu"
        return DeepAR.load_from_checkpoint(ckpt_path, map_location=map_location)
    print(f"──── {label}: training {EPOCHS} epochs")
    early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, mode="min")
    ckpt_dir   = os.path.dirname(ckpt_path)
    ckpt_name  = os.path.splitext(os.path.basename(ckpt_path))[0]
    ckpt_cb    = ModelCheckpoint(dirpath=ckpt_dir, filename=ckpt_name,
                                 monitor="val_loss", mode="min", save_top_k=1,
                                 enable_version_counter=False)
    trainer = pl.Trainer(
        max_epochs=EPOCHS, accelerator="auto", devices=1,
        enable_model_summary=False, default_root_dir=ckpt_dir,
        gradient_clip_val=1.0,
        logger=False, callbacks=[early_stop, ckpt_cb],
        enable_checkpointing=True, enable_progress_bar=True,
    )
    model = DeepAR.from_dataset(
        training, learning_rate=LR, hidden_size=N_HIDDEN, rnn_layers=N_LAYERS,
        optimizer="Adam", loss=LossCls(),
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    print(f"✓ {label}: saved → {ckpt_path}\n")
    map_location = None if torch.cuda.is_available() else "cpu"
    return DeepAR.load_from_checkpoint(ckpt_path, map_location=map_location)

# Train or load models for each distribution and store in a dict.
models = {label: train_or_load(label, *args) for label, args in DISTRIBUTIONS.items()}


# ─────────────────────────────────────────────────────────────────────────────────────
# ── Sample forecasts, then evaluate MAE (sample median) and CRPS ─────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
# Use the empirical median of the predictive samples as the point forecast.
# DeepAR's default `mode="prediction"` returns the *mean* of samples, which
# is biased upward for right-skewed distributions (LogNormal mean
# = exp(μ+σ²/2) ≫ exp(μ) = median). The median makes MAE comparable across
# Normal / Student-t / LogNormal.
print("\n" + "─" * 60)
print("📏  Evaluate forecasts (MAE of sample median, CRPS over samples)")
print("─" * 60)

N_SAMPLES = 200   # ensemble size — 200+ is enough for stable CRPS estimates

print(f"Sampling {N_SAMPLES} forecasts per series from each model...")
print(f"  baseline (last-value, point prediction): MAE = CRPS = {mae_baseline:.4f}\n")

forecast_cache = {}    # store samples for the next cell (forecast plots)
for label, model in models.items():
    model.eval()
    preds = model.predict(
        test_dl, mode="samples", n_samples=N_SAMPLES,
        trainer_kwargs=dict(accelerator="auto", devices=1,
                            enable_progress_bar=False, logger=False),
        return_y=True, batch_size=2,
    )
    samples = preds.output.cpu()         # (B, T, N_SAMPLES)
    target  = preds.y[0].cpu()           # (B, T)
    median  = samples.median(dim=-1).values            # (B, T)
    mae     = float((median - target).abs().mean())
    # torchmetrics' CRPS expects (batch, ensemble) preds and (batch,) target;
    # flatten the (B, T) → B*T batch axis to evaluate per-(series, horizon).
    crps_mean = float(crps_fn(
        samples.reshape(-1, N_SAMPLES),
        target.reshape(-1),
    ))
    rel_mae  = (mae_baseline - mae)       / mae_baseline * 100
    rel_crps = (mae_baseline - crps_mean) / mae_baseline * 100
    print(f"  {label:9s}: MAE  = {mae:.4f}  ({rel_mae:+.1f}% vs. baseline)")
    print(f"  {' ' * 9}  CRPS = {crps_mean:.4f}  ({rel_crps:+.1f}% vs. baseline)\n")
    forecast_cache[label] = dict(samples=samples.numpy(), target=target.numpy())

# ─────────────────────────────────────────────────────────────────────────────────────
# ── Visualize forecasts: context + predictive distributions ──────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("📈  Visualize forecasts: context + predictive distributions")
print("─" * 60)

QUANTILES_TO_PLOT = [0.025, 0.10, 0.50, 0.90, 0.975]
LIKELIHOOD_DISPLAY = {
    "Normal":     "Gaussian",
    "Student-t":  r"Student-$t$",
    "Log-Normal": "LogNormal",
}
CONTEXT_SHOW = BLOCKS_PER_DAY * 3     # show last 3 days of encoder context

# Pull encoder targets from test_dl so we can plot context next to each forecast.
encoder_targets = []
for x, _ in test_dl:
    encoder_targets.append(x["encoder_target"].cpu())
encoder_target_all = torch.cat(encoder_targets, dim=0).numpy()    # (B, CONTEXT)
context_show = min(CONTEXT_SHOW, encoder_target_all.shape[1])

# Pick 3 validation series spanning low / median / high target magnitude.
y_targets = forecast_cache[next(iter(forecast_cache))]["target"]    # (B, T)
sort_idx = np.argsort(y_targets.mean(axis=1))
plot_indices = [
    int(sort_idx[len(sort_idx) // 6]),       # ~16th percentile
    int(sort_idx[len(sort_idx) // 2]),       # median
    int(sort_idx[5 * len(sort_idx) // 6]),   # ~83rd percentile
]
print(f"Plotting series indices {plot_indices}  "
      f"(showing last {context_show} blocks of context)")

likelihoods = list(forecast_cache.keys())
n_rows = len(plot_indices)
n_cols = len(likelihoods)
horizon_len = y_targets.shape[1]

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(3.5 * n_cols + 0.5, 2.0 * n_rows + 0.6),
    sharey="row", sharex="col", layout="constrained", squeeze=False,
)

for r, idx in enumerate(plot_indices):
    target  = y_targets[idx]                                          # (T,)
    context = encoder_target_all[idx, -context_show:]                 # last K blocks
    t_context = np.arange(-context_show, 0)
    t_horizon = np.arange(horizon_len)
    for c, label in enumerate(likelihoods):
        samples = forecast_cache[label]["samples"][idx]               # (T, N_SAMPLES)
        q = np.quantile(samples, QUANTILES_TO_PLOT, axis=-1)          # (5, T)
        ax = axes[r, c]
        # Encoder context (real past)
        ax.plot(t_context, context, "-", color="black", lw=0.9, alpha=0.7,
                label="context" if (r == 0 and c == 0) else None)
        # Forecast bands
        ax.fill_between(t_horizon, q[0], q[4], color="C0", alpha=0.18, lw=0, label="95% PI")
        ax.fill_between(t_horizon, q[1], q[3], color="C0", alpha=0.32, lw=0, label="80% PI")
        ax.plot(t_horizon, q[2], color="C0", lw=1.4, label="median")
        # Actual future values (held out)
        ax.plot(t_horizon, target, "o-", color="black", lw=0.8, ms=1.5, label="actual")
        # Forecast-start cue
        ax.axvline(-0.5, color="0.5", lw=0.8, ls="--", alpha=0.7)
        if r == 0:
            ax.set_title(LIKELIHOOD_DISPLAY[label])
        if c == 0:
            ax.set_ylabel(f"series #{idx}")
        if r == n_rows - 1:
            ax.set_xlabel("block (0 = forecast start)")
        ax.grid(True, alpha=0.3)

axes[0, -1].legend(loc="upper right", framealpha=0.9, fontsize=7)
fig.suptitle("Predictive distributions per DeepAR likelihood — context + forecast",
             fontsize=10, y=1.02)

fig.savefig(PLOT_DIR + "/fig_train_forecasts.pdf", bbox_inches="tight")
print("✓ saved → " + PLOT_DIR + "/fig_train_forecasts.pdf")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────────────
# ── Wrap up ──────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("✅  Done with DeepAR training and evaluation")
print(f"Total runtime: {(time.time() - start_time) / 60:.1f} minutes")
print("─" * 60)
