"""DeepAR-specific Section 5.3.2 / Appendix D.3.1 validation pipeline.

End-to-end pipeline that turns a trained DeepAR checkpoint into the
sample-based estimator validation results in Figure D.4. The six steps are
numbered below and correspond directly to the section-banner comments:

    Step 1.  build_inputs           — pick I forecast origins (instances) and
                                       K background origins, enumerate all
                                       2^P coalitions, write inputs.npz.
    Step 2.  draw_nested_samples    — run DeepAR on the imputed (I, n_coal, K)
                                       grid, get (N_max trajectories, raw
                                       distribution params) per row.
    Step 3.  compute_reference      — analytical Level-2 / Level-3 reference
                                       value functions, evaluated from DeepAR's
                                       own per-step parametric output.
    Step 4.  evaluate_estimators    — run the four sample-based estimators
                                       (mvn, mixture_copula, kde_copula, knn)
                                       on the trajectory cache across the
                                       N_grid trajectory budgets.
    Step 5.  aggregate_mae          — per-instance MAE on v(S) (5a, 5b) and
                                       on the resulting Shapley value phi_p
                                       (5c).
    Step 6.  plot_fig3_combined     — Figure D.4 (2 × N_dist grid of scale
                                       vs. topology errors).

All caches live under ``<results_dir>/<distribution>/`` with per-seed file
suffixes; meta JSON files store the checkpoint hash so a different DeepAR
training automatically invalidates the cache. Background origins are sampled
i.i.d. across all 100 series and across the training period — this matches
the standard marginal-Shapley imputation convention (Sec. 4.2 of the paper).
"""
from __future__ import annotations

import os
import math
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import json
import torch

from .utils_datasets import (
    build_feature_groups,
    build_feature_vec,
    build_completed_inputs,
)
from .imputer import DeepARLogNormalImputer, DeepARNormalImputer, DeepARStudentTImputer
from .estimators import (
    KDECopulaImputer,
    KNNImputer,
    MVNSampleImputer,
    MixtureCopulaImputer,
    evaluate_on_cached_samples
)

# =============================================================================
#                              Step 1: build_inputs
# =============================================================================

def build_inputs(
    df_long: pd.DataFrame,
    training_cutoff: int,
    *,
    CONTEXT: int,
    T: int,
    P_FEAT: int,
    BLOCKS_PER_DAY: int,
    I: int,
    K: int,
    seed_bg: int = 0,
    feature_groups: list[list[int]] | None = None,
    results_dir: str = "../results/53",
) -> dict:
    """Pick I local instances and K background imputations, enumerate all 2^P
    coalitions, and save to results_dir/inputs.npz (shared across distributions).

    Sampling convention (matches Sec. 4.2 of the paper, marginal Shapley):

    * **Instances** ``x_explain`` are drawn i.i.d. from the *validation/test*
      window (``time_idx > training_cutoff``), across all available series.
      Each instance is a single forecast origin: its 84 history blocks plus
      the (hour, weekday, month) at the prediction edge.

    * **Backgrounds** ``bg`` are drawn i.i.d. from the *training* window
      (``time_idx <= training_cutoff``), again across all series. No filter
      on calendar features — backgrounds are uniform-random origins. This is
      the standard marginal-Shapley convention (Lundberg & Lee 2017; Aas et
      al. 2021): the Shapley expectation over out-of-coalition features is
      approximated by averaging over the unconditional marginal distribution
      of feature vectors. It does *not* try to "match similar days" — that
      would be a conditional-Shapley approach (mentioned as alternative in
      Sec. 4.2 of the paper, not implemented here).

    The disjointness between instance and background windows ensures the
    imputed Frankenstein histories never reuse the explained origin itself.

    Returns a dict with:
        x_explain      : (I, P_FEAT)
        bg             : (K, P_FEAT)
        coalitions     : (2^P, P) bool — coalition of feature groups
        feature_groups : list[list[int]]
        meta           : dict
    """
    if feature_groups is None:
        feature_groups = build_feature_groups(CONTEXT, blocks_per_day=BLOCKS_PER_DAY)

    rng = np.random.default_rng(seed_bg)

    # --- Pool: validation period only, drop edge rows that lack full encoder context.
    test_start = training_cutoff + 1
    test_end   = df_long["time_idx"].max() - T
    pool = df_long[
        (df_long["time_idx"] >= test_start) &
        (df_long["time_idx"] <= test_end)
    ][["series_id", "time_idx"]].drop_duplicates()

    # --- I local instances (stratified by hour-of-day for diversity).
    pool = pool.sample(frac=1.0, random_state=seed_bg)  # shuffle deterministically
    inst_rows: list[np.ndarray] = []
    inst_series: list[str] = []
    inst_tidx: list[int] = []
    for _, row in pool.iterrows():
        if len(inst_rows) >= I:
            break
        try:
            inst_rows.append(build_feature_vec(df_long, row["series_id"], int(row["time_idx"]),
                                                context=CONTEXT))
            inst_series.append(str(row["series_id"]))
            inst_tidx.append(int(row["time_idx"]))
        except AssertionError:
            continue
    if len(inst_rows) < I:
        raise RuntimeError(f"Only found {len(inst_rows)}/{I} instances with full context.")
    x_explain = np.stack(inst_rows)                                              # (I, P_FEAT)
    instance_series_id = np.array(inst_series, dtype=object)                     # (I,)
    instance_time_idx  = np.array(inst_tidx,  dtype=np.int64)                    # (I,)

    # --- K background rows from the *training* period (so background ≠ explained).
    bg_pool = df_long[
        (df_long["time_idx"] >= CONTEXT) &
        (df_long["time_idx"] <= training_cutoff)
    ][["series_id", "time_idx"]].drop_duplicates().sample(n=K, random_state=seed_bg + 1)
    bg_rows: list[np.ndarray] = []
    for _, row in bg_pool.iterrows():
        bg_rows.append(build_feature_vec(df_long, row["series_id"], int(row["time_idx"]),
                                          context=CONTEXT))
    bg = np.stack(bg_rows)                                                        # (K, P_FEAT)

    # --- All 2^P coalitions over feature *groups*.
    p_groups = len(feature_groups)
    coalitions = np.array(
        [[(s >> j) & 1 for j in range(p_groups)] for s in range(2 ** p_groups)],
        dtype=bool,
    )                                                                              # (2^P, P)

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "inputs.npz",
        x_explain=x_explain, bg=bg, coalitions=coalitions,
        feature_groups=np.array(feature_groups, dtype=object),
        instance_series_id=instance_series_id,
        instance_time_idx=instance_time_idx,
    )
    meta = dict(I=I, K=K, p_groups=p_groups, T=T, CONTEXT=CONTEXT,
                P_FEAT=P_FEAT, BLOCKS_PER_DAY=BLOCKS_PER_DAY, seed_bg=seed_bg)
    _save_meta(out_dir / "inputs_meta.json", meta)
    print(f"✓ inputs saved → {out_dir/'inputs.npz'}\n -->"
          f"x_explain {x_explain.shape}, bg {bg.shape}, coalitions {coalitions.shape},",
          f"feature groups {len(feature_groups)}")

    return dict(
        x_explain=x_explain, bg=bg, coalitions=coalitions,
        feature_groups=feature_groups, meta=meta, dir=out_dir,
        instance_series_id=instance_series_id,
        instance_time_idx=instance_time_idx,
    )

# =============================================================================
#                       Step 2: draw_nested_samples
# =============================================================================

def pl_seed_everything(seed: int) -> None:
    """Seed Python, numpy, torch (CPU + CUDA) for the trajectory draw."""
    import random
    import lightning.pytorch as pl
    pl.seed_everything(seed, verbose=False)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def draw_nested_samples(
    model,
    inputs: dict,
    *,
    T: int,
    P: int,
    distribution: str,
    N_max: int,
    seed: int,
    make_dataloader,
    predict_batch_size: int = 64,
    results_dir: str = "../results/53",
    ckpt_path: str | None = None,
    save: bool = True,
) -> dict:
    """Draw N_max trajectory samples for every (instance, coalition, background) row.

    With ``save=True`` (default), persists/loads ``samples_seed{seed}.npy`` of
    shape (I, n_coal, K, N_max, T) and ``raw_seed{seed}.npy`` of shape
    (I, n_coal, K, N_max, T, n_params).

    With ``save=False`` no disk I/O happens (no cache read, no write); the
    tensors are only returned in memory. Used for instance-batched runs where
    the full samples tensor never has to materialize on disk.
    """
    if save:
        out_dir = _cache_root(distribution, results_dir)
        s_path  = out_dir / f"samples_seed{seed}.npy"
        r_path  = out_dir / f"raw_seed{seed}.npy"
        m_path  = out_dir / f"samples_seed{seed}_meta.json"

        expected_meta = dict(
            N_max=N_max, seed=seed, distribution=distribution,
            ckpt_sha256=_sha256(ckpt_path) if ckpt_path else None,
            T=T, P=P,
        )
        if s_path.exists() and r_path.exists():
            _check_meta(m_path, expected_meta)
            print(f"✓ {distribution} seed={seed}: cached samples loaded ({s_path})")
            return dict(samples=np.load(s_path, mmap_mode="r"),
                        raw=np.load(r_path, mmap_mode="r"),
                        meta=json.loads(m_path.read_text()))
    else:
        expected_meta = None

    pl_seed_everything(seed)

    x_explain  = inputs["x_explain"]
    bg         = inputs["bg"]
    coalitions = inputs["coalitions"]
    feature_groups = inputs["feature_groups"]
    p_feat = x_explain.shape[1]
    I, K   = x_explain.shape[0], bg.shape[0]
    n_coal = coalitions.shape[0]

    completed_flat = build_completed_inputs(
        x_explain, bg, coalitions, feature_groups, p_feat,
    )                                                          # (I*n_coal*K, P_FEAT)

    ImpCls = _imputer_class(distribution)
    imputer = ImpCls(
        model=model, make_dataloader=make_dataloader, n_samples=N_max,
        sampler=None, x_explain=x_explain, T=T,
        feature_groups=feature_groups, predict_batch_size=predict_batch_size,
    )
    samples_flat, raw_flat = imputer._predict(completed_flat)
    # Shapes: samples_flat (I*n_coal*K, N_max, T), raw_flat (..., T, n_params).

    samples = samples_flat.reshape(I, n_coal, K, N_max, T)
    raw     = raw_flat.reshape(I, n_coal, K, N_max, T, raw_flat.shape[-1])

    if save:
        np.save(s_path, samples)
        np.save(r_path, raw)
        _save_meta(m_path, expected_meta)
        print(f"✓ samples → {s_path} (shape {samples.shape}, "
              f"{samples.nbytes/1e6:.0f} MB), raw → {r_path}")

    return dict(samples=samples, raw=raw, meta=expected_meta)


# =============================================================================
#                       Step 3: compute_reference
# =============================================================================

def compute_reference(
    samples: np.ndarray,
    raw: np.ndarray,
    *,
    distribution: str,
    results_dir: str = "../results/53",
    seed: int = 0,
    save: bool = True,
) -> dict:
    """Density-aware DeepAR reference value functions, averaged over K backgrounds.

    Returns
    -------
    {'v_ref_L2': (I, n_coal, T), 'v_ref_L3': (I, n_coal)}
    """
    if save:
        out_dir = _cache_root(distribution, results_dir)
        cache   = out_dir / f"refs_seed{seed}.npz"
        if cache.exists():
            npz = np.load(cache)
            print(f"✓ {distribution} seed={seed}: cached reference loaded ({cache})")
            return {"v_ref_L2": npz["v_ref_L2"], "v_ref_L3": npz["v_ref_L3"]}

    ImpCls = _imputer_class(distribution)
    # Instantiate stub: only `_compute_L2`/`_compute_L3` are called, not `_predict`.
    stub = ImpCls(
        model=None, make_dataloader=None, n_samples=samples.shape[3],
        sampler=None, x_explain=np.zeros((1, 1)), T=samples.shape[-1],
    )

    I, n_coal, K = samples.shape[:3]
    # Flatten (I, n_coal, K) → batch axis n=I*n_coal*K so the existing
    # _compute_L2 (designed for n×M×T inputs) just works.
    s_flat = samples.reshape(I * n_coal * K, samples.shape[3], samples.shape[4])
    r_flat = raw    .reshape(I * n_coal * K, raw.shape[3], raw.shape[4], raw.shape[5])

    L2_flat = stub._compute_L2((s_flat, r_flat))                     # (I*n_coal*K, T)
    L2 = L2_flat.reshape(I, n_coal, K, -1).mean(axis=2)              # (I, n_coal, T)
    # L3 via chain rule (Prop. 1) — equivalent to averaging L3_flat over K,
    # but avoids the extra _compute_L3 pass.
    L3 = L2.sum(axis=-1)                                              # (I, n_coal)

    out = {"v_ref_L2": L2, "v_ref_L3": L3}
    if save:
        np.savez(cache, **out)
        print(f"✓ reference → {cache} (L2 {L2.shape}, L3 {L3.shape})")
    return out


def save_reference(out: dict, *, distribution: str, results_dir: str, seed: int) -> None:
    """Persist a reference dict (returned by ``compute_reference(..., save=False)``).

    Used by instance-batched callers that accumulate over batches and write
    once at the end of a (dist, seed) run.
    """
    cache = _cache_root(distribution, results_dir) / f"refs_seed{seed}.npz"
    np.savez(cache, **out)
    print(f"    ✓ reference → {cache} (L2 {out['v_ref_L2'].shape}, L3 {out['v_ref_L3'].shape})")

# =============================================================================
#                       Step 4: evaluate_estimators
# =============================================================================

def evaluate_estimators(
    samples: np.ndarray,
    raw: np.ndarray,
    *,
    distribution: str,
    estimators: dict[str, type],
    N_grid: list[int],
    seed: int,
    results_dir: str = "../results/53",
    save: bool = True,
    knn_k: int = 5,
    n_jobs: int = 1,
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """Slice cache to each N in N_grid, run each estimator, average over K.

    Returns nested dict: {estimator: {N: {'L1': (I, n_coal, T), 'L2': ..., 'L3': (I, n_coal)}}}.

    n_jobs : int
        Parallelises the per-row inner loop in each estimator across processes
        (forwarded to `evaluate_on_cached_samples`).
    """
    out_dir = _cache_root(distribution, results_dir) if save else None
    I, n_coal, K, _, T_local = samples.shape

    out: dict[str, dict[int, dict[str, np.ndarray]]] = {est_name: {} for est_name in estimators}

    for est_name, est_cls in estimators.items():
        for N in N_grid:
            if save:
                cache = out_dir / f"est_{est_name}_seed{seed}_N{N}.npz"
                if cache.exists():
                    npz = np.load(cache)
                    out[est_name][N] = {k: npz[k] for k in ("L1", "L2", "L3")}
                    continue

            s_slice = samples[:, :, :, :N, :]                        # (I, n_coal, K, N, T)
            r_slice = raw    [:, :, :, :N, :, :] if est_cls is not KNNImputer else None

            s_flat = s_slice.reshape(I * n_coal * K, N, T_local)
            r_flat = (r_slice.reshape(I * n_coal * K, N, T_local, raw.shape[-1])
                      if r_slice is not None else None)

            if est_cls is KNNImputer:
                # L1 from KNN is dead work in this pipeline (only L2/L3 are
                # aggregated downstream); skipping it saves T extra univariate
                # cKDTree builds per row.
                kwargs = {"k": knn_k, "skip_l1": True}
            elif est_cls is MixtureCopulaImputer:
                kwargs = {"distribution": distribution}
            else:
                kwargs = {}
            L1f, L2f, L3f = evaluate_on_cached_samples(
                s_flat, r_flat, est_cls, T=T_local, n_jobs=n_jobs, **kwargs,
            )

            L1 = L1f.reshape(I, n_coal, K, T_local).mean(axis=2)     # (I, n_coal, T)
            L2 = L2f.reshape(I, n_coal, K, T_local).mean(axis=2)
            L3 = L3f.reshape(I, n_coal, K).mean(axis=2)              # (I, n_coal)

            # Chain-rule assertion (per estimator).
            assert np.allclose(L3, L2.sum(axis=-1), atol=1e-8), \
                f"Chain rule broken: {est_name} N={N}"

            entry = dict(L1=L1, L2=L2, L3=L3)
            out[est_name][N] = entry
            if save:
                np.savez(cache, **entry)

    return out


def save_estimators(
    ests: dict[str, dict[int, dict[str, np.ndarray]]],
    *,
    distribution: str,
    results_dir: str,
    seed: int,
) -> None:
    """Persist estimator tables (returned by ``evaluate_estimators(..., save=False)``)."""
    out_dir = _cache_root(distribution, results_dir)
    for est_name, by_N in ests.items():
        for N, entry in by_N.items():
            np.savez(out_dir / f"est_{est_name}_seed{seed}_N{N}.npz", **entry)


# =============================================================================
#                       Step 5a: aggregate MAE on v(S)
# =============================================================================

def aggregate_mae(
    refs_by_dist_seed: dict[str, dict[int, dict[str, np.ndarray]]],
    ests_by_dist_seed: dict[str, dict[int, dict]],
    *,
    distributions: tuple[str, ...],
    N_grid: list[int],
    seeds: list[int],
    estimators: tuple[str, ...],
    out_path: str = "../results/53/mae_table.csv",
) -> pd.DataFrame:
    """Per-instance MAE on v(S) for every (likelihood, estimator, seed, N).

    Returns one row per (likelihood, estimator, seed, instance, N, level): the
    MAE on v(S) is computed *per instance* (averaging over coalitions × horizon
    for L2, over coalitions for L3). MAE matches the metric used in
    :func:`aggregate_shapley_error` so v(S) and \\phi_p errors live in the same
    nats unit and can be compared directly.

    refs_by_dist_seed[distribution][seed] = {'v_ref_L2', 'v_ref_L3'}
    ests_by_dist_seed[distribution][seed] = {est: {N: {'L1','L2','L3'}}}
    """
    rows = []
    for dist in distributions:
        for seed in seeds:
            ref = refs_by_dist_seed.get(dist, {}).get(seed)
            if ref is None:
                continue
            ests = ests_by_dist_seed.get(dist, {}).get(seed, {})
            for est_name in estimators:
                for N in N_grid:
                    if est_name not in ests or N not in ests[est_name]:
                        continue
                    e = ests[est_name][N]
                    for level, ref_arr in (("L2", ref["v_ref_L2"]),
                                           ("L3", ref["v_ref_L3"])):
                        diff = e[level] - ref_arr             # (I, n_coal[, T])
                        # Reduce all axes except the leading instance axis →
                        # one MAE value per instance.
                        per_instance = np.mean(
                            np.abs(diff), axis=tuple(range(1, diff.ndim))
                        )                                      # (I,)
                        for i_idx, mae_i in enumerate(per_instance):
                            rows.append(dict(likelihood=dist, estimator=est_name,
                                             seed=seed, instance=int(i_idx),
                                             N=N, level=level, mae=float(mae_i)))
    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✓ MAE-on-v(S) table → {out_path} ({len(df)} rows)")
    return df


# =============================================================================
#                       Step 5b: aggregate MAE by coalition cardinality
# =============================================================================

def aggregate_mae_by_cardinality(
    refs_by_dist_seed: dict[str, dict[int, dict[str, np.ndarray]]],
    ests_by_dist_seed: dict[str, dict[int, dict]],
    *,
    distributions: tuple[str, ...],
    estimators: tuple[str, ...],
    seeds: list[int],
    n_players: int,
    N: int,
    level: str = "L2",
    out_path: str | None = None,
) -> pd.DataFrame:
    """Per-instance MAE on ``v(S)`` aggregated by coalition cardinality ``|S|``.

    For each (likelihood, estimator, seed, instance, |S|) row, the value is the
    mean absolute error between estimator and reference value functions,
    averaged over (coalitions of that cardinality × horizon for L2). Useful
    for diagnosing where in the coalition lattice estimator bias is exposed —
    on the §5.3 setup the MVN-fit on non-Gaussian likelihoods peaks at the
    middle cardinalities (|S|≈3) and dips at the extremes.

    Parameters
    ----------
    N : int
        Single trajectory budget to slice (typically the largest in N_grid:
        the bias signature is most pronounced once MC noise has shrunk).
    level : "L2" or "L3"
    """
    coalitions = np.arange(2 ** n_players)
    cardinalities = np.array([bin(int(s)).count("1") for s in coalitions])

    rows = []
    for dist in distributions:
        for seed in seeds:
            ref = refs_by_dist_seed.get(dist, {}).get(seed)
            if ref is None:
                continue
            ref_arr = ref["v_ref_L2"] if level == "L2" else ref["v_ref_L3"]
            ests = ests_by_dist_seed.get(dist, {}).get(seed, {})
            for est_name in estimators:
                if est_name not in ests or N not in ests[est_name]:
                    continue
                est_arr = ests[est_name][N][level]                # (I, n_coal[, T])
                diff = np.abs(est_arr - ref_arr)
                # Reduce over T (if present) → (I, n_coal).
                if diff.ndim == 3:
                    diff_per_coal = diff.mean(axis=-1)
                else:
                    diff_per_coal = diff                          # already (I, n_coal)
                # Now aggregate per cardinality bucket: average over coalitions
                # in each bucket → (I, p+1) per-instance per-cardinality MAE.
                for k in range(n_players + 1):
                    mask = cardinalities == k
                    if not mask.any():
                        continue
                    per_inst = diff_per_coal[:, mask].mean(axis=-1)   # (I,)
                    for i_idx, mae_i in enumerate(per_inst):
                        rows.append(dict(likelihood=dist, estimator=est_name,
                                         seed=seed, instance=int(i_idx),
                                         level=level, N=N,
                                         cardinality=int(k),
                                         mae=float(mae_i)))
    df = pd.DataFrame(rows)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"✓ MAE-by-cardinality table → {out_path} ({len(df)} rows)")
    return df


def plot_fig3_per_cardinality(
    df: pd.DataFrame,
    *,
    distributions: tuple[str, ...],
    distribution_display: dict[str, str],
    estimators: tuple[str, ...],
    estimator_display: dict[str, str],
    estimator_colors: dict[str, str],
    estimator_markers: dict[str, str],
    level: str = "L2",
    n_players: int = 6,
    save_path: str = "../plots/fig3_per_cardinality",
):
    """1 × N_dist figure: MAE on v(S) by coalition cardinality |S|.

    Reveals where in the coalition lattice each estimator's bias is exposed.
    The MVN-fit baseline shows a characteristic "tent" peaking at the middle
    cardinalities on non-Gaussian likelihoods; the Gaussian-copula estimators
    are flatter; kNN's pattern depends on dimensionality at full coalition.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    if not distributions:
        raise ValueError("No likelihoods to plot — `distributions` is empty.")

    n_cols = len(distributions)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(3.0 * n_cols + 0.6, 3.2),
        sharex=True, layout="constrained", squeeze=False,
    )
    axes = axes[0]

    for c, dist in enumerate(distributions):
        ax = axes[c]
        sub = df[(df["likelihood"] == dist) & (df["level"] == level)]
        for est in estimators:
            d = sub[sub["estimator"] == est]
            if d.empty:
                continue
            grp = d.groupby("cardinality")["mae"].agg(["mean", "sem"]).reset_index()
            ax.errorbar(
                grp["cardinality"], grp["mean"],
                yerr=grp["sem"].where(grp["sem"].notna(), 0.0),
                color=estimator_colors[est], lw=1.5,
                marker=estimator_markers[est], markersize=6.0,
                markeredgewidth=0.5, markeredgecolor="white",
                ecolor=estimator_colors[est],
                elinewidth=0.9, capsize=2.5, capthick=0.9,
                label=estimator_display[est],
            )
        ax.set_xticks(range(n_players + 1))
        ax.set_xticklabels([str(k) for k in range(n_players + 1)])
        ax.minorticks_off()
        ax.grid(True, axis="y", alpha=0.30)
        ax.grid(True, axis="x", alpha=0.18, lw=0.6)
        ax.set_title(distribution_display.get(dist, dist))
        ax.set_xlabel(r"coalition cardinality $|S|$")

    axes[0].set_ylabel(rf"MAE on $v(S)$  [nats]   ({level})")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(handles),
        bbox_to_anchor=(0.5, -0.10),
        framealpha=0.9, fontsize=8.5,
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path + ".pdf", bbox_inches="tight")
    fig.savefig(save_path + ".png", bbox_inches="tight", dpi=300)
    print(f"✓ figure → {save_path}.pdf")
    return fig



# =============================================================================
#               Step 5c: Shapley-level error (downstream metric)
# =============================================================================
#
# RMSE on v(S) mixes bias and variance. On the *outer* aggregation (Shapley)
# variance averages out across the 2^p coalition lattice while bias compounds
# into systematic per-feature distortion. Reporting Shapley-level MAE gives a
# downstream-faithful view of which estimator the pipeline should prefer.

def _shapley_values_from_table(v: np.ndarray, n_players: int) -> np.ndarray:
    """Exact Shapley values from a v(S) table indexed by binary coalition mask.

    Last axis must be the coalition axis with length 2**n_players, encoded as
    bit j ↔ player j (matching `build_inputs` coalition layout).

    Parameters
    ----------
    v : (..., 2**n_players)
        Value-function table.

    Returns
    -------
    phi : (..., n_players)
        Shapley contribution per player.
    """
    n = n_players
    if v.shape[-1] != 2 ** n:
        raise ValueError(f"v.shape[-1]={v.shape[-1]} ≠ 2**{n}={2**n}")
    n_fact = math.factorial(n)
    cards = np.array([bin(s).count("1") for s in range(2 ** n)], dtype=int)
    # Weight only enters for coalitions that *exclude* the player p, so |S| <= n-1
    # always holds at the lookup; the |S|=n entry is never indexed but must exist
    # for shape, so set it to 0.
    weights = np.array(
        [math.factorial(k) * math.factorial(n - k - 1) / n_fact if k < n else 0.0
         for k in cards]
    )
    phi = np.zeros(v.shape[:-1] + (n,), dtype=v.dtype)
    for p in range(n):
        bit = 1 << p
        S_no_p   = np.array([s for s in range(2 ** n) if not (s & bit)])
        S_with_p = S_no_p | bit
        marg = v[..., S_with_p] - v[..., S_no_p]
        phi[..., p] = (marg * weights[S_no_p]).sum(axis=-1)
    return phi


def _shapley_from_value_table(v: np.ndarray, n_players: int) -> np.ndarray:
    """Wrap `_shapley_values_from_table` for the (I, n_coal[, T]) layout.

    Coalition axis is at index 1 (for both L2 of shape (I, n_coal, T) and
    L3 of shape (I, n_coal)). Returns:
      L3: (I, p)
      L2: (I, T, p)  -- horizon axis preserved
    """
    if v.ndim == 2:
        return _shapley_values_from_table(v, n_players)
    if v.ndim == 3:
        v_T = np.moveaxis(v, 1, -1)                          # (I, T, n_coal)
        return _shapley_values_from_table(v_T, n_players)    # (I, T, p)
    raise ValueError(f"Unsupported v.ndim={v.ndim}")


def aggregate_shapley_error(
    refs_by_dist_seed: dict[str, dict[int, dict[str, np.ndarray]]],
    ests_by_dist_seed: dict[str, dict[int, dict]],
    *,
    distributions: tuple[str, ...],
    N_grid: list[int],
    seeds: list[int],
    estimators: tuple[str, ...],
    n_players: int,
    out_path: str = "../results/53/shapley_error_table.csv",
) -> pd.DataFrame:
    """Per-instance MAE on Shapley vector for every (dist, est, seed, N, level).

    Returns one row per (likelihood, estimator, seed, instance, N, level): the
    MAE on phi is computed *per instance* (averaging over players × horizon for
    L2, over players for L3). Aggregating across (instance × seed) in the plot
    yields bands that combine sample-noise (seeds) and test-set variability
    (instances).
    """
    rows = []
    for dist in distributions:
        for seed in seeds:
            ref = refs_by_dist_seed.get(dist, {}).get(seed)
            if ref is None:
                continue
            phi_ref_L2 = _shapley_from_value_table(ref["v_ref_L2"], n_players)  # (I, T, p)
            phi_ref_L3 = _shapley_from_value_table(ref["v_ref_L3"], n_players)  # (I, p)

            ests = ests_by_dist_seed.get(dist, {}).get(seed, {})
            for est_name in estimators:
                for N in N_grid:
                    if est_name not in ests or N not in ests[est_name]:
                        continue
                    e = ests[est_name][N]
                    for level, phi_ref in (("L2", phi_ref_L2), ("L3", phi_ref_L3)):
                        phi_est = _shapley_from_value_table(e[level], n_players)
                        diff = np.abs(phi_est - phi_ref)            # (I, T, p) or (I, p)
                        per_instance = np.mean(
                            diff, axis=tuple(range(1, diff.ndim))
                        )                                            # (I,)
                        for i_idx, mae_i in enumerate(per_instance):
                            rows.append(dict(likelihood=dist, estimator=est_name,
                                             seed=seed, instance=int(i_idx),
                                             N=N, level=level,
                                             mae_phi=float(mae_i)))
    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✓ Shapley-error table → {out_path} ({len(df)} rows)")
    return df


# =============================================================================
#               Step 6: combined 2 x N_dist figure (scale vs. topology)
# =============================================================================
#
# Layout:
#     rows = metric (top: RMSE on v(S),  bottom: MAE on phi_p)
#     cols = likelihood (in caller-provided `distributions` order)
# Only ONE level (`L2` for the main figure; pass `level="L3"` for the appendix
# version).
#
# Default y-axis is linear everywhere — all panels span <11x range in this
# experiment, and linear lets the small spreads above the shared constant
# offsets stay visible. `yscale="mixed"` reverts to log for Gaussian RMSE +
# all Shapley panels, linear for non-Gaussian RMSE panels (the historical
# motivation: bias-floor structure compressed by log on heavy-tailed data).

def plot_fig3_combined(
    mae_v_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    *,
    distributions: tuple[str, ...],
    distribution_display: dict[str, str],
    estimators: tuple[str, ...],
    estimator_display: dict[str, str],
    estimator_colors: dict[str, str],
    estimator_markers: dict[str, str],
    level: str = "Level 2",
    save_path: str = "../plots/fig3_combined",
    yscale: str = "linear",
):
    """Combined 2 x N_dist narrative figure.

    Parameters
    ----------
    mae_v_df, shap_df : long-format DataFrames produced by ``aggregate_mae``
        and ``aggregate_shapley_error`` respectively.
    distributions : tuple[str, ...]
        Which likelihoods to include as columns (in column order).
    distribution_display, estimator_display, estimator_colors, estimator_markers
        Per-key style dicts owned by the caller (notebook).
    level : "L2" or "L3"
        Which entropy level to render. L2 → main figure; L3 → appendix.
    yscale : {"linear", "log", "mixed"}
        Y-axis scale.
        - "linear" (default) — every panel linear.
        - "log" — every panel log.
        - "mixed" — log for Gaussian RMSE and all Shapley panels; linear for
          non-Gaussian RMSE panels. Historically used to reveal the small
          spread above the shared constant offset on heavy-tailed data.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    if not distributions:
        raise ValueError("No likelihoods to plot — `distributions` is empty.")

    n_cols = len(distributions)
    fig, axes = plt.subplots(
        2, n_cols, figsize=(3.4 * n_cols + 0.8, 5.4),
        sharex="col", layout="constrained", squeeze=False,
    )

    # Per-panel y-scale resolution.
    def _scale_for(row: int, dist: str) -> str:
        if yscale in ("linear", "log"):
            return yscale
        if yscale == "mixed":
            # rmse panels (row=0) get linear for non-Gaussian likelihoods (heavy-
            # tailed → constant offset dominates on log); Gaussian and all
            # Shapley panels stay log.
            if row == 0 and dist != "normal":
                return "linear"
            return "log"
        raise ValueError(f"yscale must be 'linear', 'log', or 'mixed'; got {yscale!r}")

    metric_specs = [
        # (row, df, value_col)
        (0, mae_v_df, "mae"),
        (1, shap_df,  "mae_phi"),
    ]
    # Pin x-ticks to actual N_grid values (powers-of-10 default ticks miss
    # e.g. N=250, 2000); collected from the union across the two metric tables.
    N_values = sorted(set(int(n) for df, _ in [(mae_v_df, None), (shap_df, None)]
                          for n in df["N"].unique()))

    for r, df, value_col in metric_specs:
        for c, dist in enumerate(distributions):
            ax = axes[r, c]
            sub = df[(df["likelihood"] == dist) & (df["level"] == level)]
            for est in estimators:
                d = sub[sub["estimator"] == est]
                if d.empty:
                    continue
                # Aggregate across (instance × seed) replicates per N. With
                # per-instance rows in the table, n = I * |seeds| (e.g. 75 for
                # I=25, 3 seeds) — sem of the mean is informative and not
                # overwhelming like std would be.
                grp = d.groupby("N")[value_col].agg(["mean", "sem"]).reset_index()
                ax.errorbar(
                    grp["N"], grp["mean"],
                    yerr=grp["sem"].where(grp["sem"].notna(), 0.0),
                    color=estimator_colors[est], lw=1.4,
                    marker=estimator_markers[est], markersize=5.5,
                    markeredgewidth=0.5, markeredgecolor="white",
                    ecolor=estimator_colors[est],
                    elinewidth=0.9, capsize=2.5, capthick=0.9,
                    label=estimator_display[est],
                )
            ax.set_xscale("log")
            ax.set_yscale(_scale_for(r, dist))
            # Every N_grid value gets a labelled major tick + vertical gridline.
            ax.set_xticks(N_values)
            ax.set_xticklabels([str(n) for n in N_values])
            ax.minorticks_off()
            ax.set_xlim(N_values[0] / 1.15, N_values[-1] * 1.15)
            ax.grid(True, axis="y", alpha=0.30)
            ax.grid(True, axis="x", alpha=0.18, lw=0.6)

    # Column titles (top row only).
    for c, dist in enumerate(distributions):
        axes[0, c].set_title(distribution_display.get(dist, dist))

    # Row labels via y-axis labels on the left column.
    axes[0, 0].set_ylabel(rf"MAE on value function $v(S)$")
    axes[1, 0].set_ylabel(rf"MAE on Shapley value $\phi_p$")

    # X labels only on bottom row.
    for c in range(n_cols):
        axes[1, c].set_xlabel(r"Samples $N$")

    # In "mixed" mode the linear panels are non-default; annotate why.
    if yscale == "mixed":
        for c, dist in enumerate(distributions):
            if _scale_for(0, dist) == "linear":
                axes[0, c].text(
                    0.04, 0.96,
                    "linear: spread above\nshared constant offset",
                    transform=axes[0, c].transAxes,
                    fontsize=7, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="0.7", alpha=0.85),
                )

    # Single global legend at the bottom.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(handles),
        bbox_to_anchor=(0.5, -0.08),
        framealpha=0.9, fontsize=8.5,
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path + ".pdf", bbox_inches="tight")
    fig.savefig(save_path + ".png", bbox_inches="tight", dpi=300)
    print(f"✓ figure → {save_path}.pdf")
    return fig


# =============================================================================
#                              Internal helpers
# =============================================================================

def _save_meta(path: Path, meta: dict) -> None:
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

def _cache_root(distribution: str, results_dir: str) -> Path:
    p = Path(results_dir) / distribution
    p.mkdir(parents=True, exist_ok=True)
    return p

def _sha256(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _check_meta(path: Path, expected: dict, *, strict_keys=("ckpt_sha256",)) -> None:
    if not path.exists():
        return
    with open(path) as f:
        got = json.load(f)
    for k in strict_keys:
        if k in expected and got.get(k) != expected[k]:
            raise RuntimeError(
                f"Cache meta mismatch at {path}: {k}={got.get(k)} but expected {expected[k]}.\n"
                f"Delete {path.parent} or change the checkpoint to refresh."
            )

def _imputer_class(distribution: str):
    if distribution == "normal":
        return DeepARNormalImputer
    if distribution == "studentt":
        return DeepARStudentTImputer
    if distribution == "lognormal":
        return DeepARLogNormalImputer
    raise ValueError(f"Unknown distribution {distribution!r}")
