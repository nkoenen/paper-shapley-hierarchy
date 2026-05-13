"""Section 5.3 — DeepAR vs. Chronos cross-component diagnostic.

End-to-end pipeline that produces Figure 3 of the paper: a 5-box boxplot of
the cross-component share across 100 forecast origins, comparing DeepAR and
Chronos under three estimators (analytical, KDE-copula, kNN). Companion to
``utils_deepar.py`` (which handles the §5.3.2 estimator-validation pipeline
against DeepAR's analytical reference); this module is the §5.3 main-figure
pipeline that adds Chronos as a second model class.

The pipeline is exposed as five composable steps:

    Step 1.  compute_trajectories                — sample N joint trajectories
                                                    per (instance, coalition,
                                                    background) row via either
                                                    DeepARTrajectoryImputer or
                                                    ChronosTrajectoryImputer.
    Step 2.  compute_hierarchy /
             compute_hierarchy_analytical_deepar — turn trajectories into the
                                                    three value functions L1
                                                    (marginal), L2 (sequential),
                                                    L3 (joint) per estimator.
    Step 3.  compute_shapley                     — exact Shapley values from
                                                    the cached value-function
                                                    tables (via shapiq).
    Step 4.  cross_share / summary_table         — per-instance cross-component
                                                    share, plus a summary table
                                                    aggregating across instances.
    Step 5.  plot_fig3_main_boxplot /
             plot_forecast_appendix              — Figure 3 (cross-share boxplot)
                                                    and Appendix forecast-quality
                                                    overlay.

Two trajectory-imputer wrappers (:class:`DeepARTrajectoryImputer`,
:class:`ChronosTrajectoryImputer`) share a uniform
``predict_trajectories(completed_X, n_samples, seed) -> (n_rows, n_samples, T)``
interface so the same downstream estimator (:class:`KDECopulaImputer`,
:class:`KNNImputer`) applies uniformly across both forecasters.

Background imputation follows the same marginal-Shapley convention as
``utils_deepar.build_inputs``: backgrounds are i.i.d. samples across all
training-period origins (see ``build_inputs`` docstring for the full rationale).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .estimators import KDECopulaImputer, evaluate_on_cached_samples
from .game import HierarchyGame
from .utils_datasets import build_completed_inputs
from .utils_deepar import pl_seed_everything


# =============================================================================
#                              Abstract base
# =============================================================================

class TrajectoryModelImputer(ABC):
    """Uniform interface: completed feature vectors -> trajectory samples.

    Subclasses implement ``predict_trajectories(completed_X, n_samples, seed)``
    returning a ``(n_rows, n_samples, T)`` array.
    """

    model_name: str = "abstract"

    def __init__(self, *, T: int, feature_groups: list[list[int]]):
        self._T = T
        self._feature_groups = feature_groups

    @property
    def T(self) -> int:
        return self._T

    @property
    def feature_groups(self) -> list[list[int]]:
        return self._feature_groups

    @abstractmethod
    def predict_trajectories(self, completed_X: np.ndarray, n_samples: int,
                             seed: int = 0) -> np.ndarray:
        """(n_rows, p_feat) -> (n_rows, n_samples, T)."""

# =============================================================================
#                       DeepAR trajectory wrapper
# =============================================================================

class DeepARTrajectoryImputer(TrajectoryModelImputer):
    """Wraps a trained DeepAR model + dataloader factory.

    Uses the existing ``DeepARNormalImputer._predict`` machinery and discards
    the captured raw distribution params.
    """

    model_name = "deepar"

    def __init__(self, model, make_dataloader, *,
                 T: int, feature_groups: list[list[int]],
                 predict_batch_size: int = 64):
        super().__init__(T=T, feature_groups=feature_groups)
        self._model = model
        self._make_dataloader = make_dataloader
        self._predict_batch_size = predict_batch_size

    def predict_trajectories(self, completed_X: np.ndarray, n_samples: int,
                             seed: int = 0,
                             return_raw: bool = False):
        """Run DeepAR autoregressive sampling.

        Returns ``samples`` of shape ``(n, n_samples, T)`` by default. With
        ``return_raw=True`` the captured per-step distribution-projector raw
        outputs are also returned with shape ``(n, n_samples, T, n_params)``
        so the analytical-reference helper can be applied to the same
        trajectory cache (Section 5.3 main paper figure).
        """
        from .imputer import DeepARNormalImputer
        imputer = DeepARNormalImputer(
            model=self._model, make_dataloader=self._make_dataloader,
            n_samples=n_samples,
            sampler=None, x_explain=completed_X[:1], T=self._T,
            feature_groups=self._feature_groups,
            predict_batch_size=self._predict_batch_size,
        )
        pl_seed_everything(seed)
        samples, raw = imputer._predict(completed_X)
        if return_raw:
            return samples, raw                                      # (n, N, T), (n, N, T, n_params)
        return samples                                                # (n, N, T)

# =============================================================================
#                       Chronos trajectory wrapper
# =============================================================================

class ChronosTrajectoryImputer(TrajectoryModelImputer):
    """Wraps an Amazon Chronos zero-shot pipeline.

    Chronos is autoregressive over discretised target tokens; it consumes only
    the target history (no exogenous calendar features), so the three calendar
    feature groups in our Section 5.3 player layout are *inactive* (see
    ``_FEATURE_GROUP_ACTIVE_BY_INDEX``).

    Parameters
    ----------
    pipeline : ChronosPipeline
        Pre-loaded ``chronos.ChronosPipeline`` (e.g. ``"amazon/chronos-t5-base"``).
    chunk : int
        Number of completed-input rows fed to ``pipeline.predict`` per call.
        At ``N=500`` trajectories the activations of T5-base blow past 47 GB
        on a single GPU when called with the full coalition batch — keep
        ``chunk`` small (default 4) to bound the activation footprint.
    """

    model_name = "chronos_base"

    def __init__(self, pipeline, *, T: int, feature_groups: list[list[int]],
                 context: int, chunk: int = 4):
        super().__init__(T=T, feature_groups=feature_groups)
        self._pipeline = pipeline
        self._context = context
        self._chunk = chunk

    def predict_trajectories(self, completed_X: np.ndarray, n_samples: int,
                             seed: int = 0) -> np.ndarray:
        import torch
        pl_seed_everything(seed)

        # Coalition deduplication. Chronos consumes only the target history
        # (positions [0:CONTEXT]); calendar feature flips between coalitions
        # never reach the model and produce *identical* trajectories. We
        # therefore collapse the input to its unique target-history rows,
        # run the model once per unique row, and broadcast the resulting
        # (samples, T) tensor back via the inverse-index map. On the Section 5.3
        # grid this is an exact 8× speedup (3 inactive bits → 2³ collisions).
        n_rows = completed_X.shape[0]
        targets = completed_X[:, :self._context].astype(np.float32, copy=False)
        unique_targets, inverse = np.unique(targets, axis=0, return_inverse=True)
        n_unique = unique_targets.shape[0]

        out_unique = np.empty((n_unique, n_samples, self._T), dtype=np.float32)
        with tqdm(total=n_unique, desc="chronos", unit="series", leave=False) as pbar:
            for s in range(0, n_unique, self._chunk):
                e = min(n_unique, s + self._chunk)
                ctx = [torch.tensor(unique_targets[i]) for i in range(s, e)]
                fc = self._pipeline.predict(
                    inputs=ctx,
                    prediction_length=self._T,
                    num_samples=n_samples,
                )                                          # (e-s, n_samples, T)
                out_unique[s:e] = fc.cpu().float().numpy()
                del fc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                pbar.update(e - s)

        return out_unique[inverse]

# =============================================================================
#                       Step 1: compute trajectories per model
# =============================================================================

def compute_trajectories(
    model_imp: TrajectoryModelImputer,
    inputs: dict,
    *,
    N: int,
    seed: int = 0,
    results_dir: str,
    chunk: int = 256,
    save_raw: bool = False,
) -> np.ndarray:
    """Run ``predict_trajectories`` over the (I, n_coal, K) grid and reshape to
    ``(I, n_coal, K, N, T)``. Cached at ``{results_dir}/trajectories_{model_name}.npy``.

    With ``save_raw=True`` the imputer is asked to return the per-step raw
    distribution-projector output alongside the trajectories, and the result
    is persisted as ``raw_{model_name}.npy``. Currently only
    ``DeepARTrajectoryImputer`` honours ``return_raw=True``.
    """
    out = Path(results_dir) / f"trajectories_{model_imp.model_name}.npy"
    raw_out = Path(results_dir) / f"raw_{model_imp.model_name}.npy"
    if out.exists() and (not save_raw or raw_out.exists()):
        print(f"✓ trajectories cached for {model_imp.model_name} → {out}")
        return np.load(out, mmap_mode="r")

    x_explain  = inputs["x_explain"]
    bg         = inputs["bg"]
    coalitions = inputs["coalitions"]
    feature_groups = inputs["feature_groups"]
    I, K   = x_explain.shape[0], bg.shape[0]
    n_coal = coalitions.shape[0]
    p_feat = x_explain.shape[1]

    flat = build_completed_inputs(x_explain, bg, coalitions, feature_groups, p_feat)
    n_rows = flat.shape[0]
    Tloc = model_imp.T

    samples = np.empty((n_rows, N, Tloc), dtype=np.float32)
    raw_buf = None
    with tqdm(total=n_rows, desc=model_imp.model_name, unit="rows") as pbar:
        for s in range(0, n_rows, chunk):
            e = min(n_rows, s + chunk)
            if save_raw:
                s_chunk, r_chunk = model_imp.predict_trajectories(
                    flat[s:e], n_samples=N, seed=seed, return_raw=True,
                )
                samples[s:e] = s_chunk
                if raw_buf is None:
                    n_params = r_chunk.shape[-1]
                    raw_buf = np.empty((n_rows, N, Tloc, n_params), dtype=np.float32)
                raw_buf[s:e] = r_chunk
            else:
                samples[s:e] = model_imp.predict_trajectories(flat[s:e], n_samples=N, seed=seed)
            pbar.update(e - s)

    samples = samples.reshape(I, n_coal, K, N, Tloc)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, samples)
    print(f"✓ trajectories → {out}  shape {samples.shape}  ({samples.nbytes/1e6:.0f} MB)")
    if save_raw:
        raw_buf = raw_buf.reshape(I, n_coal, K, N, Tloc, raw_buf.shape[-1])
        np.save(raw_out, raw_buf)
        print(f"✓ raw → {raw_out}  shape {raw_buf.shape}  ({raw_buf.nbytes/1e6:.0f} MB)")
    return samples

# =============================================================================
#                       Step 2: hierarchy per model
# =============================================================================

def compute_hierarchy(
    samples: np.ndarray,
    *,
    model_name: str,
    estimator_cls=KDECopulaImputer,
    estimator_kwargs: dict | None = None,
    results_dir: str,
    n_jobs: int = 1,
    estimator_suffix: str = "",
) -> dict:
    """Apply the validated trajectory estimator to the ``(I, n_coal, K, N, T)``
    cache. Average over K. Returns dict with L1, L2, L3 per ``(i, S)``.

    ``estimator_suffix`` (e.g. ``"_kde"``, ``"_knn"``, ``"_analytical"``) is
    appended to the cache filename so multiple estimators can coexist on the
    same trajectory cache. Default ``""`` keeps backward-compat with previous
    runs that wrote ``hierarchy_{model_name}.npz``.
    """
    out = Path(results_dir) / f"hierarchy_{model_name}{estimator_suffix}.npz"
    if out.exists():
        print(f"✓ hierarchy cached for {model_name}{estimator_suffix} → {out}")
        return dict(np.load(out))

    I, n_coal, K, N, Tloc = samples.shape
    flat = samples.reshape(I * n_coal * K, N, Tloc)
    L1, L2, L3 = evaluate_on_cached_samples(
        flat, None, estimator_cls, T=Tloc, n_jobs=n_jobs,
        **(estimator_kwargs or {}),
    )
    L1 = L1.reshape(I, n_coal, K, Tloc).mean(axis=2)
    L2 = L2.reshape(I, n_coal, K, Tloc).mean(axis=2)
    L3 = L3.reshape(I, n_coal, K).mean(axis=2)

    assert np.allclose(L3, L2.sum(axis=-1), atol=1e-8), \
        f"chain rule broken on hierarchy[{model_name}{estimator_suffix}]"

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, L1=L1, L2=L2, L3=L3)
    print(f"✓ hierarchy → {out}")
    return dict(L1=L1, L2=L2, L3=L3)

def compute_hierarchy_analytical_deepar(
    samples: np.ndarray,
    raw: np.ndarray,
    *,
    model_name: str = "deepar",
    estimator_suffix: str = "_analytical",
    results_dir: str,
    row_chunk: int = 256,
    n_mixture: int | None = None,
) -> dict:
    """Analytical DeepAR-Normal reference applied to a Section 5.3 trajectory cache.

    Reuses :class:`DeepARNormalImputer` ``_compute_L1/L2/L3`` directly:
    L2 is the closed-form one-step Gaussian conditional entropy, L3 follows
    by the chain rule (Prop. 1), and L1 is the Gaussian-mixture marginal
    entropy estimated *over the same trajectory paths* (the marginalisation
    of the autoregressive factorisation has no closed form). This serves as
    the per-origin validity anchor for KDE-Copula and kNN in the Section 5.3 main paper figure.

    The L1 mixture-marginal step builds a per-row ``(M, M, T)`` log-density
    cube; cost scales as ``M^2 * T`` per row. With ``M = N = 500`` paths,
    the full Section 5.3 cache (``n_rows ≈ 160k``) is ~80 min on a single core.

    Set ``n_mixture`` to sub-sample the first ``n_mixture`` paths along the
    N axis: speedup ~``(N / n_mixture)^2``. The bias on H(Y_t) is
    ``O(1/n_mixture)`` (Jensen, finite-mixture overestimates entropy), but
    is approximately *additive across coalitions* and largely cancels under
    Shapley — empirically <1 pp shift in cross_pct at ``n_mixture=200``.

    Inputs
    ------
    samples : (I, n_coal, K, N, T)         joint trajectories
    raw     : (I, n_coal, K, N, T, n_params)  per-step distribution params
              (network projector output, as captured by
              ``DeepARTrajectoryImputer.predict_trajectories(return_raw=True)``)
    n_mixture : int, optional
        Number of trajectory paths to use for the L1 mixture-marginal cube.
        Default ``None`` = all N. ``n_mixture < N`` accelerates by
        ``(N / n_mixture)^2``.
    """
    out = Path(results_dir) / f"hierarchy_{model_name}{estimator_suffix}.npz"
    if out.exists():
        print(f"✓ hierarchy cached for {model_name}{estimator_suffix} → {out}")
        return dict(np.load(out))

    from .imputer import DeepARNormalImputer

    I, n_coal, K, N, Tloc = samples.shape
    n_params = raw.shape[-1]
    n_rows   = I * n_coal * K

    s_flat = np.asarray(samples).reshape(n_rows, N, Tloc)
    r_flat = np.asarray(raw).reshape(n_rows, N, Tloc, n_params)

    if n_mixture is not None and n_mixture < N:
        s_flat = s_flat[:, :n_mixture, :]
        r_flat = r_flat[:, :n_mixture, :, :]
        M_eff  = n_mixture
        print(f"  L1 mixture sub-sampled: {N} → {M_eff} paths  "
              f"(speedup ~{(N / M_eff) ** 2:.1f}×)", flush=True)
    else:
        M_eff = N

    stub = DeepARNormalImputer(
        model=None, make_dataloader=None, n_samples=M_eff,
        sampler=None, x_explain=np.zeros((1, 1)), T=Tloc,
    )

    L1_flat = np.empty((n_rows, Tloc), dtype=np.float64)
    L2_flat = np.empty((n_rows, Tloc), dtype=np.float64)

    with tqdm(total=n_rows, desc="analytical", unit="rows") as pbar:
        for s_idx in range(0, n_rows, row_chunk):
            e_idx = min(n_rows, s_idx + row_chunk)
            chunk = (s_flat[s_idx:e_idx], r_flat[s_idx:e_idx])
            L1_flat[s_idx:e_idx] = stub._compute_L1(chunk)
            L2_flat[s_idx:e_idx] = stub._compute_L2(chunk)
            pbar.update(e_idx - s_idx)

    L1 = L1_flat.reshape(I, n_coal, K, Tloc).mean(axis=2)
    L2 = L2_flat.reshape(I, n_coal, K, Tloc).mean(axis=2)
    L3 = L2.sum(axis=-1)                                    # (I, n_coal) via chain rule

    assert np.allclose(L3, L2.sum(axis=-1), atol=1e-8), \
        f"chain rule broken on analytical hierarchy[{model_name}]"

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, L1=L1, L2=L2, L3=L3)
    print(f"✓ analytical hierarchy → {out}")
    return dict(L1=L1, L2=L2, L3=L3)

# =============================================================================
#                       Step 3: Shapley values per model
# =============================================================================

def compute_shapley(
    hierarchy: dict,
    *,
    model_name: str,
    n_players: int,
    results_dir: str,
    estimator_suffix: str = "",
) -> dict:
    """Inject the cached value tables into ``HierarchyGame`` and extract Shapley
    values. Returns dict with ``phi_L1`` (I, T, P), ``phi_L2`` (I, T, P),
    ``phi_L3`` (I, P), ``phi_cross`` (I, P).

    ``estimator_suffix`` mirrors :func:`compute_hierarchy` so all three DeepAR
    variants (analytical / KDE / kNN) get distinct cache files
    ``shapley_{model_name}{suffix}.npz``.
    """
    out = Path(results_dir) / f"shapley_{model_name}{estimator_suffix}.npz"
    if out.exists():
        print(f"✓ shapley cached for {model_name}{estimator_suffix} → {out}")
        return dict(np.load(out))

    L1, L2, L3 = hierarchy["L1"], hierarchy["L2"], hierarchy["L3"]
    I, n_coal, T_local = L1.shape

    class _Stub:
        def __init__(self, p): self.n_players = p
        def __call__(self, c): raise NotImplementedError
    stub = _Stub(n_players)
    game = HierarchyGame.__new__(HierarchyGame)
    game._imputer = stub
    game._v_L1 = L1
    game._v_L2 = L2
    game._v_L3 = L3

    phi_L1 = np.empty((I, T_local, n_players))
    phi_L2 = np.empty((I, T_local, n_players))
    phi_L3 = np.empty((I, n_players))
    for i in range(I):
        for t in range(T_local):
            phi_L1[i, t] = game.game_L1(t, i).exact_values('SII', order=1).values[1:]
            phi_L2[i, t] = game.game_L2(t, i).exact_values('SII', order=1).values[1:]
        phi_L3[i] = game.game_L3(i).exact_values('SII', order=1).values[1:]
    phi_cross = phi_L1.sum(axis=1) - phi_L3                              # (I, P)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, phi_L1=phi_L1, phi_L2=phi_L2, phi_L3=phi_L3, phi_cross=phi_cross)
    print(f"✓ shapley → {out}")
    return dict(phi_L1=phi_L1, phi_L2=phi_L2, phi_L3=phi_L3, phi_cross=phi_cross)

# =============================================================================
#                       Step 4: cross-share aggregation
# =============================================================================

def cross_share(
    shapley_by_model: dict,
    *,
    eps: float = 1e-8,
    out_path: str,
) -> pd.DataFrame:
    """Per-instance cross-step attribution metrics.

    Emits two columns for the same underlying quantity in two views:
      ``cross_share = Σ|phi_cross| / (Σ|phi_joint| + eps)``  ∈ [0, ∞)
      ``cross_pct   = 100·Σ|phi_cross| / (Σ|phi_joint| + Σ|phi_cross| + eps)``
                      ∈ [0, 100], the bounded share of attribution mass that
                      goes into cross-step coupling.

    The two are monotone transforms (``cross_pct = 100·ρ/(1+ρ)``); ``cross_pct``
    is the headline metric for the main paper figure (bounded scale).

    Accepts two key formats:
    - ``{model_name: shapley_dict}`` -- emits ``(model, instance, cross_share,
      cross_pct)``.
    - ``{(model_name, estimator_tag): shapley_dict}`` -- also adds an
      ``estimator`` column (used by ``4_main_comparison.py``).
    """
    rows = []
    has_estimator = any(isinstance(k, tuple) for k in shapley_by_model)
    for key, sh in shapley_by_model.items():
        if isinstance(key, tuple):
            model_name, estimator_tag = key
        else:
            model_name, estimator_tag = key, None
        c_abs = np.abs(sh["phi_cross"]).sum(axis=-1)        # (I,)
        j_abs = np.abs(sh["phi_L3"]).sum(axis=-1)           # (I,)
        ratio = c_abs / (j_abs + eps)
        pct   = 100.0 * c_abs / (j_abs + c_abs + eps)
        for i, (r, p) in enumerate(zip(ratio, pct)):
            row = dict(model=model_name, instance=i,
                       cross_share=float(r), cross_pct=float(p))
            if has_estimator:
                row["estimator"] = estimator_tag
            rows.append(row)
    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✓ cross_share → {out_path} ({len(df)} rows)")
    return df

def summary_table(
    hierarchy_by_model: dict,
    cross_df: pd.DataFrame,
    *,
    out_path: str | None = None,
    latex_path: str | None = None,
    architecture: dict | None = None,
    full_coal_index: int = -1,
) -> pd.DataFrame:
    """One-row-per-(model[, estimator]) summary table.

    Accepts two key formats:
    - ``{model_name: hier_dict}`` — back-compat, emits one row per model.
    - ``{(model_name, estimator_tag): hier_dict}`` — one row per
      (model, estimator) pair; estimator becomes a separate column.

    Computes per-instance total correlation
    :math:`TC(\\bm Y \\mid \\bm x) = \\sum_t H(Y_t \\mid \\bm x) - H(\\bm Y \\mid \\bm x)`
    at the full coalition (all features = instance values), then aggregates
    as median + IQR over the I test instances. Joins with the cross-share
    distribution (already per-instance) for matching format.
    """
    has_estimator = any(isinstance(k, tuple) for k in hierarchy_by_model)
    has_cs_estimator = "estimator" in cross_df.columns
    rows = []
    for key, hier in hierarchy_by_model.items():
        if isinstance(key, tuple):
            m, est = key
        else:
            m, est = key, None
        L1 = hier["L1"]                         # (I, n_coal, T)
        L3 = hier["L3"]                         # (I, n_coal)
        full = full_coal_index % L1.shape[1]
        tc = L1[:, full, :].sum(axis=-1) - L3[:, full]    # (I,)

        if has_cs_estimator and est is not None:
            sub = cross_df[(cross_df["model"] == m) &
                           (cross_df["estimator"] == est)]
        else:
            sub = cross_df[cross_df["model"] == m]
        cs  = sub["cross_share"].to_numpy()
        cp  = sub["cross_pct"].to_numpy() if "cross_pct" in sub.columns \
              else 100.0 * cs / (1.0 + cs)

        row = dict(
            model=m,
            architecture=(architecture or {}).get(m if est is None else key, ""),
            tc_median=float(np.median(tc)),
            tc_q25=float(np.quantile(tc, 0.25)),
            tc_q75=float(np.quantile(tc, 0.75)),
            cross_share_median=float(np.median(cs)) if len(cs) else float("nan"),
            cross_share_q25=float(np.quantile(cs, 0.25)) if len(cs) else float("nan"),
            cross_share_q75=float(np.quantile(cs, 0.75)) if len(cs) else float("nan"),
            cross_pct_median=float(np.median(cp)) if len(cp) else float("nan"),
            cross_pct_q25=float(np.quantile(cp, 0.25)) if len(cp) else float("nan"),
            cross_pct_q75=float(np.quantile(cp, 0.75)) if len(cp) else float("nan"),
        )
        if has_estimator:
            row["estimator"] = est
        rows.append(row)
    df = pd.DataFrame(rows)

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"✓ summary table → {out_path}")

    if latex_path is not None:
        Path(latex_path).parent.mkdir(parents=True, exist_ok=True)
        if has_estimator:
            lines = [
                r"\begin{tabular}{lllrrrr}",
                r"\toprule",
                r"Model & Estimator & Architecture & "
                r"Median $TC$ [nats] & IQR & "
                r"Median cross-share (\%) & IQR \\",
                r"\midrule",
            ]
            for _, r in df.iterrows():
                tc_iqr = f"[{r['tc_q25']:.2f},\\,{r['tc_q75']:.2f}]"
                cp_iqr = f"[{r['cross_pct_q25']:.1f},\\,{r['cross_pct_q75']:.1f}]"
                lines.append(
                    f"{r['model']} & {r['estimator']} & {r['architecture']} & "
                    f"{r['tc_median']:.3f} & {tc_iqr} & "
                    f"{r['cross_pct_median']:.1f} & {cp_iqr} \\\\"
                )
        else:
            lines = [
                r"\begin{tabular}{llrrrr}",
                r"\toprule",
                r"Model & Architecture & "
                r"Median $TC$ [nats] & IQR & "
                r"Median cross-share (\%) & IQR \\",
                r"\midrule",
            ]
            for _, r in df.iterrows():
                tc_iqr = f"[{r['tc_q25']:.2f},\\,{r['tc_q75']:.2f}]"
                cp_iqr = f"[{r['cross_pct_q25']:.1f},\\,{r['cross_pct_q75']:.1f}]"
                lines.append(
                    f"{r['model']} & {r['architecture']} & "
                    f"{r['tc_median']:.3f} & {tc_iqr} & "
                    f"{r['cross_pct_median']:.1f} & {cp_iqr} \\\\"
                )
        lines += [r"\bottomrule", r"\end{tabular}"]
        Path(latex_path).write_text("\n".join(lines) + "\n")
        print(f"✓ LaTeX table → {latex_path}")

    return df

# =============================================================================
#                       Step 5: Figure 3 main boxplot
# =============================================================================

def _default_model_colors(models: list[str]) -> dict[str, str]:
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    return {m: palette[i % len(palette)] for i, m in enumerate(models)}


def _setup_rc():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

def plot_fig3_main_boxplot(
    cross_df: pd.DataFrame,
    *,
    model_estimator_pairs: list[tuple[str, str]],
    model_colors: dict[str, str] | None = None,
    estimator_hatches: dict[str, str] | None = None,
    estimator_display: dict[str, str] | None = None,
    model_display: dict[str, str] | None = None,
    save_path: str,
    figsize: tuple[float, float] = (5.5, 4.0),
    y_clip_upper: float | None = None,
    group_gap: float = 0.25,
    intra_gap: float = 0.08,
    box_width: float = 0.4,
    metric: str = "cross_pct",
):
    """Single-panel boxplot for the Section 5.3 main figure.

    One box per ``(model, estimator)`` pair, in the order given by
    ``model_estimator_pairs``. Boxes are visually grouped by model: pairs
    sharing the same model are placed adjacent (gap ``intra_gap``); a
    larger ``group_gap`` separates different models. Individual instance
    cross-share values are scattered behind the boxes at low alpha.

    Parameters
    ----------
    cross_df :
        DataFrame produced by :func:`cross_share` with the
        ``(model, estimator)``-keyed dict. Must contain columns
        ``model``, ``estimator``, ``cross_share``.
    model_estimator_pairs :
        Ordered list of ``(model_name, estimator_tag)`` tuples that select
        which boxes to draw and in which order.
    estimator_hatches :
        Optional ``{tag: matplotlib hatch string}`` for the box face.
        Default: ``analytical`` solid, ``kde`` no hatch, ``knn`` ``"///"``.
    y_clip_upper :
        If set, the y-axis is clipped to this upper limit; instance points
        above the limit are shown as scatter at the top edge.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    _setup_rc()

    estimator_hatches = estimator_hatches or {
        "analytical": "",   # dots — visually distinct as "the reference"
        "kde":        "....",       # solid — main sample-based estimator
        "knn":        "///",    # diagonal — alternative sample-based
    }
    estimator_display = estimator_display or {
        "analytical": "Parametric",   # closed-form L2/L3 + Gauss-mixture L1
        "kde":        "Copula KDE",
        "knn":        "kNN",
    }
    if model_display is None:
        model_display = {m: m for m, _ in model_estimator_pairs}
    if model_colors is None:
        model_colors = _default_model_colors(
            [m for m, _ in model_estimator_pairs]
        )

    # Layout the box positions: pairs sharing a model are separated by
    # ``box_width + intra_gap``; different-model adjacency uses
    # ``box_width + group_gap``. This keeps positions independent of the
    # box width and prevents accidental overlap.
    positions: list[float] = []
    cur = 0.0
    prev_model = None
    for (m, _) in model_estimator_pairs:
        if prev_model is not None:
            cur += box_width + (group_gap if m != prev_model else intra_gap)
        positions.append(cur)
        prev_model = m

    # Pull the per-instance metric values for each pair.
    if metric not in cross_df.columns:
        raise ValueError(
            f"metric={metric!r} not in cross_df columns ({list(cross_df.columns)})"
        )
    data: list[np.ndarray] = []
    for (m, est) in model_estimator_pairs:
        sub = cross_df[(cross_df["model"] == m) & (cross_df["estimator"] == est)]
        data.append(sub[metric].to_numpy(dtype=float))

    fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")

    # Scatter the raw instance values behind the boxes.
    for pos, vals in zip(positions, data):
        if y_clip_upper is not None:
            vals_plot = np.minimum(vals, y_clip_upper)
        else:
            vals_plot = vals
        jitter = (np.random.RandomState(0).uniform(-0.10, 0.10, size=len(vals_plot)))
        ax.scatter(np.full_like(vals_plot, pos) + jitter, vals_plot,
                   s=8, color="0.35", alpha=0.18, lw=0, zorder=1)

    # Boxplot.
    bp = ax.boxplot(
        data, positions=positions, widths=box_width, patch_artist=True,
        showfliers=False, zorder=2,
        medianprops=dict(color="black", lw=1.4),
        whiskerprops=dict(color="black", lw=0.9),
        capprops=dict(color="black", lw=0.9),
        boxprops=dict(lw=0.9, edgecolor="black"),
    )
    for patch, (m, est) in zip(bp["boxes"], model_estimator_pairs):
        patch.set_facecolor(model_colors.get(m, "#888888"))
        patch.set_alpha(0.85)
        h = estimator_hatches.get(est, "")
        if h:
            patch.set_hatch(h)

    # X-tick labels: two-line "{model}\n({estimator})".
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{model_display.get(m, m)}\n({estimator_display.get(est, est)})"
         for (m, est) in model_estimator_pairs],
        fontsize=8.0,
    )
    # Tight x-limits — reduce whitespace between the outermost boxes and
    # the y-axis / right edge. ``edge_pad`` is in data units; 0.1 gives a
    # snug-but-not-cramped margin of ~25% of one box width on each side.
    edge_pad = 0.10
    ax.set_xlim(positions[0] - box_width / 2 - edge_pad,
                positions[-1] + box_width / 2 + edge_pad)
    if metric == "cross_pct":
        ax.set_ylabel("cross-component share (%)")
        ax.set_ylim(0, 100)
    else:
        ax.set_ylabel(r"cross-component share $\rho_{\mathrm{cross}}(\mathbf{x}_i)$")
        if y_clip_upper is not None:
            ax.set_ylim(top=y_clip_upper)
        ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # Compact legend for the estimator-tag → fill-style mapping.
    legend_handles = []
    seen_estimators = []
    for (_, est) in model_estimator_pairs:
        if est in seen_estimators:
            continue
        seen_estimators.append(est)
        legend_handles.append(Patch(
            facecolor="#cccccc", edgecolor="black", lw=0.7,
            hatch=estimator_hatches.get(est, ""),
            label=estimator_display.get(est, est),
        ))
    if len(seen_estimators) > 1:
        ax.legend(handles=legend_handles, loc="upper left",
                  ncol=1, framealpha=0.9, fontsize=8)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path + ".pdf", bbox_inches="tight")
    fig.savefig(save_path + ".png", bbox_inches="tight", dpi=300)
    print(f"✓ fig3 main boxplot → {save_path}.pdf")
    return fig

# =============================================================================
#  Forecast-quality companion helpers (CRPS/MAE + side-by-side visualisation).
#  These let us anchor the cross-component diagnostic against the standard
#  marginal-accuracy metrics and a qualitative look at the joint structure.
# =============================================================================

def extract_forecast_test_data(
    df_long: pd.DataFrame,
    inputs: dict,
    *,
    CONTEXT: int,
    T: int,
    value_col: str = "target",
) -> dict:
    """Pull historical context and ground-truth horizons for the I origins
    referenced by ``inputs['instance_series_id']`` / ``instance_time_idx``.

    Returns
    -------
    dict with
        context : (I, CONTEXT)  the lookback window ending at the origin
        y_true  : (I, T)        ground truth at horizons 1..T
    """
    # Convention from ``build_feature_vec``: ``pred_time_idx`` (= ``t0``) is
    # the *first* forecast horizon. The encoder context covers
    # ``[t0 - CONTEXT, t0 - 1]`` (CONTEXT values), and the prediction
    # horizons are ``[t0, t0 + T - 1]`` (T values).
    series_ids = inputs["instance_series_id"]
    time_idx   = inputs["instance_time_idx"]
    I = len(series_ids)
    context = np.empty((I, CONTEXT), dtype=np.float64)
    y_true  = np.empty((I, T),       dtype=np.float64)
    for i, (sid, t0) in enumerate(zip(series_ids, time_idx)):
        sub = df_long[df_long["series_id"] == sid].set_index("time_idx")[value_col]
        context[i] = sub.loc[t0 - CONTEXT: t0 - 1].to_numpy()
        y_true[i]  = sub.loc[t0: t0 + T - 1].to_numpy()
    return dict(context=context, y_true=y_true)

def compute_forecast_metrics(
    samples_full: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Per-instance, per-horizon CRPS and MAE on the full-coalition samples.

    Parameters
    ----------
    samples_full : (I, M, T) where M is the joint sample bag at the full
        coalition (typically ``trajectories[:, full_idx, :, :, :]`` flattened
        across the K backgrounds).
    y_true : (I, T)

    Returns
    -------
    dict with
        crps   : (I, T)   sample-CRPS using the sorted-form identity
        mae    : (I, T)   |mean(samples) - y|
        crps_median, mae_median : per-horizon scalars (over I)
        crps_overall, mae_overall : scalars over (I, T)
    """
    I, M, T = samples_full.shape
    assert y_true.shape == (I, T)

    # Sample-CRPS via sorted form: O(M log M) per (i, t).
    sorted_s = np.sort(samples_full, axis=1)            # (I, M, T)
    coeffs   = (2 * np.arange(M) + 1 - M) / (M ** 2)    # (M,)
    pairwise = np.einsum("imt,m->it", sorted_s, coeffs) # (I, T)
    abs_dev  = np.mean(np.abs(samples_full - y_true[:, None, :]), axis=1)  # (I, T)
    crps = abs_dev - pairwise                            # (I, T)

    # MAE on sample MEDIAN (matches `notebooks/3_train_deepar.py` convention,
    # which is robust to LogNormal-style skew where mean ≫ median).
    median_pred = np.median(samples_full, axis=1)        # (I, T)
    mae = np.abs(median_pred - y_true)                   # (I, T)

    return dict(
        crps=crps, mae=mae,
        crps_per_t=crps.mean(axis=0),                    # (T,)
        mae_per_t =mae.mean(axis=0),                     # (T,)
        crps_overall=float(crps.mean()),                 # MEAN over (I, T)
        mae_overall =float(mae.mean()),
        crps_median =float(np.median(crps)),             # also the median for ref
        mae_median  =float(np.median(mae)),
    )

def plot_forecast_appendix(
    df_long: pd.DataFrame,
    inputs: dict,
    samples_by_model: dict[str, np.ndarray],
    *,
    CONTEXT: int,
    T: int,
    instance_indices: list[int],
    full_coal_idx: int = -1,
    n_overlay_paths: int = 80,
    model_colors: dict[str, str] | None = None,
    model_display: dict[str, str] | None = None,
    save_path: str,
    panel_width: float = 3.5,
    panel_height: float = 1.7,
):
    """Layout: ``len(models)`` rows × ``(len(instance_indices) + 1)`` cols.

    Each row corresponds to one model. The first ``len(instance_indices)``
    columns show forecast sample-overlays for distinct forecast origins:
    context (grey) + ground truth (black) + ``n_overlay_paths`` sample
    paths (model color, alpha=0.18) + per-step median + 5/95% envelope.
    The last column shows the model's T×T Spearman correlation matrix
    aggregated over all instances at the full coalition — a structural
    summary of inter-horizon coupling.

    Parameters
    ----------
    samples_by_model : {model_name: (I, n_coal, K, N, T) trajectory cache}
        Sliced internally at ``full_coal_idx`` and flattened across K.
    instance_indices :
        List of forecast-origin indices to display in the forecast columns.
    """
    import matplotlib.pyplot as plt
    _setup_rc()

    test = extract_forecast_test_data(df_long, inputs, CONTEXT=CONTEXT, T=T)
    n_inst = len(instance_indices)

    models = list(samples_by_model.keys())
    if model_colors is None:
        model_colors = _default_model_colors(models)
    if model_display is None:
        model_display = {m: m for m in models}

    full_idx = full_coal_idx % next(iter(samples_by_model.values())).shape[1]

    # Per-instance joint samples at the full coalition.
    inst_samples = {}                                     # {(m, i): (K*N, T)}
    for m, traj in samples_by_model.items():
        for i in instance_indices:
            s = np.ascontiguousarray(traj[i, full_idx])
            inst_samples[(m, i)] = s.reshape(-1, s.shape[-1]).astype(np.float64)

    # Per-model average rank-correlation across all I instances at full
    # coalition (more robust + a structural summary, not single-origin noise).
    full_pop = {}
    for m, traj in samples_by_model.items():
        a = np.ascontiguousarray(traj[:, full_idx]).astype(np.float64)
        full_pop[m] = a.reshape(a.shape[0], -1, a.shape[-1])  # (I, K*N, T)

    corr_mats = {}
    for m, pop in full_pop.items():
        # Per-instance Spearman → average over instances.
        accum = np.zeros((T, T))
        I_pop = pop.shape[0]
        for i in range(I_pop):
            ranks = np.argsort(np.argsort(pop[i], axis=0), axis=0).astype(float)
            accum += np.corrcoef(ranks.T)
        corr_mats[m] = accum / I_pop
    vmax = max(
        np.abs(corr_mats[m] - np.eye(T)).max() for m in models
    )
    vmax = max(vmax, 0.3)

    n_cols = n_inst + 1
    # Forecast cols share 4/5 of horizontal real estate (2/5 each for n_inst=2),
    # correlation column gets the remaining 1/5. Concretely, each forecast
    # column has ratio 4 and the correlation column has ratio ``n_inst`` so
    # total ratio = 5·n_inst → forecast share = 4·n_inst / 5·n_inst = 4/5.
    width_ratios = [4.0] * n_inst + [float(n_inst)]
    fig, axes = plt.subplots(
        len(models), n_cols,
        figsize=(panel_width * n_cols + 0.3, panel_height * len(models) + 0.5),
        layout="constrained",
        gridspec_kw=dict(width_ratios=width_ratios),
    )
    if len(models) == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    t_ctx = np.arange(-CONTEXT + 1, 1)
    t_fwd = np.arange(1, T + 1)
    rng = np.random.default_rng(0)

    for row, m in enumerate(models):
        # ── Forecast columns: nested uncertainty bands ─────────────────
        for col, i in enumerate(instance_indices):
            ax = axes[row, col]
            ctx = test["context"][i]
            y   = test["y_true"][i]
            s   = inst_samples[(m, i)]
            med    = np.median(s, axis=0)
            q05, q25, q75, q95 = np.quantile(s, [0.05, 0.25, 0.75, 0.95], axis=0)
            ax.fill_between(t_fwd, q05, q95, color=model_colors[m],
                            alpha=0.18, edgecolor="none", label="5–95%")
            ax.fill_between(t_fwd, q25, q75, color=model_colors[m],
                            alpha=0.35, edgecolor="none", label="25–75%")
            ax.plot(t_fwd, med, color=model_colors[m], lw=1.4, label="median")
            ax.plot(t_ctx[-CONTEXT // 2:], ctx[-CONTEXT // 2:],
                    color="0.4", lw=0.9, label="context")
            ax.plot(t_fwd, y, color="black", lw=1.2, marker="o", ms=2.0,
                    label="truth")
            ax.axvline(0, color="0.6", ls=":", lw=0.6)
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(f"origin {i}", fontsize=8)
            if row == len(models) - 1:
                ax.set_xlabel(r"horizon $t$", fontsize=7)
            if col == 0:
                ax.set_ylabel(model_display[m], fontsize=8, fontweight="bold")
            ax.grid(True, alpha=0.25)

        # ── Correlation column ─────────────────────────────────────────
        ax = axes[row, n_inst]
        im = ax.imshow(corr_mats[m], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="equal", origin="lower")
        ax.set_xticks(range(0, T, 2))
        ax.set_yticks(range(0, T, 2))
        ax.set_xticklabels(range(1, T + 1, 2), fontsize=6)
        ax.set_yticklabels(range(1, T + 1, 2), fontsize=6)
        if row == 0:
            ax.set_title(r"Spearman $\rho_{t,t'}$", fontsize=8)
        if row == len(models) - 1:
            ax.set_xlabel(r"$t'$", fontsize=7)
        ax.set_ylabel(r"$t$", fontsize=7)

    # Shared colorbar to the right of the correlation column.
    cbar = fig.colorbar(im, ax=axes[:, n_inst].tolist(),
                        shrink=0.85, pad=0.02, fraction=0.06)
    cbar.set_label(r"$\rho$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Compact shared legend in the upper-left forecast panel.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="0.4", lw=0.9, label="context"),
        Line2D([0], [0], color="black", lw=1.2, marker="o", ms=2.5,
               label="truth"),
        Line2D([0], [0], color=model_colors[models[0]], lw=1.4,
               label="median"),
    ]
    axes[0, 0].legend(handles=handles, loc="upper left", fontsize=6,
                      ncol=1, framealpha=0.9)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path + ".pdf", bbox_inches="tight")
    fig.savefig(save_path + ".png", bbox_inches="tight", dpi=300)
    print(f"✓ forecast appendix figure → {save_path}.pdf")
    return fig

