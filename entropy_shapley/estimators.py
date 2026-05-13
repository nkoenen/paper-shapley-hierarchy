"""
Sample-based entropy estimators for Section 5.3.2 (Estimator Validation).

All four estimators inherit from `SampleImputer`, a model-agnostic base that
takes a `predict_fn(X) -> (samples, raw)` callable. `raw` may be None for
models that do not expose distributional parameters (e.g. zero-shot
foundation models like Chronos); only `MixtureCopulaImputer` requires it.

Estimators
----------
MVNSampleImputer
    Empirical: fit multivariate normal to sample trajectories, apply
    closed-form Gaussian entropy formulas. Biased when marginals are
    heavy-tailed or skewed, i.e, mismatched to the true distribution.

KDECopulaImputer
    Marginals: Gaussian-KDE with leave-one-out entropy.
    Dependence: Gaussian copula on PIT-transformed samples.
    The fully model-agnostic pipeline used in Section 5.3.3.

MixtureCopulaImputer
    Marginals: analytical mixture entropy from raw distribution parameters
                (reuses the same logsumexp construction as imputer.py).
    Dependence: Gaussian copula on PIT-transformed samples.
    Validation-only — isolates the copula approximation error.
    Requires `raw` parameters from the predict_fn.

KNNImputer
    Kozachenko-Leonenko nonparametric estimator.
    L1 via univariate KL per horizon, L3 via multivariate KL on (M, T)
    samples, L2 via chain-rule differences L3_{1:t} - L3_{1:t-1}.
    Degrades in high T due to curse of dimensionality.

All four estimators share the same (L1, L2, L3) API defined by the base
class HierarchyImputer. Internal per-row computation is cached across the
three _compute_* calls to avoid threefold recomputation inside a chunk.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm, gaussian_kde, lognorm
from scipy.stats import t as student_t
from scipy.special import digamma, gammaln, logsumexp, expit
from scipy.spatial import cKDTree

from .imputer import HierarchyImputer


# =============================================================================
#                  Base class: SampleImputer (model-agnostic)
# =============================================================================

class SampleImputer(HierarchyImputer):
    """Hierarchy imputer for sample-based entropy estimators.

    Decouples sample generation from entropy estimation: pass any callable
    ``predict_fn(X) -> (samples, raw)`` that produces trajectories from
    completed feature matrices. ``raw`` may be ``None`` for models that do
    not expose distributional parameters (e.g. zero-shot foundation models
    like Chronos); only ``MixtureCopulaImputer`` requires it.

    Subclasses implement ``_compute_all((samples, raw)) -> (L1, L2, L3)``.
    The triple is cached per ``id(samples)`` so that ``_process_chunk``'s
    three calls to ``_compute_L{1,2,3}`` share work (KDE fit, PIT,
    correlation, Cholesky).

    Parameters
    ----------
    predict_fn : Callable[[np.ndarray], tuple[np.ndarray, np.ndarray | None]] | None
        ``(X: (n, p)) -> (samples: (n, M, T), raw: (n, M, T, n_params) | None)``.
        Pass ``None`` when feeding a precomputed cache directly via
        ``_compute_all`` (see ``evaluate_on_cached_samples``).
    """

    def __init__(self, predict_fn, *args, **kwargs):
        super().__init__(model=None, *args, **kwargs)
        self._predict_fn = predict_fn
        self._cache_id: int | None = None
        self._cache_triple: tuple | None = None

    def _predict(self, X: np.ndarray):
        if self._predict_fn is None:
            raise RuntimeError(
                "SampleImputer has no predict_fn — call _compute_all directly "
                "or use evaluate_on_cached_samples for cached trajectories.")
        return self._predict_fn(X)

    def _compute_all(self, dist) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Subclasses implement this. Returns (L1, L2, L3) for the chunk."""
        raise NotImplementedError

    def _get_cached(self, dist):
        samples = dist[0] if isinstance(dist, tuple) else dist
        dist_id = id(samples)
        if dist_id != self._cache_id:
            self._cache_id = dist_id
            self._cache_triple = self._compute_all(dist)
        return self._cache_triple

    def _compute_L1(self, dist):
        return self._get_cached(dist)[0]

    def _compute_L2(self, dist):
        return self._get_cached(dist)[1]

    def _compute_L3(self, dist):
        return self._get_cached(dist)[2]


# =============================================================================
#       Section 4.2.3 — "Parametric Gaussian estimation" 
# =============================================================================

class MVNSampleImputer(SampleImputer):
    """Fit MVN to sample trajectories; use closed-form Gaussian entropy.

    Approximates both marginals and dependence as Gaussian. Strawman that
    shows why the Gaussian-copula approach is needed: when the true
    marginals are heavy-tailed (StudentT) or skewed (LogNormal), the
    Gaussian moment-match is biased, and the bias compounds in the L3
    determinant.

    Uses the same analytical formulas as GaussianImputer in imputer.py
    but with Sigma replaced by the empirical sample covariance.
    """

    def __init__(self, *args, ridge: float = 1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self._ridge = ridge

    def _compute_all(self, dist):
        samples = dist[0] if isinstance(dist, tuple) else dist  # (n, M, T)
        n, M, T = samples.shape

        eye_T = self._ridge * np.eye(T)
        half_log_2pie = 0.5 * np.log(2 * np.pi * np.e)

        # Vectorised over the n=(c × n_inst × n_coal) rows. Build all sample-
        # covariance matrices at once (batched gemm) and use numpy's batched
        # Cholesky (LAPACK potrf). At a 12 800-row sweep this is ~2.5× faster
        # than the per-row loop variant.
        centered = samples - samples.mean(axis=1, keepdims=True)              # (n, M, T)
        Sigma    = (centered.transpose(0, 2, 1) @ centered) / (M - 1)         # (n, T, T)
        Sigma    = Sigma + eye_T                                              # broadcast ridge
        Sigma    = 0.5 * (Sigma + Sigma.transpose(0, 2, 1))                   # numerical symmetrise
        diag_var = np.diagonal(Sigma, axis1=-2, axis2=-1)                     # (n, T)

        try:
            chol = np.linalg.cholesky(Sigma)                                  # batched
        except np.linalg.LinAlgError:
            # Rank-deficient matrices (e.g. M < T) or borderline-PD ones break
            # batched Cholesky on a single bad row. Fall back to per-row with
            # adaptive jitter — the loop is slower but only triggered when needed.
            chol = np.empty_like(Sigma)
            for i in range(n):
                S = Sigma[i]
                jitter = max(self._ridge, 1e-8)
                for _ in range(20):
                    try:
                        chol[i] = np.linalg.cholesky(S)
                        break
                    except np.linalg.LinAlgError:
                        S = S + jitter * np.eye(T)
                        jitter *= 10
                else:
                    raise np.linalg.LinAlgError(
                        f"MVNSampleImputer: row {i} not PD even at jitter={jitter:.1e}")
        diag_L = np.diagonal(chol, axis1=-2, axis2=-1)                        # (n, T)

        L1 = half_log_2pie + 0.5 * np.log(diag_var)
        L2 = half_log_2pie + np.log(diag_L)
        L3 = L2.sum(axis=-1)
        return L1, L2, L3


# =============================================================================
#   Mixture Copula Imputer
#   Used in Section 5.3.2 to isolate the copula approximation error: same 
#   pipeline as KDECopulaImputer but with analytical mixture marginals from 
#   raw distribution parameters.
# =============================================================================

class MixtureCopulaImputer(SampleImputer):
    """Analytical mixture marginals + Gaussian copula on PIT-transformed samples.

    Requires `raw` distribution parameters from the predict_fn — only models
    that expose per-step parameters (e.g. DeepAR) can drive this estimator.

    Parameters
    ----------
    distribution : str
        'normal', 'studentt', or 'lognormal'. Determines how raw parameters
        are mapped to the per-step log-density for the marginal mixture.
    shrinkage : float
        Optional shrinkage toward the identity for the correlation matrix.
        Default 0.0 — instead, ill-conditioned matrices are stabilised via
        eigenvalue clipping in `_project_to_correlation_matrix`, which avoids
        the constant off-diagonal bias of fixed shrinkage.
    """

    def __init__(self, *args, distribution: str = "normal",
                 shrinkage: float = 0.0, cdf_eps: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        if distribution not in {"normal", "studentt", "lognormal"}:
            raise ValueError(f"unknown distribution: {distribution}")
        self._distribution = distribution
        self._shrinkage = shrinkage
        self._cdf_eps = cdf_eps
        # n_params per step depends on distribution
        self._n_params = 3 if distribution == "studentt" else 2

    # ── Analytical mixture marginal entropies ────────────────────────────────

    def _mixture_marginal_entropy(self, samples_i: np.ndarray,
                                    raw_i: np.ndarray) -> np.ndarray:
        """H(Y_t) as mixture over M trajectories via logsumexp.

        samples_i : (M, T)
        raw_i     : (M, T, n_params)
        returns   : (T,)

        For each (y, θ) pair we evaluate the per-step log-density via
        ``scipy.stats.{norm,lognorm,t}.logpdf``, average over the M trajectory
        components in log-space (``logsumexp − log M``), and report
        ``H(Y_t) = −mean_y log p_mix(y)``.
        """
        M, T = samples_i.shape
        y = samples_i[:, None, :]                                            # (M_y, 1, T)

        if self._distribution == "normal":
            mu    = raw_i[..., 0][None, :, :]                                # (1, M_s, T)
            sigma = np.log1p(np.exp(raw_i[..., 1]))[None, :, :]              # softplus
            log_comps = norm.logpdf(y, loc=mu, scale=sigma)

        elif self._distribution == "lognormal":
            mu    = raw_i[..., 0][None, :, :]
            # LogNormalDistributionLoss uses sigmoid(raw) + 1e-3 for sigma
            # (see entropy_shapley/losses.py). scipy.stats.lognorm uses the
            # (s = σ, scale = exp(μ)) parameterisation.
            sigma = (expit(raw_i[..., 1]) + 1e-3)[None, :, :]
            log_comps = lognorm.logpdf(np.clip(y, 1e-6, None),
                                       s=sigma, scale=np.exp(mu))

        else:  # studentt
            df    = (np.log1p(np.exp(raw_i[..., 0])) + 2)[None, :, :]
            mu    = raw_i[..., 1][None, :, :]
            sigma = np.log1p(np.exp(raw_i[..., 2]))[None, :, :]
            log_comps = student_t.logpdf(y, df=df, loc=mu, scale=sigma)

        log_p = logsumexp(log_comps, axis=1) - np.log(M)
        return -log_p.mean(axis=0)

    # ── Dependence via Gauss copula on PIT-transformed samples ───────────────

    def _copula_log_L_diag(self, samples_i: np.ndarray) -> np.ndarray:
        """log of Cholesky-diagonal of the Gauss-copula correlation R.

        returns (T,). Sum gives 0.5 * log det R (the copula entropy up to sign).
        """
        M, T = samples_i.shape
        if T == 1:
            return np.zeros(1)
        # Empirical rank-INT (rank-based inverse-normal transform)
        ranks = np.argsort(np.argsort(samples_i, axis=0), axis=0) + 1
        u = np.clip(ranks / (M + 1), self._cdf_eps, 1 - self._cdf_eps)
        z = norm.ppf(u)

        R = np.corrcoef(z, rowvar=False)
        if not np.all(np.isfinite(R)):
            R = np.eye(T)
        if self._shrinkage > 0:
            R = (1 - self._shrinkage) * R + self._shrinkage * np.eye(T)
        # Eigenvalue clipping handles rank-deficient / borderline-PSD cases
        # without introducing a constant off-diagonal bias on the correlations.
        R = _project_to_correlation_matrix(R)
        L = np.linalg.cholesky(R)
        return np.log(np.maximum(np.diag(L), 1e-12))

    def _compute_all(self, dist):
        if not isinstance(dist, tuple) or len(dist) != 2 or dist[1] is None:
            raise ValueError(
                "MixtureCopulaImputer requires raw distribution parameters; "
                "got samples only. Use KDECopulaImputer for sample-only inputs.")
        samples, raw = dist                           # (n, M, T), (n, M, T, n_params)
        n, M, T = samples.shape
        L1 = np.empty((n, T))
        L2 = np.empty((n, T))
        L3 = np.empty(n)
        for i in range(n):
            H_marg = self._mixture_marginal_entropy(samples[i], raw[i])
            log_L  = self._copula_log_L_diag(samples[i])
            L1[i] = H_marg
            L2[i] = H_marg + log_L
            L3[i] = L2[i].sum()
        return L1, L2, L3


# =============================================================================
#  Section 4.2.3 — "Semiparametric Gaussian copula" (Algorithm in Appendix D).
#  Main paper estimator. Fully model-agnostic: KDE marginals + Gauss copula.
# =============================================================================

class KDECopulaImputer(SampleImputer):
    """KDE marginals + Gaussian copula on PIT-transformed samples.

    Implements the algorithm from Appendix D, steps 1–6, applied to a
    pre-existing trajectory bag. Used in Section 5.3.3 on NeuralForecast
    models without a distribution head. Validated in 5.3.2 on DeepAR.

    Parameters
    ----------
    shrinkage : float
        Optional shrinkage toward the identity for the correlation matrix.
        Default 0.0 — instead, ill-conditioned matrices are stabilised via
        eigenvalue clipping in `_project_to_correlation_matrix`, which avoids
        the constant off-diagonal bias of fixed shrinkage.
    marginal_lower_bound : float | None
        For bounded-support marginals (e.g. LogNormal), pass 0.0.
    marginal_transform : {'none', 'log'}
        'log' recommended for heavily skewed positive-support marginals.
    """

    def __init__(self, *args,
                 shrinkage: float = 0.0,
                 cdf_eps: float = 1e-6,
                 marginal_lower_bound: float | None = None,
                 marginal_transform: str = "none",
                 marginal_transform_eps: float = 1e-9,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._shrinkage = shrinkage
        self._cdf_eps = cdf_eps
        self._lower_bound = marginal_lower_bound
        self._transform = marginal_transform
        self._transform_eps = marginal_transform_eps

    def _estimate_one(self, samples_i: np.ndarray):
        """samples_i: (M, T) -> (L1, L2, L3)."""
        M, T = samples_i.shape
        L1 = np.empty(T)
        u = np.empty_like(samples_i)

        for t in range(T):
            kde = GaussianKDE1D(
                samples_i[:, t],
                lower_bound=self._lower_bound,
                transform=self._transform,
                transform_eps=self._transform_eps,
            )
            L1[t] = kde.entropy_loo()
            u[:, t] = np.clip(kde.cdf(samples_i[:, t]), self._cdf_eps, 1 - self._cdf_eps)

        if T == 1:
            return L1, L1.copy(), float(L1[0])

        z = norm.ppf(u)
        R = np.corrcoef(z, rowvar=False)
        if not np.all(np.isfinite(R)):
            R = np.eye(T)
        if self._shrinkage > 0:
            R = (1 - self._shrinkage) * R + self._shrinkage * np.eye(T)
        # Eigenvalue clipping handles rank-deficient / borderline-PSD cases
        # without introducing a constant off-diagonal bias on the correlations.
        R = _project_to_correlation_matrix(R)
        chol = np.linalg.cholesky(R)
        log_L_diag = np.log(np.maximum(np.diag(chol), 1e-12))

        L2 = L1 + log_L_diag
        return L1, L2, float(L2.sum())

    def _compute_all(self, dist):
        samples = dist[0] if isinstance(dist, tuple) else dist  # ignore raw
        n = samples.shape[0]
        T = samples.shape[-1]
        L1 = np.empty((n, T))
        L2 = np.empty((n, T))
        L3 = np.empty(n)
        for i in range(n):
            L1[i], L2[i], L3[i] = self._estimate_one(samples[i])
        return L1, L2, L3


# =============================================================================
#       Section 4.2.3 — "Nonparametric kNN estimation" (Kozachenko-Leonenko)
# =============================================================================

class KNNImputer(SampleImputer):
    """Nonparametric Kozachenko-Leonenko kNN estimator.

    L1: univariate KL on each horizon column.
    L3: multivariate KL on the full (M, T) sample matrix.
    L2: chain-rule difference L3_{1:t} - L3_{1:t-1}.

    The difference-based L2 is noisy for large t — this is expected behavior
    and part of the validation narrative (curse of dimensionality of kNN in
    multivariate settings).
    """

    def __init__(self, *args, k: int = 5, jitter: float = 1e-12,
                 skip_l1: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._k = k
        self._jitter = jitter
        # L1 = T independent univariate kNN trees per row. The DeepAR validation
        # pipeline (Section 5.3.2) only aggregates L2/L3, so this is dead work
        # there. Set skip_l1=True to fill L1 with NaN and save ~30-40% per row.
        self._skip_l1 = skip_l1

    def _compute_all(self, dist):
        samples = dist[0] if isinstance(dist, tuple) else dist  # (n, M, T)
        n, M, T = samples.shape
        L1 = (np.full((n, T), np.nan) if self._skip_l1 else np.empty((n, T)))
        L2 = np.empty((n, T))
        L3 = np.empty(n)

        for i in range(n):
            y = samples[i]  # (M, T)
            if not self._skip_l1:
                L1[i] = np.array([
                    _knn_entropy(y[:, t:t + 1], k=self._k, jitter=self._jitter)
                    for t in range(T)
                ])
            # Joint prefix entropies H(Y_{1:t}) — needed for L2/L3 via chain rule
            joint_prefix = np.array([
                _knn_entropy(y[:, :t + 1], k=self._k, jitter=self._jitter)
                for t in range(T)
            ])
            L2[i, 0] = joint_prefix[0]
            if T > 1:
                L2[i, 1:] = np.diff(joint_prefix)
            L3[i] = joint_prefix[-1]
        return L1, L2, L3


# =============================================================================
#                  Run an estimator on pre-cached (samples, raw)
# =============================================================================

def evaluate_on_cached_samples(
    samples: np.ndarray,
    raw: np.ndarray | None,
    estimator_cls: type,
    T: int,
    *,
    n_jobs: int = 1,
    backend: str = "loky",
    **estimator_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a `SampleImputer` subclass on a pre-computed (samples, raw)
    cache, bypassing the predict_fn / sampler entirely.

    `samples` has shape (n, M, T) and `raw` shape (n, M, T, n_params) (or None
    for purely sample-based estimators). Returns (L1, L2, L3) shaped (n, T),
    (n, T), (n,).

    Used by Section 5.3: the same nested sample cache is fed through three
    estimators (kNN, MVN, Gaussian copula) without ever generating samples again.

    n_jobs : int
        If > 1, split the n rows into n_jobs equal chunks and run `_compute_all`
        in parallel. Each estimator's per-row inner loop is pure Python with no
        shared state, so this scales near-linearly on multi-core machines.
    """
    if not issubclass(estimator_cls, SampleImputer):
        raise TypeError(
            f"{estimator_cls.__name__} must be a SampleImputer subclass")

    p_dummy = 1
    def _make_est():
        return estimator_cls(
            predict_fn=None,
            sampler=None,
            x_explain=np.zeros((1, p_dummy)),
            T=T,
            **estimator_kwargs,
        )

    if n_jobs == 1:
        return _make_est()._compute_all((samples, raw))

    # Split rows into n_jobs contiguous chunks. Slices are views (no copy);
    # joblib's loky backend memmaps large numpy args automatically.
    n = samples.shape[0]
    bounds = np.linspace(0, n, n_jobs + 1, dtype=int)

    def _run_chunk(s_chunk, r_chunk):
        return _make_est()._compute_all((s_chunk, r_chunk))

    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_run_chunk)(
            samples[s:e],
            raw[s:e] if raw is not None else None,
        )
        for s, e in zip(bounds[:-1], bounds[1:])
        if e > s
    )

    L1 = np.concatenate([r[0] for r in results], axis=0)
    L2 = np.concatenate([r[1] for r in results], axis=0)
    L3 = np.concatenate([r[2] for r in results], axis=0)
    return L1, L2, L3


# =============================================================================
#                      Gaussian KDE with leave-one-out entropy
# =============================================================================

class GaussianKDE1D:
    """1D Gaussian-kernel KDE with leave-one-out entropy and CDF.

    The base density is fitted via ``scipy.stats.gaussian_kde`` (Silverman's
    rule of thumb); on top of that we layer (i) a leave-one-out plug-in
    entropy estimator and (ii) optional boundary handling via reflection or a
    log-transform with Jacobian correction (needed for lower-bounded supports
    such as LogNormal marginals).
    """

    def __init__(
        self,
        samples: np.ndarray,
        bandwidth: float | None = None,
        lower_bound: float | None = None,
        transform: str = "none",
        transform_eps: float = 1e-9,
    ) -> None:
        self.samples = np.asarray(samples, dtype=float).ravel()
        if self.samples.size < 2:
            raise ValueError("need at least 2 samples for KDE")
        self.n = self.samples.size
        self.lower_bound = None if lower_bound is None else float(lower_bound)
        self.transform = str(transform)
        self.transform_eps = float(transform_eps)

        if self.transform not in {"none", "log"}:
            raise ValueError("transform must be 'none' or 'log'")
        if self.transform == "log":
            if self.lower_bound is None:
                raise ValueError("lower_bound required when transform='log'")
            shifted = self.samples - self.lower_bound + self.transform_eps
            if np.any(shifted <= 0):
                raise ValueError("samples must satisfy y > lower_bound - eps")
            self._work = np.log(shifted)
        else:
            self._work = self.samples

        # Bandwidth: scipy's Silverman rule of thumb by default; otherwise a
        # caller-supplied scalar is passed through as `bw_method / std(data)`
        # (scipy's internal convention).
        std_work = float(np.std(self._work, ddof=1)) or 1.0
        if bandwidth is None:
            self._kde = gaussian_kde(self._work, bw_method="silverman")
            self.bandwidth = float(self._kde.factor * std_work)
        else:
            self._kde = gaussian_kde(self._work, bw_method=float(bandwidth) / std_work)
            self.bandwidth = float(bandwidth)
        # (n, n) pairwise differences scaled by bandwidth — needed for LOO
        self._pairwise = (self._work[:, None] - self._work[None, :]) / self.bandwidth

    # ── Transform helpers ────────────────────────────────────────────────────

    def _transform_x(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        if self.transform == "none":
            return x, np.ones_like(x, dtype=bool)
        shifted = x - self.lower_bound + self.transform_eps
        valid = shifted > 0
        x_work = np.where(valid, np.log(np.where(valid, shifted, 1.0)), -np.inf)
        return x_work, valid

    def _jacobian_abs(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.transform == "none":
            return np.ones_like(x, dtype=float)
        shifted = x - self.lower_bound + self.transform_eps
        return np.where(shifted > 0, 1.0 / np.maximum(shifted, self.transform_eps), 0.0)

    # ── Density and CDF ──────────────────────────────────────────────────────

    def pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x_work, valid = self._transform_x(x)
        # Base PDF in the working space via scipy.stats.gaussian_kde.
        # Evaluate only finite entries (log-transform may produce -inf for
        # invalid inputs), then reshape back to x_work.shape.
        flat_work = x_work.ravel()
        finite = np.isfinite(flat_work)
        dens_flat = np.zeros_like(flat_work)
        if finite.any():
            dens_flat[finite] = self._kde.evaluate(flat_work[finite])
        dens_work = dens_flat.reshape(x_work.shape)

        if self.transform == "log":
            return np.where(valid, dens_work * self._jacobian_abs(x), 0.0)

        dens = dens_work
        if self.lower_bound is not None:
            # Reflection correction for positive-support marginals
            mirror = 2.0 * self.lower_bound - self.samples
            z_ref = (np.expand_dims(x, -1) - mirror) / self.bandwidth
            dens = dens + norm.pdf(z_ref).mean(axis=-1) / self.bandwidth
            dens = np.where(x < self.lower_bound, 0.0, dens)
        return dens

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x_work, valid = self._transform_x(x)
        z = (np.expand_dims(x_work, -1) - self._work) / self.bandwidth
        cdf = norm.cdf(z).mean(axis=-1)

        if self.transform == "log":
            return np.clip(np.where(valid, cdf, 0.0), 0.0, 1.0)

        if self.lower_bound is not None:
            z_ref = (np.expand_dims(x, -1) + self.samples - 2.0 * self.lower_bound) / self.bandwidth
            cdf = cdf + norm.cdf(z_ref).mean(axis=-1) - 1.0
            cdf = np.where(x < self.lower_bound, 0.0, cdf)
        return np.clip(cdf, 0.0, 1.0)

    def entropy_loo(self, jitter: float = 1e-12) -> float:
        """Leave-one-out plug-in entropy estimate: H = -mean log p̂_{-i}(y_i)."""
        kern = norm.pdf(self._pairwise) / self.bandwidth
        np.fill_diagonal(kern, 0.0)
        dens = kern.sum(axis=1)

        if self.transform == "log":
            dens = dens / max(self.n - 1, 1)
            dens = dens * self._jacobian_abs(self.samples)
            return float(-np.mean(np.log(np.maximum(dens, jitter))))

        if self.lower_bound is not None:
            mirror = (self.samples[:, None] + self.samples[None, :] - 2.0 * self.lower_bound) / self.bandwidth
            kern_ref = norm.pdf(mirror) / self.bandwidth
            np.fill_diagonal(kern_ref, 0.0)
            dens = dens + kern_ref.sum(axis=1)

        dens = dens / max(self.n - 1, 1)
        return float(-np.mean(np.log(np.maximum(dens, jitter))))



# =============================================================================
#                              Helper functions
# =============================================================================

def _project_to_correlation_matrix(R: np.ndarray, min_eig: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix to a positive-definite correlation matrix
    by eigenvalue clipping and diagonal rescaling."""
    R = 0.5 * (R + R.T)
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, min_eig)
    R_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.maximum(np.diag(R_psd), min_eig))
    corr = R_psd / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return 0.5 * (corr + corr.T)


def _log_unit_ball_volume(dim: int) -> float:
    """Log of Lebesgue volume of d-dimensional unit ball (radius 1)."""
    return 0.5 * dim * np.log(np.pi) - gammaln(0.5 * dim + 1.0)


def _knn_entropy(samples: np.ndarray, k: int = 5, jitter: float = 1e-12) -> float:
    """Kozachenko-Leonenko kNN differential entropy estimator.

    H = psi(n) - psi(k) + log(V_d) + d * mean(log rho_i)

    where rho_i is the Euclidean distance to the k-th nearest neighbor
    (self excluded) and V_d is the volume of the d-dimensional unit ball.

    Follows the radius-convention consistent with Wikipedia and the original
    Kozachenko-Leonenko (1987) paper. See https://infomeasure.readthedocs.io/en/0.5.0/guide/entropy/kozachenko_leonenko/
    for details.
    """
    x = np.atleast_2d(samples).astype(float)
    if x.ndim == 1:
        x = x[:, None]
    n, dim = x.shape
    if n <= k:
        raise ValueError(f"need n > k (got n={n}, k={k})")

    tree = cKDTree(x)
    distances, _ = tree.query(x, k=k + 1, p=2)
    rho = np.maximum(distances[:, -1], jitter)

    return float(
        digamma(n) - digamma(k) + _log_unit_ball_volume(dim) + dim * np.mean(np.log(rho))
    )