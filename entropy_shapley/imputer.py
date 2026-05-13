from abc import ABC, abstractmethod
from math import ceil

import numpy as np
from joblib import Parallel, delayed
from scipy.special import logsumexp, expit
from scipy.stats import norm, lognorm
from scipy.stats import t as student_t
from tqdm import tqdm



############################################################################
#                          Hierarchy-Imputer Class
############################################################################

class HierarchyImputer(ABC):
    """Marginalizes entropy levels over background for arbitrary coalitions.

    Accepts one or more instances to explain simultaneously.  Pass a 1-D array
    ``(p,)`` for a single instance (backward-compatible) or a 2-D array
    ``(n_inst, p)`` to process multiple instances in one shot.  The marginal
    path vectorises over instances; the conditional path requires n_inst == 1
    because each conditional sampler holds its own x_explain.

    __call__(coalitions) -> (v_L1, v_L2, v_L3)
        v_L1, v_L2 : (n_inst, n_coal, T) or None
        v_L3       : (n_inst, n_coal)    or None

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers for chunk processing (default 1 = sequential).
    backend : str
        joblib backend (default ``'loky'``).  Use ``'threading'`` only when the
        model provably releases the GIL (e.g. pure Cython with no Python callbacks).
        ``'loky'`` (multiprocessing) avoids GIL contention at the cost of a
        one-time serialisation overhead that is negligible for n_bg ≳ 200.
    """

    def __init__(self, model, sampler, x_explain: np.ndarray, T: int,
                 chunk_size: int | None = None, n_jobs: int = 1,
                 backend: str = 'loky',
                 feature_groups: list[list[int]] | None = None):
        self._model          = model
        self._sampler        = sampler
        self._x              = np.atleast_2d(x_explain)   # always (n_inst, p)
        self._T              = T
        self._chunk_size     = chunk_size
        self._n_jobs         = n_jobs
        self._backend        = backend
        self._feature_groups = feature_groups

    @property
    def _n_inst(self) -> int:
        return self._x.shape[0]

    @property
    def n_players(self) -> int:
        """Number of Shapley players (groups if feature_groups set, else features)."""
        return len(self._feature_groups) if self._feature_groups is not None else self._x.shape[-1]

    @abstractmethod
    def _predict(self, X_flat: np.ndarray):
        """Run model on X_flat (n, p) and return distribution output."""

    def _compute_L1(self, dist) -> np.ndarray | None:
        """H(Y_t | x) for each t. Input: (n, ...). Returns (n, T) or None."""
        return None

    def _compute_L2(self, dist) -> np.ndarray | None:
        """H(Y_t | Y_{<t}, x) for each t. Input: (n, ...). Returns (n, T) or None."""
        return None

    def _compute_L3(self, dist) -> np.ndarray | None:
        """H(Y | x). Input: (n, ...). Returns (n,) or None."""
        return None

    def __call__(self, coalitions: np.ndarray) -> tuple:
        return self._call_marginal(coalitions)

    def _expand_coalitions(self, coalitions: np.ndarray, p: int) -> np.ndarray:
        """Expand group coalitions (n_coal, n_groups) → (n_coal, p)."""
        if self._feature_groups is None:
            return coalitions
        expanded = np.zeros((len(coalitions), p), dtype=bool)
        for g, indices in enumerate(self._feature_groups):
            expanded[:, indices] = coalitions[:, g : g + 1]
        return expanded

    def _process_chunk(self, chunk: np.ndarray, coalitions: np.ndarray,
                       n_inst: int, n_coal: int, p: int) -> tuple:
        """Process one background chunk; returns partial sums (not divided by n_bg)."""
        c     = len(chunk)
        coalitions = self._expand_coalitions(coalitions, p)

        # Impute data -> c*n_inst*n_coal, p
        X_imp = np.where(
            coalitions[None, None, :, :],           # (1, 1,      n_coal, p)
            self._x[None, :, None, :],              # (1, n_inst, 1,      p)
            chunk[:, None, None, :],                # (c, 1,      1,      p)
        ).reshape(-1, p)                            # (c*n_inst*n_coal,   p)

        # Predict all
        dist = self._predict(X_imp)

        # Compute levels based on the distribution
        r1 = self._compute_L1(dist)
        r2 = self._compute_L2(dist)
        r3 = self._compute_L3(dist)

        # Reshape to format (c, n_inst, n_coal, T)
        a1 = r1.reshape(c, n_inst, n_coal, -1).sum(0) if r1 is not None else None
        a2 = r2.reshape(c, n_inst, n_coal, -1).sum(0) if r2 is not None else None
        a3 = r3.reshape(c, n_inst, n_coal).sum(0)     if r3 is not None else None
        return a1, a2, a3

    def _call_marginal(self, coalitions: np.ndarray) -> tuple:
        bg         = self._sampler()               # (n_bg, p)
        n_bg       = len(bg)
        n_coal     = len(coalitions)
        n_inst     = self._n_inst
        p          = self._x.shape[-1]
        n_jobs     = self._n_jobs

        # Set chunk size
        if n_jobs == 1:
            chunk_size = self._chunk_size or n_bg
        else:
            chunk_size = self._chunk_size or max(1, ceil(n_bg / (n_jobs)))

        # Set chunk of background data
        chunks = [bg[s:s + chunk_size] for s in range(0, n_bg, chunk_size)]

        # Process chunk
        if n_jobs == 1:
            results = [
                self._process_chunk(chunk, coalitions, n_inst, n_coal, p)
                for chunk in tqdm(chunks)
            ]
        else:
            results = list(tqdm(
                Parallel(n_jobs=n_jobs, backend=self._backend, return_as='generator')(
                    delayed(self._process_chunk)(chunk, coalitions, n_inst, n_coal, p)
                    for chunk in chunks
                ),
                total=len(chunks),
            ))

        # Collect all results of all chunks
        acc_L1 = acc_L2 = acc_L3 = None
        for r1, r2, r3 in results:
            if r1 is not None:
                acc_L1 = r1 if acc_L1 is None else acc_L1 + r1
            if r2 is not None:
                acc_L2 = r2 if acc_L2 is None else acc_L2 + r2
            if r3 is not None:
                acc_L3 = r3 if acc_L3 is None else acc_L3 + r3

        return (
            None if acc_L1 is None else acc_L1 / n_bg,  # (n_inst, n_coal, T)
            None if acc_L2 is None else acc_L2 / n_bg,
            None if acc_L3 is None else acc_L3 / n_bg,  # (n_inst, n_coal)
        )




############################################################################
#                            Gaussian Imputer
############################################################################

# ── Gaussian Imputer ──────────────────────────────────────────────────────
class GaussianImputer(HierarchyImputer):
    """Entropy hierarchy for models with multivariate Gaussian output.

    Expects model(X) to return Sigma (n, T, T).

    Entropy levels:
        L1: H(Y_t | x)         = 0.5 * log(2πe * Σ_tt)
        L2: H(Y_t | Y_{<t}, x) = 0.5 * log(2πe * L_tt²)
        L3: H(Y | x)           = T/2 * log(2πe) + Σ_t log(L_tt)
    where L = cholesky(Σ(x)).
    """

    # Predict the covariance matrix Σ for each input row; shape (n, T, T)
    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self._model(X)   # (n, T, T)

    # Level 1: marginal entropy of each step, which is based on the diagonal of Σ
    # H(Y_t | x) = 0.5 * log(2πe * Σ_tt)
    def _compute_L1(self, S: np.ndarray) -> np.ndarray:
        return 0.5 * np.log(2 * np.pi * np.e * np.diagonal(S, axis1=-2, axis2=-1))
    
    # Level 2: conditional entropy of each step given the past, which is based on 
    # the diagonal of L = cholesky(Σ)
    # H(Y_t | Y_{<t}, x) = 0.5 * log(2πe * L_tt²)
    def _compute_L2(self, S: np.ndarray) -> np.ndarray:
        d = np.diagonal(np.linalg.cholesky(S), axis1=-2, axis2=-1)
        return 0.5 * np.log(2 * np.pi * np.e * d ** 2)

    # Level 3: joint entropy of the whole sequence, which is based on the product of L_tt
    # H(Y | x) =  T/2 * log(2πe) +  Σ_t log(L_tt)
    def _compute_L3(self, S: np.ndarray) -> np.ndarray:
        d = np.diagonal(np.linalg.cholesky(S), axis1=-2, axis2=-1)
        return 0.5 * self._T * np.log(2 * np.pi * np.e) + np.sum(np.log(d), axis=-1)


############################################################################
#                  DeepAR Imputers (Normal, LogNormal, StudentT)
############################################################################

# ── Gaussian Imputer ──────────────────────────────────────────────────────
class DeepARImputer(HierarchyImputer):
    """Base class for DeepAR-based entropy hierarchy imputers.

    Hooks the distribution_projector to capture raw per-step parameters, then
    draws M autoregressive sample trajectories via model.predict(mode='samples').

    Subclasses implement _compute_L1/L2/L3 with distribution-specific formulas.

    Parameters
    ----------
    model              : pytorch-forecasting DeepAR model (eval mode)
    make_dataloader    : callable  X (n, p) -> DataLoader
    n_samples          : MC trajectories per predict call (default 200)
    predict_batch_size : int | None
        Override the dataloader batch size for the GPU `predict()` call.
        Default ``None`` lets `make_dataloader` use ``len(X)`` (one batch).
        At high `n_samples` the LSTM autoregressive decode allocates
        `batch * n_samples * T * hidden` per step — tune downward (e.g. 64)
        when n_samples is large to avoid CUDA OOM.
    """

    _n_params: int = 2  # number of raw projector outputs; override in subclasses

    def __init__(self, model, make_dataloader, n_samples: int = 200, *args,
                 predict_batch_size: int | None = None, **kwargs):
        super().__init__(model, *args, **kwargs)
        self._make_dataloader = make_dataloader
        self._n_samples = n_samples
        self._predict_batch_size = predict_batch_size

    def _compute_L3(self, dist: tuple) -> np.ndarray:
        """Joint entropy via chain rule (Prop. 1): H(Y|x) = Σ_t H(Y_t|Y_{<t}, x)."""
        return self._compute_L2(dist).sum(axis=-1)

    def _predict(self, X: np.ndarray) -> tuple:
        import torch
        import torch.nn.functional as F
        n = len(X)
        captured = []
        hook = self._model.distribution_projector.register_forward_hook(
            lambda m, inp, out: captured.append(out.detach().cpu())
        )
        with torch.no_grad():
            pred = self._model.predict(
                self._make_dataloader(X, batch_size=self._predict_batch_size),
                mode='samples', n_samples=self._n_samples,
                return_index=True,
                trainer_kwargs=dict(accelerator='auto', devices=1,
                                     enable_progress_bar=False, logger=False),
            )
        hook.remove()

        # TimeSeriesDataSet groups by series_id and emits batches in
        # encoder-index order, which is alphabetical for new categories — so
        # the dataloader silently re-orders rows relative to X. Recover the
        # input order from `pred.index`, which is in dataloader-output order.
        sids = pred.index["series_id"].astype(str).to_numpy()
        try:
            output_to_input = np.array([int(s.lstrip("v")) for s in sids])
        except ValueError as e:
            raise RuntimeError(
                f"Cannot parse series_id back to input row index. "
                f"Expected 'v<int>' (set in make_dataloader). Got: {sids[:5]}"
            ) from e
        inv_perm = np.argsort(output_to_input)

        samples_np = pred.output.numpy().transpose(0, 2, 1)[inv_perm]        # (n, M, T)
        n_params   = captured[0].shape[-1]

        # The hook fires once per decoder step (T_decoder steps per batch), so
        # `captured` has length `T * n_batches`. For each batch the T captures
        # are (batch_rows × M, 1, n_params) and concat along dim=1 gives the
        # batch's raw tensor of shape (batch_rows × M, T, n_params). The hook
        # captures in dataloader order, same as pred.output, so the same
        # inv_perm restores input order.
        T_dec = self._T
        assert len(captured) % T_dec == 0, (
            f"hook captured {len(captured)} tensors, expected multiple of T={T_dec}")
        batch_raws = []
        for b in range(len(captured) // T_dec):
            batch_t = torch.cat(captured[b * T_dec:(b + 1) * T_dec], dim=1)  # (B*M, T, n_p)
            Bb = batch_t.shape[0] // self._n_samples
            batch_raws.append(batch_t.reshape(Bb, self._n_samples, T_dec, n_params))
        raw = torch.cat(batch_raws, dim=0)[inv_perm]
        assert raw.shape[0] == n, f"raw row count {raw.shape[0]} ≠ input n={n}"
        return samples_np, raw.numpy()


class DeepARNormalImputer(DeepARImputer):
    """Entropy hierarchy for DeepAR with Normal output.

    Entropy levels:
        L1: H(Y_t | x)         = -E[log p̂(Y_t)]              Gaussian mixture entropy
        L2: H(Y_t | Y_{<t}, x) = ½ log(2πe) + E_M[log σ_t]   exact given paths
        L3: H(Y | x)           = Σ_t L2_t                     exact via chain rule
    """

    _n_params = 2

    def _compute_L1(self, dist: tuple) -> np.ndarray:
        samples, raw = dist                                                   # (n, M, T), (n, M, T, 2)
        mus   = raw[..., 0][:, None, :, :]                                    # (n, 1, M, T)
        sigma = np.log1p(np.exp(raw[..., 1]))[:, None, :, :]                  # softplus
        y     = samples[:, :, None, :]                                        # (n, M, 1, T)
        M = samples.shape[1]
        log_comps = norm.logpdf(y, loc=mus, scale=sigma)                      # (n, M, M, T)
        log_p     = logsumexp(log_comps, axis=2) - np.log(M)                  # (n, M, T)
        return -log_p.mean(axis=1)                                             # (n, T)

    def _compute_L2(self, dist: tuple) -> np.ndarray:
        _, raw = dist                                                         # (n, M, T, 2)
        log_sigma = np.log(np.log1p(np.exp(raw[..., 1])))                     # (n, M, T)
        return 0.5 * np.log(2 * np.pi * np.e) + log_sigma.mean(axis=1)        # (n, T)


class DeepARLogNormalImputer(DeepARImputer):
    """Entropy hierarchy for DeepAR with LogNormal output.

    The network outputs (loc=μ, scale_raw) of an underlying Normal; the
    LogNormal arises via ExpTransform.  Differential entropy:
        H(LogNormal(μ, σ)) = μ + log(σ) + ½ log(2πe)

    Entropy levels:
        L1: H(Y_t | x)         = -E[log p̂(Y_t)]              LogNormal mixture entropy
        L2: H(Y_t | Y_{<t}, x) = E_M[μ_t + log σ_t] + ½ log(2πe)
        L3: H(Y | x)           = Σ_t L2_t                     exact via chain rule
    """

    _n_params = 2

    @staticmethod
    def _scale_transform(raw_scale: np.ndarray) -> np.ndarray:
        """Match LogNormalDistributionLoss.rescale_parameters exactly:
        ``sigma = sigmoid(raw) + 1e-3`` (see entropy_shapley/losses.py)."""
        return expit(raw_scale) + 1e-3

    def _compute_L1(self, dist: tuple) -> np.ndarray:
        samples, raw = dist                                                   # (n, M, T), (n, M, T, 2)
        mus   = raw[..., 0][:, None, :, :]                                    # (n, 1, M, T)  μ of underlying Normal
        sigma = self._scale_transform(raw[..., 1])[:, None, :, :]             # (n, 1, M, T)
        y     = np.clip(samples[:, :, None, :], 1e-6, None)                   # (n, M, 1, T)
        M = samples.shape[1]
        # scipy.stats.lognorm parameterisation: s = σ, scale = exp(μ).
        log_comps = lognorm.logpdf(y, s=sigma, scale=np.exp(mus))             # (n, M, M, T)
        log_p     = logsumexp(log_comps, axis=2) - np.log(M)                  # (n, M, T)
        return -log_p.mean(axis=1)                                             # (n, T)

    def _compute_L2(self, dist: tuple) -> np.ndarray:
        _, raw = dist                                                         # (n, M, T, 2)
        mus       = raw[..., 0]                                               # (n, M, T)
        log_sigma = np.log(self._scale_transform(raw[..., 1]))                # (n, M, T)
        return (mus + log_sigma).mean(axis=1) + 0.5 * np.log(2 * np.pi * np.e)  # (n, T)


class DeepARStudentTImputer(DeepARImputer):
    """Entropy hierarchy for DeepAR with Student-T output.

    The network outputs (df_raw, loc, scale_raw); actual df = softplus(df_raw) + 2.
    Differential entropy (Lazo & Rathie 1978):
        H(t_ν(μ, σ)) = log(σ√ν · B(ν/2, ½)) - (ν+1)/2 · (ψ((ν+1)/2) - ψ(ν/2))

    Entropy levels:
        L1: H(Y_t | x)         = -E[log p̂(Y_t)]              StudentT mixture entropy
        L2: H(Y_t | Y_{<t}, x) = E_M[H(t_{ν_t}(μ_t, σ_t))]  per-component entropy
        L3: H(Y | x)           = Σ_t L2_t                     exact via chain rule
    """

    _n_params = 3

    def _compute_L1(self, dist: tuple) -> np.ndarray:
        samples, raw = dist                                                   # (n, M, T), (n, M, T, 3)
        df    = (np.log1p(np.exp(raw[..., 0])) + 2)[:, None, :, :]            # (n, 1, M, T)
        mus   = raw[..., 1][:, None, :, :]                                    # (n, 1, M, T)
        sigma = np.log1p(np.exp(raw[..., 2]))[:, None, :, :]                  # softplus, (n, 1, M, T)
        y     = samples[:, :, None, :]                                        # (n, M, 1, T)
        M = samples.shape[1]
        log_comps = student_t.logpdf(y, df=df, loc=mus, scale=sigma)          # (n, M, M, T)
        log_p     = logsumexp(log_comps, axis=2) - np.log(M)                  # (n, M, T)
        return -log_p.mean(axis=1)                                             # (n, T)

    def _compute_L2(self, dist: tuple) -> np.ndarray:
        _, raw = dist                                                         # (n, M, T, 3)
        df    = np.log1p(np.exp(raw[..., 0])) + 2                             # (n, M, T)
        sigma = np.log1p(np.exp(raw[..., 2]))                                 # (n, M, T)
        H     = student_t.entropy(df=df, scale=sigma)                         # (n, M, T)
        return H.mean(axis=1)                                                  # (n, T)
