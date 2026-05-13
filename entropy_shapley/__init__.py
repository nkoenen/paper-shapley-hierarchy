"""A Hierarchy of Entropy-Shapley Games for Multivariate Predictive Uncertainty.

Reproduction library for the paper of the same name. The package implements
the three-level entropy-Shapley hierarchy from Section 4 of the paper, with
estimators that span the regimes from closed-form Gaussian outputs (4.2.1)
through factorized autoregressive likelihoods (4.2.2) to fully sample-based
predictive distributions (4.2.3).

Modules
-------

``game`` (Section 4.1)
    :class:`HierarchyGame` -- the three-level value-function game and the
    chain-rule / total-correlation derivations (Propositions 1, 2).

``imputer`` (Sections 4.2.1, 4.2.2)
    Analytical and factorized value-function imputers:

    * :class:`HierarchyImputer` -- abstract base, marginal Shapley path.
    * :class:`GaussianImputer` -- closed-form for multivariate Gaussian outputs.
    * :class:`DeepARImputer` -- abstract base for autoregressive likelihoods.
    * :class:`DeepARNormalImputer`,
      :class:`DeepARStudentTImputer`,
      :class:`DeepARLogNormalImputer` -- one per supported one-step likelihood.

``estimators`` (Section 4.2.3)
    Sample-based entropy estimators that turn a trajectory bag into the
    three-level value functions:

    * :class:`SampleImputer` -- abstract base, takes a ``predict_fn`` callable.
    * :class:`MVNSampleImputer` -- parametric Gaussian fit on samples.
    * :class:`KDECopulaImputer` -- KDE marginals + Gaussian copula
      (Algorithm in Appendix C).
    * :class:`MixtureCopulaImputer` -- analytical mixture marginals + Gaussian
      copula (validation only; not used in the paper figures).
    * :class:`KNNImputer` -- Kozachenko-Leonenko nearest-neighbour estimator.
    * :func:`evaluate_on_cached_samples` -- batch evaluation on a precomputed
      ``(samples, raw)`` cache.

``sampler``
    Background distribution samplers used for the Shapley imputation step:

    * :class:`MarginalTabularSampler` -- iid background draw.
    * :class:`BaselineTabularSampler` -- constant baseline.

``losses``
    :class:`StudentTDistributionLoss` and :class:`LogNormalDistributionLoss`,
    used by ``notebooks/3_train_deepar.py`` to train the non-Gaussian DeepAR
    likelihoods used in the Section 5.3 main comparison and the Appendix D.3.1
    validation.

``utils_datasets``, ``utils_deepar``, ``utils_paradigm``
    Pipeline helpers for the UCI Electricity experiments (Sections 5.3 and
    Appendix D.3.1): data loading and feature grouping, DeepAR training and
    analytical-reference helpers, and the DeepAR / Chronos trajectory
    pipeline that produces Figure 3.
"""

from .sampler import MarginalTabularSampler, BaselineTabularSampler
from .imputer import (
    HierarchyImputer,
    GaussianImputer,
    DeepARImputer,
    DeepARNormalImputer,
    DeepARStudentTImputer,
    DeepARLogNormalImputer,
)
from .estimators import (
    SampleImputer,
    MVNSampleImputer,
    KDECopulaImputer,
    MixtureCopulaImputer,
    KNNImputer,
    evaluate_on_cached_samples,
)
from .game import HierarchyGame

__all__ = [
    "MarginalTabularSampler",
    "BaselineTabularSampler",
    "HierarchyImputer",
    "GaussianImputer",
    "DeepARImputer",
    "DeepARNormalImputer",
    "DeepARStudentTImputer",
    "DeepARLogNormalImputer",
    "SampleImputer",
    "MVNSampleImputer",
    "KDECopulaImputer",
    "MixtureCopulaImputer",
    "KNNImputer",
    "evaluate_on_cached_samples",
    "HierarchyGame",
]
