##################################################################################
#                               shapiq wrapper 
#  that bundles the three hierarchy levels into a single imputer pass.
#
# ``HierarchyGame`` wraps a ``HierarchyImputer`` whose ``__call__`` returns the
# value functions for all three levels (Level 1 marginal, Level 2 sequential,
# Level 3 joint) simultaneously. ``precompute()`` evaluates all 2^p coalitions
# in **one** imputer call and caches the resulting value-function tables;
# subsequent ``game_L1(t)`` / ``game_L2(t)`` / ``game_L3()`` factory methods
# return lightweight ``shapiq.Game`` sub-games that read from those tables.
# Each sub-game is then passed to any shapiq Shapley estimator
# (``exact_values`` for small p, or an approximator like
# ``PermutationSamplingSV`` for large p).
#
# The expensive part — the 2^p model evaluations — runs *once* and is shared
# across all 3 × T + 1 sub-games; Shapley extraction per sub-game is a cheap
# Möbius transform over the cached coalition table.
###################################################################################
import numpy as np
import shapiq

from .imputer import HierarchyImputer


class _SubGame(shapiq.Game):
    """A single-level/horizon game backed by a HierarchyImputer.

    Can operate in two modes:
      - on-the-fly: calls imputer(coalitions) each time value_function is called
      - cached:     uses a precomputed lookup table (set _cache after precompute)
    """

    def __init__(self, imputer: HierarchyImputer, level: str,
                 t: int | None = None, inst_idx: int = 0):
        super().__init__(n_players=imputer.n_players, normalize=False)
        self._imputer  = imputer
        self._level    = level      # 'L1', 'L2', or 'L3'
        self._t        = t          # horizon index for L1/L2
        self._inst_idx = inst_idx   # which instance to explain
        self._cache: np.ndarray | None = None

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        if self._cache is not None:
            idx = (coalitions * (1 << np.arange(self.n_players))).sum(1).astype(int)
            return self._cache[idx]
        v_L1, v_L2, v_L3 = self._imputer(coalitions)
        # v_L1: (n_inst, n_coal, T), v_L3: (n_inst, n_coal)
        i = self._inst_idx
        if self._level == 'L1':
            return v_L1[i, :, self._t]
        if self._level == 'L2':
            return v_L2[i, :, self._t]
        return v_L3[i, :]


class HierarchyGame:
    """Coordinator that creates per-level/horizon sub-games from a HierarchyImputer.

    Supports single and multi-instance imputers.  After ``precompute()``, pass
    ``inst_idx`` to ``game_L1`` / ``game_L2`` / ``game_L3`` to get the sub-game
    for a specific instance (default: 0).

    Usage — single instance (backward-compatible)
    -----
        game = HierarchyGame(imputer).precompute()
        phi  = game.game_L1(t).exact_values("SII", order=1).values[1:]

    Usage — multiple instances
    -----
        imputer = GaussianImputer(model, sampler, X_instances, T)
        game    = HierarchyGame(imputer).precompute()
        for i in range(n_inst):
            phi_L1 = np.stack([game.game_L1(t, i).exact_values("SII", order=1).values[1:]
                               for t in range(T)])

    Approximate (large p, any shapiq approximator):
        game   = HierarchyGame(imputer)
        approx = shapiq.approximator.PermutationSamplingSV(n=p)
        phi    = approx(budget=512, game=game.game_L1(t)).values[1:]
    """

    def __init__(self, imputer: HierarchyImputer):
        self._imputer = imputer
        self._v_L1: np.ndarray | None = None
        self._v_L2: np.ndarray | None = None
        self._v_L3: np.ndarray | None = None

    def precompute(self) -> "HierarchyGame":
        """Evaluate all 2^p coalitions in a single imputer call and cache results."""
        p = self._imputer.n_players
        all_coal = np.array(
            [[(s >> j) & 1 for j in range(p)] for s in range(2 ** p)], dtype=bool
        )
        self._v_L1, self._v_L2, self._v_L3 = self._imputer(all_coal)
        return self

    def _sub(self, level: str, t: int | None = None, inst_idx: int = 0) -> _SubGame:
        g = _SubGame(self._imputer, level, t, inst_idx)
        table = {'L1': self._v_L1, 'L2': self._v_L2, 'L3': self._v_L3}[level]
        if table is not None:
            # table may be (n_inst, n_coal, T)/(n_inst, n_coal) from HierarchyImputer,
            # or (n_coal, T)/(n_coal,) from a custom imputer without the n_inst axis.
            # Detect by ndim: L3 has n_inst when ndim>=2; L1/L2 when ndim>=3.
            has_inst_dim = (level == 'L3' and table.ndim >= 2) or \
                           (level != 'L3' and table.ndim >= 3)
            row = table[inst_idx] if has_inst_dim else table  # (n_coal, T) or (n_coal,)
            g._cache = row[:, t] if t is not None else row
        return g

    def game_L1(self, t: int, inst_idx: int = 0) -> _SubGame:
        """Sub-game for H(Y_t | x) at horizon t."""
        return self._sub('L1', t, inst_idx)

    def game_L2(self, t: int, inst_idx: int = 0) -> _SubGame:
        """Sub-game for H(Y_t | Y_{<t}, x) at horizon t."""
        return self._sub('L2', t, inst_idx)

    def game_L3(self, inst_idx: int = 0) -> _SubGame:
        """Sub-game for H(Y | x)."""
        return self._sub('L3', inst_idx=inst_idx)
