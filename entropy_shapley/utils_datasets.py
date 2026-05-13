"""Dataset utilities and setup wrappers for the UCI Electricity experiments.

Two pipelines live here:

1. **Standard training/eval setup.** ``build_long_dataframe`` turns the wide
   hourly CSV into a long-format dataframe (one row per (series, timestamp)),
   optionally re-aggregated to coarser blocks; ``build_datasets`` builds
   disjoint train / validation / test ``TimeSeriesDataSet`` splits with
   rolling-origin sampling on the validation and test windows.

2. **Shapley imputation pipeline.** ``build_feature_vec`` extracts the
   ``(encoder targets, hour, weekday, month)`` representation that
   ``HierarchyImputer`` operates on, ``build_feature_groups`` partitions those
   features into the six Shapley players used in Section 5.3 / Appendix D.3,
   and ``make_dataloader_factory`` rebuilds ``TimeSeriesDataSet`` DataLoaders
   from imputed (i.e. *synthetic*) feature vectors — these histories don't
   correspond to any real timestamp in the dataset, so a thin synthetic-row
   shim is needed to feed them back through ``DeepAR.predict``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer, NaNLabelEncoder



# ── Long-form dataframe ──────────────────────────────────────────────────────
def build_long_dataframe(data_path: str,
                         start_date: str,
                         n_series: int,
                         drop_series: str | set[str] | None,
                         blocks_per_day: int) -> pd.DataFrame:
    """Wide CSV -> long format, drop dead series, per-series mean normalization.

    blocks_per_day < 24: resample hourly raw data to (24/blocks_per_day)-hour
    block means before building the long format. The "hour" column then encodes
    block-of-day instead of hour-of-day. blocks_per_day=24 leaves data hourly.
    """
    if 24 % blocks_per_day != 0:
        raise ValueError(
            f"blocks_per_day ({blocks_per_day}) must divide 24 evenly")
    block_hours = 24 // blocks_per_day

    if drop_series is None:
        drop = set()
    elif isinstance(drop_series, set):
        drop = drop_series
    else:
        drop = {drop_series}
    raw = pd.read_csv(data_path, index_col=0, parse_dates=True)
    series_cols = [str(i) for i in range(n_series) if str(i) not in drop]
    df = raw[series_cols].loc[start_date:]

    if blocks_per_day < 24:
        df = df.resample(f'{block_hours}h').mean()

    records = []
    for col in series_cols:
        tmp = df[[col]].rename(columns={col: "target"})
        if blocks_per_day < 24:
            # block-of-day encoding (e.g. for 2h blocks: 0,1,2,...,11)
            tmp["hour"] = (tmp.index.hour // block_hours).astype(str)
        else:
            tmp["hour"] = tmp.index.hour.astype(str)
        tmp["weekday"]   = tmp.index.dayofweek.astype(str)
        tmp["month"]     = (tmp.index.month - 1).astype(str)
        tmp["series_id"] = col
        tmp = tmp.reset_index(drop=False)
        records.append(tmp)

    df_long = pd.concat(records, ignore_index=True)
    timestamps = (df_long["index"].drop_duplicates()
                                  .sort_values()
                                  .reset_index(drop=True))
    df_long["time_idx"] = df_long["index"].map({ts: i for i, ts in enumerate(timestamps)})
    df_long["target"]   = (df_long.groupby("series_id")["target"]
                                   .transform(lambda x: x / x.mean()))
    return df_long


def _build_dataset_kwargs(df_long: pd.DataFrame, *, context: int, t_horizon: int) -> dict:
    """``TimeSeriesDataSet`` kwargs with categorical encoders fitted on ``df_long``."""
    return dict(
        time_idx="time_idx", target="target", group_ids=["series_id"],
        max_encoder_length=context, max_prediction_length=t_horizon,
        target_normalizer=EncoderNormalizer(method="identity",
                                            transformation=None, center=False),
        scalers=None,
        time_varying_known_categoricals=["hour", "weekday", "month"],
        time_varying_unknown_reals=["target"],
        categorical_encoders={
            "series_id": NaNLabelEncoder(add_nan=True ).fit(df_long.series_id),
            "hour":      NaNLabelEncoder(add_nan=False).fit(df_long.hour),
            "weekday":   NaNLabelEncoder(add_nan=False).fit(df_long.weekday),
            "month":     NaNLabelEncoder(add_nan=False).fit(df_long.month),
        },
    )


def build_datasets(
    df_long: pd.DataFrame,
    *,
    context: int,
    t_horizon: int,
    validation_period: int,
    test_period: int | None = None,
):
    """Disjoint training / validation / test ``TimeSeriesDataSet``s.

    Layout (in ``time_idx``, with VP=``validation_period``, TP=``test_period``;
    ``test_period`` defaults to ``validation_period``)::

        train: time_idx <= train_cutoff
        val:   forecast start in (train_cutoff, val_cutoff],   stride = t_horizon
        test:  forecast start in (val_cutoff,   max_time_idx], stride = t_horizon

    Validation and test windows are temporally disjoint and use rolling-origin
    sampling with ``t_horizon`` as the stride (no overlapping windows).
    """
    if test_period is None:
        test_period = validation_period

    max_t = int(df_long["time_idx"].max())
    val_cutoff   = max_t - test_period
    train_cutoff = val_cutoff - validation_period

    df_train = df_long[df_long["time_idx"] <= train_cutoff]
    df_val   = df_long[df_long["time_idx"] <= val_cutoff]

    kw = _build_dataset_kwargs(df_long, context=context, t_horizon=t_horizon)
    training = TimeSeriesDataSet(df_train, **kw)

    val_first = train_cutoff + 1
    validation = TimeSeriesDataSet.from_dataset(
        training, df_val,
        min_prediction_idx=val_first,
        stop_randomization=True,
    )
    validation = validation.filter(
        lambda idx: (idx["time_idx_first_prediction"] - val_first) % t_horizon == 0
    )

    test_first = val_cutoff + 1
    test = TimeSeriesDataSet.from_dataset(
        training, df_long,
        min_prediction_idx=test_first,
        stop_randomization=True,
    )
    test = test.filter(
        lambda idx: (idx["time_idx_first_prediction"] - test_first) % t_horizon == 0
    )

    return training, validation, test, train_cutoff, val_cutoff

def build_feature_groups(context: int, blocks_per_day: int) -> list[list[int]]:
    """Index groups for the six Shapley players used in Section 5.3 / Appendix D.3.

    Feature-vector layout (matches :func:`build_feature_vec`):
        ``x[0:context]``           = encoder targets (the history),
        ``x[context..context+2]``  = ``(hour, weekday, month)`` at the prediction edge.

    The 87 raw scalar inputs (CONTEXT=84 history blocks + 3 calendar features
    on the electricity setup) are grouped into six Shapley players:
        - 3 calendar players (hour, weekday, month)
        - ``recent``: last 24 hours of history (``= blocks_per_day`` blocks)
        - ``middle``: the three preceding days
        - ``old``: the remainder of the week-long context
    See Appendix D.3 "Shapley setup" of the paper.
    """
    bpd = blocks_per_day
    return [
        [context],
        [context + 1],
        [context + 2],
        list(range(context - bpd,         context)),            # recent (last day)
        list(range(context - 4 * bpd,     context - bpd)),      # middle (3 preceding days)
        list(range(0,                     context - 4 * bpd)),  # old (remainder)
    ]


def build_completed_inputs(x_explain: np.ndarray, bg: np.ndarray,
                           coalitions: np.ndarray,
                           feature_groups: list[list[int]],
                           p_feat: int) -> np.ndarray:
    """Apply Shapley coalition masks to build the flat (I·n_coal·K, p_feat) batch.

    For every (instance, coalition, background) triple this is the canonical
    marginal-Shapley imputation step: features inside the coalition keep the
    instance value ``x_explain[i, j]``; features outside the coalition are
    overwritten by the background value ``bg[k, j]``.

    Group-level coalition bits are first expanded to feature-level masks via
    ``feature_groups`` — each player ``g`` corresponds to the index list
    ``feature_groups[g]``.

    Parameters
    ----------
    x_explain : (I, p_feat)         instance feature vectors
    bg        : (K, p_feat)         background feature vectors
    coalitions: (n_coal, n_groups)  bool coalition matrix over Shapley players
    feature_groups : list[list[int]]
        Maps each Shapley player to its raw-feature indices.
    p_feat    : int                 total number of raw features

    Returns
    -------
    (I · n_coal · K, p_feat)
        Row layout: outer axis is instance ``i``, then coalition ``s``, then
        background ``k``.
    """
    I = x_explain.shape[0]
    K = bg.shape[0]
    n_coal = coalitions.shape[0]
    feat_mask = np.zeros((n_coal, p_feat), dtype=bool)
    for g, idxs in enumerate(feature_groups):
        feat_mask[:, idxs] = coalitions[:, g:g + 1]
    completed = np.where(
        feat_mask[None, :, None, :],          # (1, n_coal, 1, p_feat)
        x_explain[:, None, None, :],          # (I, 1,      1, p_feat)
        bg[None, None, :, :],                 # (1, 1,      K, p_feat)
    )                                          # (I, n_coal, K, p_feat)
    return completed.reshape(I * n_coal * K, p_feat)


def build_feature_vec(df_long: pd.DataFrame, series_id: str,
                      pred_time_idx: int, context: int) -> np.ndarray:
    """Single feature vector: ``[encoder targets (context), hour, weekday, month]``."""
    enc = df_long[
        (df_long["series_id"] == series_id) &
        (df_long["time_idx"].between(pred_time_idx - context, pred_time_idx - 1))
    ].sort_values("time_idx")
    assert len(enc) == context, f"Expected {context} rows, got {len(enc)}"
    targets  = enc["target"].values
    hour_val = int(enc["hour"].iloc[-1])
    wd_val   = int(enc["weekday"].iloc[-1])
    mo_val   = int(enc["month"].iloc[-1])
    return np.concatenate([targets, [hour_val, wd_val, mo_val]])

# ── Dataloader factory ───────────────────────────────────────────────────────
def make_dataloader_factory(training, df_long: pd.DataFrame,
                            context: int, t_horizon: int,
                            blocks_per_day: int):
    """Build DataLoaders from synthetic Shapley-imputed feature vectors.

    Returns ``make_dataloader(X, batch_size=None)`` that takes feature
    vectors ``X`` of shape ``(n, context + 3)`` — first ``context`` columns
    are encoder targets, last three are ``(hour, weekday, month)`` — and
    produces a ``DataLoader`` suitable for ``DeepAR.predict``.

    These feature vectors come out of the Shapley imputation step
    (see :class:`HierarchyImputer`): they generally do *not* correspond to
    any real timestamp in ``df_long``, so we cannot just look them up. We
    instead build a thin synthetic row per input by:

    1. Picking a *calendar template* — any real ``(time_idx, hour, weekday,
       month)`` window from the first available series that matches the
       requested calendar triplet (cached in ``get_cal_template``; the same
       triplet repeats often across coalitions and backgrounds, so the cache
       hit rate is very high). If the safe time window contains no match,
       falls back to the first row of the template series that has enough
       history — i.e., we lose the calendar match but keep the structure.
    2. Stamping the synthetic history with ``series_id = "v{i}"`` where
       ``i`` is the input row index. This convention is recovered downstream
       by :meth:`DeepARImputer._predict` (see ``imputer.py``) to undo the
       row reordering that ``TimeSeriesDataSet`` performs internally.
    3. Overwriting the encoder targets with the supplied imputed history.

    Calendar templates are looked up against the *first* available series in
    ``df_long`` — this is sufficient because the calendar features are
    series-agnostic.
    """
    cache: dict = {}
    template_sid = df_long["series_id"].iloc[0]
    max_tidx     = df_long["time_idx"].max()

    def get_cal_template(hour: int, weekday: int, month: int):
        key = (hour, weekday, month)
        if key not in cache:
            cands = df_long[
                (df_long["series_id"] == template_sid) &
                (df_long["hour"]      == str(hour)) &
                (df_long["weekday"]   == str(weekday)) &
                (df_long["month"]     == str(month)) &
                (df_long["time_idx"].between(context, max_tidx - t_horizon))
            ]
            if len(cands) == 0:
                cands = df_long[(df_long["series_id"] == template_sid) &
                                (df_long["time_idx"] >= context)]
            pred_tidx = int(cands["time_idx"].iloc[0])
            base = df_long[
                (df_long["series_id"] == template_sid) &
                (df_long["time_idx"].between(pred_tidx - context,
                                             pred_tidx + t_horizon - 1))
            ][["time_idx", "hour", "weekday", "month", "target"]].copy()
            cache[key] = (base, pred_tidx)
        return cache[key]

    def make_dataloader(X: np.ndarray, batch_size: int | None = None):
        bs = batch_size if batch_size is not None else len(X)
        frames = []
        for i, row in enumerate(X):
            enc_targets = row[:context]
            hour    = int(round(row[context]))     % blocks_per_day
            weekday = int(round(row[context + 1])) % 7
            month   = int(round(row[context + 2])) % 12
            base, pred_tidx = get_cal_template(hour, weekday, month)
            s = base.copy()
            s["series_id"] = f"v{i}"
            s.loc[s["time_idx"] < pred_tidx, "target"] = enc_targets
            frames.append(s)
        ds = TimeSeriesDataSet.from_dataset(
            training, pd.concat(frames, ignore_index=True),
            predict=True, stop_randomization=True,
        )
        return ds.to_dataloader(train=False, batch_size=bs, num_workers=0)

    return make_dataloader