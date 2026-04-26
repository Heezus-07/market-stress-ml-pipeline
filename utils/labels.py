import pandas as pd
import numpy as np
from typing import Literal

RET_Q = 0.01
VOL_Q = 0.90
VOL_WINDOW = 14


def _horizon_any_future(stress: pd.Series, horizon: int) -> pd.Series:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    shifted = pd.concat(
        [stress.shift(-i) for i in range(1, horizon + 1)],
        axis=1
    )
    return shifted.max(axis=1)


def generate_stress_targets_for_fold(
    es_df: pd.DataFrame,
    train_end: pd.Timestamp,
    horizon: Literal[1, 3, 5],
    ret_q: float = RET_Q,
    vol_q: float = VOL_Q,
) -> pd.Series:
    """
    Generate horizon-based stress labels using thresholds
    computed on the training window only.
    """

    df = es_df.copy()

    df["ret"] = np.log(df["close"]).diff()
    df["vol_14d"] = df["ret"].rolling(VOL_WINDOW).std()

    train = df[df["date"] <= train_end].dropna(subset=["ret", "vol_14d"])
    if train.empty:
        raise ValueError("Training window has no valid rows.")

    q_ret = train["ret"].quantile(ret_q)
    q_vol = train["vol_14d"].quantile(vol_q)

    stress_t = ((df["ret"] <= q_ret) | (df["vol_14d"] >= q_vol)).astype(int)
    stress_h = _horizon_any_future(stress_t, horizon)
    stress_h.index = df["date"]

    return stress_h
