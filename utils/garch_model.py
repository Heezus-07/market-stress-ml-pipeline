import numpy as np
import pandas as pd
from arch import arch_model
import warnings
from arch.univariate.base import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _forecast_variance(fitted_model, horizon: int):
    """
    Return h-step ahead variance forecast (scalar).
    """
    f = fitted_model.forecast(horizon=horizon, reindex=False)
    # last row, horizon column
    return f.variance.values[-1, horizon - 1]


def run_garch_fold(
    es_df: pd.DataFrame,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    horizon: int,
    p: int = 1,
    q: int = 1,
    distribution: str = "normal",
):
    """
    Rolling / expanding walk-forward GARCH.
    Returns volatility forecasts aligned with test dates.
    """

    df = es_df.copy().sort_values("date")
    df["ret"] = np.log(df["close"]).diff()
    df = df.dropna(subset=["ret"])

    train_df = df[df["date"] <= train_end].copy()
    test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train or test window empty.")

    forecasts = []

    expanding_returns = train_df["ret"].values

    for i in range(len(test_df)):

        model = arch_model(
            expanding_returns,
            mean="Zero",
            vol="GARCH",
            p=p,
            q=q,
            dist=distribution,
            rescale=False,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = model.fit(disp="off")

        var_forecast = _forecast_variance(fitted, horizon=horizon)
        vol_forecast = np.sqrt(var_forecast)

        forecasts.append(vol_forecast)

        # expand window with realized return
        expanding_returns = np.append(
            expanding_returns,
            test_df["ret"].iloc[i]
        )

    out = pd.DataFrame({
        "date": test_df["date"].values,
        "garch_vol_forecast": forecasts
    })
    
    return out