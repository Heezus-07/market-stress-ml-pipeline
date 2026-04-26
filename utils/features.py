import numpy as np
import pandas as pd

MAX_LOOKBACK = 20
MAX_LAG = 2
BURN_IN = MAX_LOOKBACK + MAX_LAG + 5


def engineer_features_fold(df_train, df_test):
    
    df_all = pd.concat([df_train, df_test], axis=0).copy()
    df_all = df_all.sort_values("date")

    # ------------------------------------------------
    # FORWARD-FILL BASE DATA
    # ------------------------------------------------
    
    fill_cols = ["vix_close", "FED_RATE_USD", "US10Y", "US2Y", "US3M",
                 "YIELD_CURVE_SLOPE", "TED_SPREAD", "BBB_SPREAD", "T10Y_IE"]
    
    for col in fill_cols:
        if col in df_all.columns:
            df_all[col] = df_all[col].ffill()

    # ------------------------------------------------
    # PRICE SANITISATION
    # ------------------------------------------------

    es_price  = df_all["es_close"].clip(lower=1e-8)
    dxy_price = df_all["dxy_close"].clip(lower=1e-8)
    vix_price = df_all["vix_close"].clip(lower=1e-8)

    # ------------------------------------------------
    # ES FEATURES
    # ------------------------------------------------

    df_all["es_ret"] = np.log(es_price).diff()
    df_all["es_abs_ret"] = df_all["es_ret"].abs()
    df_all["es_vol_7d"]  = df_all["es_ret"].rolling(7).std()
    df_all["es_vol_14d"] = df_all["es_ret"].rolling(14).std()
    df_all["es_ret_mean_5d"]  = df_all["es_ret"].rolling(5).mean()
    df_all["es_ret_mean_10d"] = df_all["es_ret"].rolling(10).mean()
    df_all["es_log_volume"] = np.log(df_all["es_volume"].replace(0, np.nan))
    df_all["es_log_volume_chg"] = df_all["es_log_volume"].diff()

    vol_mu = df_all["es_log_volume"].rolling(20).mean()
    vol_sd = df_all["es_log_volume"].rolling(20).std()
    df_all["es_volume_z"] = (df_all["es_log_volume"] - vol_mu) / vol_sd

    ret_mu = df_all["es_ret"].rolling(20).mean()
    ret_sd = df_all["es_ret"].rolling(20).std()
    df_all["es_ret_z_20"] = (df_all["es_ret"] - ret_mu) / ret_sd

    df_all["vol_ratio_7_14"] = (
        df_all["es_vol_7d"] / df_all["es_vol_14d"].replace(0, np.nan)
    )

    # ------------------------------------------------
    # DXY FEATURES
    # ------------------------------------------------

    df_all["dxy_ret"] = np.log(dxy_price).diff()
    df_all["dxy_vol_7d"]  = df_all["dxy_ret"].rolling(7).std()
    df_all["dxy_vol_14d"] = df_all["dxy_ret"].rolling(14).std()

    mu = df_all["dxy_ret"].rolling(20).mean()
    sd = df_all["dxy_ret"].rolling(20).std()
    df_all["dxy_ret_z"] = (df_all["dxy_ret"] - mu) / sd

    # ------------------------------------------------
    # VIX FEATURES
    # ------------------------------------------------

    df_all["vix_log"] = np.log(vix_price)
    df_all["vix_chg"] = df_all["vix_log"].diff()
    df_all["vix_mean_5d"]  = df_all["vix_close"].rolling(5).mean()
    df_all["vix_mean_10d"] = df_all["vix_close"].rolling(10).mean()

    mu = df_all["vix_close"].rolling(20).mean()
    sd = df_all["vix_close"].rolling(20).std()
    df_all["vix_z"] = (df_all["vix_close"] - mu) / sd

    # ------------------------------------------------
    # MACRO FEATURES
    # ------------------------------------------------

    macro_cols = [
        "FED_RATE_USD", "US10Y", "US2Y", "US3M",
        "YIELD_CURVE_SLOPE", "TED_SPREAD", "BBB_SPREAD", "T10Y_IE"
    ]

    for col in macro_cols:
        df_all[f"{col}_chg_5d"] = df_all[col].diff(5)

    # ------------------------------------------------
    # SENTIMENT FEATURES
    # ------------------------------------------------

    df_all["sentiment_change"] = df_all["sentiment_mean"].diff()
    df_all["stress_ratio_change"] = df_all["stress_ratio"].diff()

    for w in [5, 10]:
        df_all[f"sentiment_mean_roll_{w}"]  = df_all["sentiment_mean"].rolling(w).mean()
        df_all[f"sentiment_std_roll_{w}"]   = df_all["sentiment_mean"].rolling(w).std()
        df_all[f"stress_ratio_roll_{w}"]    = df_all["stress_ratio"].rolling(w).mean()

    mu = df_all["sentiment_mean"].rolling(10).mean()
    sd = df_all["sentiment_mean"].rolling(10).std()
    df_all["sentiment_zscore_10"] = (df_all["sentiment_mean"] - mu) / sd

    for col in ["sentiment_mean", "sentiment_std", "pct_negative",
            "sentiment_change", "sentiment_zscore_10",
            "stress_ratio_change"]:
        df_all[f"{col}_lag1"] = df_all[col].shift(1)
        df_all[f"{col}_lag2"] = df_all[col].shift(2)

    # ------------------------------------------------
    # BURN-IN / CLEAN
    # ------------------------------------------------

    first_valid = df_all.dropna().index.min()
    df_all = df_all.loc[first_valid:].copy()

    df_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_all.dropna(inplace=True)

    # ------------------------------------------------
    # SPLIT BY DATE
    # ------------------------------------------------

    train_dates = df_train["date"].values
    test_dates = df_test["date"].values

    X_train = df_all[df_all["date"].isin(train_dates)].copy()
    X_test = df_all[df_all["date"].isin(test_dates)].copy()

    return X_train, X_test