# Market Stress Forecasting on ES Futures

A pipeline that predicts short-term stress events on the S&P 500 E-mini (ES) futures using market data, macro data, and news sentiment.

## What it does

For each trading day, the model predicts whether a stress event will happen in the next **1, 3, or 5 days**.

A stress event is a day with either:
- A very large drop (return in the bottom 1% of the training set), **or**
- A very high 14-day rolling volatility (top 10% of the training set).

The thresholds are fit on the training window only. No look-ahead.

## Data

| Source | What | Frequency |
|---|---|---|
| ES futures | Price, volume | Daily |
| DXY | US dollar index | Daily |
| VIX | Volatility index | Daily |
| FRED | Fed rate, US Treasury yields, TED spread, BBB spread, breakevens | Daily |
| GDELT + FinBERT | News sentiment (mean, std, % negative, stress ratio) | Daily |

Range: **2019 → 2025**.

## Pipeline

1. **`data_collection_and_cleaning.ipynb`** — pull raw data, clean, align to a daily calendar.
2. **`sentiment_analysis.ipynb`** — fetch GDELT news, filter, clean, run FinBERT, aggregate to daily sentiment.
3. **`feature_engineering_and_label_generation.ipynb`** — build features and stress labels for each fold.
4. **`model_training_and_evaluation.ipynb`** — walk-forward training, hold-out test on 2025, SHAP feature importance.
5. **`ablation_model_training.ipynb`** — same pipeline with feature groups removed, to measure each group's contribution.

Helper code:

```
utils/
├── features.py        # engineer_features_fold()
├── labels.py          # generate_stress_targets_for_fold()
├── garch_model.py     # GARCH baseline
└── garch_stress.py    # GARCH with rescaling for stress periods
```

Config: `model_config.yaml` (horizons, model grids, metrics, walk-forward settings).

## Models

- Logistic Regression (baseline)
- XGBoost
- CatBoost
- GARCH(p,q) — volatility baseline

All grids live in `model_config.yaml`.

## Validation

- **Walk-forward, expanding window**, starting from 2019.
- Test window: 1 year.
- Final out-of-sample test: **2025**.
- Primary metric: **PR-AUC** (data is imbalanced).
- Secondary: F1, recall, precision@k (k = 5%, 10%).

## How to run

### 1. Install
```bash
pip install -r requirements.txt
```

Main libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `catboost`, `lightgbm`, `arch`, `shap`, `fredapi`, `gdeltdoc`, `transformers`, `torch`.

### 2. Set keys
Create a `.env` file in the repo root:
```
FRED_KEY=your_fred_api_key
```

### 3. Run notebooks in order
```
1. data_collection_and_cleaning.ipynb
2. sentiment_analysis.ipynb
3. feature_engineering_and_label_generation.ipynb
4. model_training_and_evaluation.ipynb
5. ablation_model_training.ipynb     (optional)
```

## Folder layout

```
.
├── notebooks/
│   ├── data_collection_and_cleaning.ipynb
│   ├── sentiment_analysis.ipynb
│   ├── feature_engineering_and_label_generation.ipynb
│   ├── model_training_and_evaluation.ipynb
│   └── ablation_model_training.ipynb
├── utils/
│   ├── features.py
│   ├── labels.py
│   ├── garch_model.py
│   └── garch_stress.py
├── data/
│   ├── raw/          # downloaded CSVs
│   ├── processed/    # cleaned and merged
│   ├── models/       # fitted models
│   └── ablation/     # ablation results
├── config/
│   └── model_config.yaml
├── .env              # not committed
└── README.md
```

## Notes

- All thresholds, scalers, and quantiles are fit on training data only.
- Sentiment features are lagged by 1 and 2 days to avoid using same-day news.
- Burn-in of ~25 rows is dropped at the start of each fold to clear rolling-window NaNs.

## References

- Holló, Kremer & Lo Duca (2012). *CISS — A Composite Indicator of Systemic Stress in the Financial System.*
- Kritzman & Li (2010). *Skulls, Financial Turbulence, and Risk Management.*

## License

MIT.
