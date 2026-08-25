# Sales Forecasting — Time Series Demo

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![statsmodels](https://img.shields.io/badge/statsmodels-SARIMA%20%7C%20Holt--Winters-orange)

A time-series forecasting pipeline comparing SARIMA against Holt-Winters Exponential Smoothing on monthly sales data, with trend detection, seasonality analysis, and a 12-period forward forecast.

**This is a self-contained demo, not a Power BI project.** It generates 3 years of realistic synthetic daily sales data on the fly (trend + yearly seasonality + noise, seeded for reproducibility) if no `sales_data.csv` is present, so it's runnable by anyone without a private dataset. The forecasting methodology transfers directly to real sales data with the same `date` / `sales_amount` columns.

## Run it yourself

```bash
pip install -r requirements.txt
python sales_forecaster.py
```

## What it does

1. Loads (or generates) daily sales data, aggregates to monthly totals
2. Computes 7- and 30-day moving averages, YoY growth rate, trend direction, and seasonal peak/trough months
3. Trains both SARIMA and Holt-Winters Exponential Smoothing on an 80/20 train/test split
4. Picks the better model by MAPE (mean absolute percentage error) and forecasts the next 12 periods

## Actual output (reproducible — run it and you'll get this)

| Model | MAE | RMSE | MAPE | Accuracy (100 − MAPE) |
|---|---|---|---|---|
| **Holt-Winters (best)** | **$37,607** | **$44,023** | **1.74%** | **98.3%** |
| SARIMA | $39,470 | $51,266 | 1.80% | 98.2% |

**Context that matters:** this accuracy is measured against synthetic data engineered with a smooth, well-behaved trend and seasonality pattern — real sales data with promotions, stockouts, and irregular demand shocks would show meaningfully more forecast error than this. The comparison methodology (train/test split, MAE/RMSE/MAPE side-by-side, picking the better model objectively) is the transferable part.

## Stack

Python, pandas, NumPy, statsmodels (SARIMAX, Holt-Winters), scikit-learn (metrics only).

## Author

[Deepanraj Arumugam](https://deeepanbe.github.io) — Data Analyst / BI Developer
