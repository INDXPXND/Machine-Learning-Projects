# Retail Store Sales Forecasting

A machine learning project for predicting weekly retail sales across stores and departments.

## Dataset

3 CSV files covering 50 stores, 20 departments, period 2022–2024:

- `sales.csv` — weekly sales (`store_id`, `department`, `date`, `weekly_sales`, `is_holiday`)
- `stores.csv` — store metadata (type A/B/C, size, region)
- `features.csv` — external factors (temperature, fuel price, CPI, unemployment, markdowns, holidays)

## Target

`weekly_sales` — revenue for a specific department in a specific store for a given week.

## Feature Engineering

- Date features: year, month, week number, quarter, day of year
- **Lag features**: sales of the same store+dept pair 1 week ago (`lag_1`), 4 weeks (`lag_4`), 1 year ago (`lag_52`)
- 4-week rolling mean (`rolling_mean_4`)
- Sum of all markdown discounts (`total_markdown`)
- Label Encoding for categorical features (store type, region, season)
- `log1p` transformation of the target to reduce the effect of outliers

## Train/Test Split

Time-based split — last 3 months (October–December 2024) used as test set. Random split is not suitable for time series due to data leakage through lag features.

## Models

Trained and compared three gradient boosting models:

| Model | R² | MAE | RMSE | MAPE |
|---|---|---|---|---|
| XGBoost | 0.557 | 30 941 | 44 104 | 66.2% |
| CatBoost | 0.558 | 30 801 | 43 600 | 66.8% |
| LightGBM | — | — | — | — |

Also tuned CatBoost hyperparameters using `HalvingRandomSearchCV`.

## Stack

```
pandas / numpy / matplotlib / seaborn
scikit-learn (HalvingRandomSearchCV, metrics)
xgboost / catboost / lightgbm
```
etail_Store_Sales_Forecasting.ipynb in Jupyter or Google Colab
```