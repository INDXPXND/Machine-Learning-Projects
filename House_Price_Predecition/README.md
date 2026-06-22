# House Price Prediction

Regression project for predicting house prices using an ensemble of gradient boosting models.

## Preprocessing

- Missing values in numeric columns filled with **median**
- Missing values in categorical columns filled with **mode**
- Log transformation (`log1p`) applied to `Price` and `Area_SqFt`
- Categorical columns encoded with **One-Hot Encoding**

## Models

Four models trained and compared:

- `GradientBoostingRegressor`
- `XGBRegressor` (XGBoost) — with L2 regularization (`reg_lambda=1`)
- `CatBoostRegressor` — with L2 regularization (`l2_leaf_reg=5`)
- `VotingRegressor` — ensemble of all three

All models: `n_estimators=1000`, `learning_rate=0.01`, `max_depth=6`

## Results

| Model | Train R² | Test R² | MAE | RMSE | MAPE |
|---|---|---|---|---|---|
| VotingRegressor | 0.9876 | 0.9305 | 25 449 | 34 290 | 4.42% |
| XGBRegressor | 0.9929 | 0.9144 | 28 397 | 39 383 | 4.89% |
| **CatBoostRegressor** | **0.9543** | **0.9377** | **23 911** | **36 289** | **4.07%** |
| GradientBoostingRegressor | 0.9967 | 0.9143 | 28 054 | 37 265 | 4.91% |

**Best model: CatBoostRegressor** — highest test R², lowest MAE and MAPE, least overfitting (train/test gap only 1.7%).

## Usage

```python
import joblib
import numpy as np

model = joblib.load('CatBoostRegressor.pkl')

# Input features must be preprocessed the same way as during training
prediction_log = model.predict(X)
prediction = np.expm1(prediction_log)
```

## Saved Models

| File | Model |
|---|---|
| `CatBoostRegressor.pkl` | Best single model |
| `VotingRegressor.pkl` | Ensemble |
| `XGBoostRegressor.pkl` | XGBoost |
| `GradientBoostingRegressor.pkl` | Sklearn GBR |

## Stack

`pandas` `numpy` `scikit-learn` `xgboost` `catboost` `matplotlib` `seaborn` `joblib`