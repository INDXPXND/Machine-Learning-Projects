# Medical Insurance Cost Prediction

Predicting an individual's `annual_medical_cost` from demographic, lifestyle, health, and
insurance-policy attributes, using a synthetic dataset of 100,000 insured people.

## Dataset

`datasets/medical_insurance.csv` — 100,000 rows, 54 columns, no duplicates. One column
(`alcohol_freq`) has ~30% missing values, treated as "no alcohol use" (filled with `"None"`).

Feature groups:
- **Demographics**: age, sex, region, urban/rural, income, education, marital status, employment, household size, dependents
- **Lifestyle / vitals**: BMI, smoking status, alcohol frequency, blood pressure, LDL, HbA1c
- **Health history**: chronic condition count and flags (hypertension, diabetes, asthma, COPD, cardiovascular disease, cancer, kidney/liver disease, arthritis, mental health), hospitalizations, visits, procedure counts
- **Policy**: plan type, network tier, deductible, copay, policy term, provider quality, risk score
- **Target**: `annual_medical_cost`

## Key finding: the original model was leaking the target

An earlier version of the notebook reported **R2 ≈ 0.996**, which is unrealistic for this kind
of prediction problem. The cause was target leakage: several columns are computed from (or
priced directly off) `annual_medical_cost` itself and would not be known in advance in a real
underwriting scenario:

| Column | Correlation with target | Why it's leakage |
|---|---|---|
| `annual_premium` / `monthly_premium` | 0.965 | Premium is priced directly off the realized cost in this dataset |
| `total_claims_paid` | 0.739 | Sum of claims paid out — a direct consequence of the cost |
| `avg_claim_amount` | 0.633 | Derived from the same claims |
| `claims_count` | 0.179 | Also derived from claims activity tied to the cost |

These five columns were **dropped**. `risk_score` and `is_high_risk` were kept — they
correlate mainly with `age`, `chronic_count`, and blood pressure (legitimate underwriting-time
signals), not with the outcome itself.

Once leakage is removed, a realistic R2 ceiling of **~0.18** emerges, which held constant
across every model tried (see below) and under 5-fold cross-validation. This means the ceiling
is set by how much genuine signal the legitimate features carry, not by model choice —
pushing R2 meaningfully higher would require richer inputs (e.g. diagnosis codes, prior-year
costs) that aren't present in this dataset.

## Methodology

1. Clean data (drop duplicates, impute `alcohol_freq`, drop identifier column).
2. Correlation analysis to identify and remove leakage columns.
3. Log1p-transform `income` (heavily right-skewed input feature).
   - Log-transforming the *target* was also tested but consistently made every model worse
     (CatBoost R2 dropped from ~0.18 to ~0.11), so the target is kept on its raw dollar scale.
4. Feature engineering: a few interaction terms (`age × chronic_count`, total utilization,
   `smoker × BMI`, `hypertension × systolic_bp`).
5. Compare four models on an 80/20 train/test split: Linear Regression, Random Forest,
   HistGradientBoosting, and CatBoost (native categorical handling).
6. Tune and validate the best model (CatBoost) with 5-fold cross-validation.
7. Evaluate with R2, MAE, RMSE, and MAPE; inspect feature importance, actual-vs-predicted, and
   residuals.

## Results

| Model | R2 | MAE ($) | RMSE ($) | MAPE |
|---|---|---|---|---|
| **CatBoost** | **0.182** | **1765** | **2837** | 1.006 |
| Linear Regression | 0.180 | 1769 | 2840 | 1.000 |
| HistGradientBoosting | 0.176 | 1765 | 2847 | 1.010 |
| Random Forest | 0.176 | 1772 | 2847 | 1.008 |

CatBoost is the best of the four and was selected as the final model.

**5-fold cross-validation (CatBoost):** R2 = 0.174 ± 0.010 — confirms the result is stable
across splits, not a lucky train/test partition.

MAPE is close to 1.0 (100%) for every model — this dataset has many low-cost individuals
(median annual cost ≈ $2,083), where even a small absolute error translates into a large
percentage error, so MAE/RMSE are more informative here than MAPE.

The strongest predictors (CatBoost feature importance) are smoking status, total healthcare
utilization, the age × chronic-condition interaction, chronic condition count, days
hospitalized, and the underwriting risk score.

The actual-vs-predicted plot shows the model tracks typical (lower-cost) cases reasonably
well but systematically underpredicts high-cost outliers — expected, since those cases are
driven by rare, high-severity events that are hard to capture from aggregate health/lifestyle
features alone.

## Visualizations

All plots are saved to [`visualization/`](visualization):

| File | Description |
|---|---|
| `target_distribution.png` | Distribution of `annual_medical_cost`, raw and log1p |
| `correlation_heatmap.png` | Correlation heatmap of all numeric features (shows the leakage) |
| `boxplot_before_log.png` / `boxplot_after_log.png` | Numeric feature spread before/after log-transforming `income` |
| `model_comparison.png` | R2 / MAE / RMSE / MAPE across all four models |
| `feature_importance.png` | Top 15 CatBoost feature importances |
| `actual_vs_predicted.png` | Predicted vs. actual cost on the test set |
| `residuals.png` | Residual distribution |

## Project structure

```
Medical_Insurance_Cost_Prediction/
├── datasets/
│   └── medical_insurance.csv
├── visualization/              # generated plots
├── Medical_Insurance_Cost_Prediction.ipynb
└── README.md
```

## Running it

```bash
pip install pandas numpy scikit-learn seaborn matplotlib catboost jupyter
jupyter nbconvert --to notebook --execute --inplace Medical_Insurance_Cost_Prediction.ipynb
```

The notebook reads `datasets/medical_insurance.csv` with a path relative to the project root,
so run it from this directory.
