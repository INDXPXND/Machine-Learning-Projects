# Heart Disease Prediction

A binary classification project predicting the presence of heart disease in a patient based on clinical features. Implemented in the notebook `Heart_Disease_Dataset.ipynb`.

## Dataset

Uses `heart.csv` with the following features:

| Feature | Description |
|---|---|
| age | patient's age |
| sex | sex |
| cp | chest pain type |
| trestbps | resting blood pressure |
| chol | serum cholesterol level |
| fbs | fasting blood sugar > 120 mg/dl |
| restecg | resting electrocardiographic results |
| thalach | maximum heart rate achieved |
| exang | exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | slope of the peak exercise ST segment |
| ca | number of major vessels colored by fluoroscopy |
| thal | thalassemia test result |
| target | target variable (0 — no disease, 1 — disease present) |

There are no missing values in the data.

## Notebook structure

1. **Getting the data ready** — loading the data, checking for missing values, visualizing the target distribution and the feature correlation matrix.
2. **Data preparation** — train/test split (70/30), feature scaling with `StandardScaler`, class balancing with `SMOTE`.
3. **Building models** — training and comparing several models:
   - `SGDClassifier`
   - `LogisticRegression`
   - `GradientBoostingClassifier`
   - `GradientBoostingClassifier` tuned with `HalvingRandomSearchCV`
   - `CatBoostClassifier` — best model

   For every model, `SMOTE` is applied **only to the training fold**, never to the validation or test data — this avoids synthetic samples leaking into evaluation and inflating the reported scores. For `CatBoostClassifier` in particular, `SMOTE` is applied to the training split before fitting (with the validation split used for early stopping kept as real, unresampled data), and it's wrapped inside an `imblearn.pipeline.Pipeline` for the learning curve and cross-validation steps so that each CV fold is resampled independently.
4. **Evaluation** — learning curve, confusion matrix, ROC curve, ROC-AUC, classification report.
5. **Top model features** — feature importance of the best model (CatBoost).
6. **Save the model** — saving the trained model to `cbc.joblib`.

## Model results (test / train accuracy)

| Model | Test accuracy | Train accuracy |
|---|---|---|
| SGDClassifier | 0.725 | 0.831 |
| LogisticRegression | 0.813 | 0.880 |
| GradientBoostingClassifier | 0.824 | 1.000 |
| GradientBoostingClassifier + HalvingRandomSearchCV | 0.824 | 0.900 |
| **CatBoostClassifier (best)** | **0.846** | 0.948 |

Test set size: 91 samples (30% of the full dataset, ~303 rows).

Final CatBoost metrics on the test set (classification report):

```
              precision    recall  f1-score   support

           0       0.87      0.83      0.85        48
           1       0.82      0.86      0.84        43

    accuracy                           0.85        91
   macro avg       0.85      0.85      0.85        91
weighted avg       0.85      0.85      0.85        91
```

ROC-AUC: ~0.894.

5-fold cross-validation accuracy (SMOTE applied inside each fold): 0.825 ± 0.032.

> These numbers reflect a fix where `SMOTE` was previously applied before the train/validation split for CatBoost, leaking synthetic samples into evaluation and inflating scores to ~0.99. The corrected pipeline keeps all evaluation data (validation, test, and CV folds) free of synthetic samples, so accuracy here is lower but honest.

## Requirements

See [`requirements.txt`](./requirements.txt):

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
catboost>=1.2.0
joblib>=1.3.0
```

Install with:

```
pip install -r requirements.txt
```

## How to run

1. Place `heart.csv` in the working directory (the notebook uses the path `/content/heart.csv`; update it if running locally).
2. Open and run `Heart_Disease_Dataset.ipynb` cell by cell.
3. After training, the best model is saved as `cbc.joblib` and can be loaded like this:

```python
import joblib
model = joblib.load('cbc.joblib')
predictions = model.predict(X_new)
```