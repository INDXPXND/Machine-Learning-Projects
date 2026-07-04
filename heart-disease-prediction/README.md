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
4. **Evaluation** — learning curve, confusion matrix, ROC curve, ROC-AUC, classification report.
5. **Top model features** — feature importance of the best model (CatBoost).
6. **Save the model** — saving the trained model to `cbc.joblib`.

## Model results (test / train accuracy)

| Model | Test accuracy | Train accuracy |
|---|---|---|
| SGDClassifier | 0.782 | 0.826 |
| LogisticRegression | 0.805 | 0.869 |
| GradientBoostingClassifier | 0.981 | 1.000 |
| GradientBoostingClassifier + HalvingRandomSearchCV | 0.951 | 0.987 |
| **CatBoostClassifier (best)** | **~0.99** | — |

Final CatBoost metrics on the test set (classification report):

```
              precision    recall  f1-score   support

           0       0.98      1.00      0.99       159
           1       1.00      0.98      0.99       149

    accuracy                           0.99       308
```

ROC-AUC: ~0.9995.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
catboost
joblib
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