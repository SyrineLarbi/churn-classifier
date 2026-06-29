# Model Card — Telco Customer Churn Classifier

> XGBoost classifier predicting the probability that a Telco customer will churn during their next billing cycle.

## Intended use

- **Primary:** educational / portfolio demonstration of an end-to-end tabular ML workflow.
- **Out of scope:** real billing or HR decisions, risk-of-loss decisions affecting customers, automated retention spend allocation. The class imbalance and modest recall make this unsafe for unsupervised production use.

## Training data

- **Source:** [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) (mirror of an earlier IBM Cognos sample).
- **Size:** 7 043 customers × 21 columns. License: free for educational use.
- **Class balance:** 73.5% No / 26.5% Yes. No re-balancing was applied; the classifier is asymmetric in its costs accordingly.
- **Demographic coverage:** synthetic / unidentified. Not representative of any real subscriber base.
- **Train / test split:** stratified 80 / 20, `random_state = 42`. No cross-validation (single holdout). Train: 4 139 No / 1 495 Yes. Test: 1 035 No / 374 Yes.

## Features

Numeric (3): `tenure`, `MonthlyCharges`, `TotalCharges`.
Categorical (16): `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`.

Dropped: `customerID`.

## Preprocessing

`sklearn.compose.ColumnTransformer` with:

- numeric: median imputation → `StandardScaler`.
- categorical: most-frequent imputation → `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.

## Model

`xgboost.XGBClassifier`:

```
n_estimators        = 500
max_depth           = 4
learning_rate       = 0.05
subsample           = 0.9
colsample_bytree    = 0.9
min_child_weight    = 2
eval_metric         = 'auc'
random_state        = 42
tree_method         = 'hist'
```

No hyper-parameter search; values picked for "sane defaults that don't overfit a 5k-row dataset."

## Evaluation (20% stratified holdout)

| Metric                      | Value |
|-----------------------------|-------|
| ROC-AUC                     | **0.836** |
| Average Precision (PR-AUC)  | 0.647 |
| Accuracy @ threshold 0.5    | 0.798 |
| Precision (Yes class)       | 0.648 |
| Recall (Yes class)          | 0.521 |
| F1 (Yes class)              | 0.578 |

Confusion matrix (threshold 0.5):

|             | Pred: No | Pred: Yes |
|-------------|----------|-----------|
| **Actual No**  | 929 | 106 |
| **Actual Yes** | 179 | 195 |

(Numbers will vary by ± a few rows depending on environment; see `artifacts/metrics.json` for the exact run.)

## Top features (mean |SHAP|)

1. `Contract = Month-to-month`
2. `tenure`
3. `MonthlyCharges`
4. `TotalCharges`
5. `OnlineSecurity = No`

These match well-known Telco churn drivers, suggesting the model isn't overfitting to spurious patterns.

## Caveats & honest limitations

- **Class imbalance not addressed.** `scale_pos_weight` / class-weight / SMOTE / threshold calibration would lift recall on the churn class. Not done because the goal is a clean baseline. At threshold 0.5 the model catches only 52% of actual churners (179 of 374 missed).
- **No hyperparameter search.** Reproducible, but a `RandomizedSearchCV` would likely add 1–2 ROC-AUC points.
- **No calibration.** Predicted probabilities are not necessarily well-calibrated; for production, run a `CalibratedClassifierCV` post-hoc.
- **Synthetic data.** Real-world drift (price changes, COVID, new plan structures) will degrade performance. Re-train at least quarterly.
- **No fairness audit.** With a real subscriber base + protected attributes (age, income, geography), I'd run group-wise metrics.
- **Single train/test split.** A K-fold cross-validation report would be more robust.

## Reproducibility

```
git clone https://github.com/SyrineLarbi/churn-classifier
cd churn-classifier
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
curl -L -o data/raw/Telco-Customer-Churn.csv \
  https://github.com/IBM/telco-customer-churn-on-icp4d/raw/master/data/Telco-Customer-Churn.csv
python -m src.train     # → artifacts/model.pkl, metrics.json, ROC + confusion plots
python -m src.explain   # → SHAP summary + waterfall plots
```

Model artifacts are committed to the repo so the Streamlit app can load `model.pkl` without retraining.

## Maintainer

[Syrine Larbi](https://github.com/SyrineLarbi) · 2026 · [syrine.dev](https://syrine.dev/?persona=data)
