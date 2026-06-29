"""
Preprocessing pipeline for the Telco churn dataset.

Exposes:
- NUMERIC, CATEGORICAL: feature lists used everywhere.
- build_preprocessor(): returns a ColumnTransformer.
- load_data(path): reads + light-cleans the CSV, returns (X, y).
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ['tenure', 'MonthlyCharges', 'TotalCharges']

CATEGORICAL = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
]

DROP_COLS = ['customerID']

TARGET = 'Churn'


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale',  StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe',    OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ('num', numeric_pipe, NUMERIC),
            ('cat', categorical_pipe, CATEGORICAL),
        ],
        remainder='drop',
        verbose_feature_names_out=True,
    )


def load_data(path: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Read the CSV, fix dtypes, return (X, y) with `Churn` removed from X."""
    path = path or 'data/raw/Telco-Customer-Churn.csv'
    df = pd.read_csv(path)
    df = df.drop(columns=DROP_COLS, errors='ignore')

    # The classic dirty-data fix
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Binary target
    df[TARGET] = (df[TARGET] == 'Yes').astype(int)

    y = df.pop(TARGET)
    return df, y
