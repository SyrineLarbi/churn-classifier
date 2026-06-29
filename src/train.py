"""
Train XGBoost on the Telco churn dataset.
- Loads via src.features.load_data
- Stratified 80/20 split
- Pipeline(preprocessor, XGBClassifier)
- Saves artifacts/model.pkl, metrics.json, ROC + confusion PNGs
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .features import build_preprocessor, load_data

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
FIG = ROOT / 'figures'
ART.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# Match the portfolio palette so README screenshots feel cohesive
plt.rcParams.update({
    'figure.dpi': 110,
    'axes.facecolor': '#1e1e1e',
    'figure.facecolor': '#222222',
    'axes.edgecolor': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'axes.titlecolor': 'white',
    'savefig.dpi': 140,
    'savefig.bbox': 'tight',
})


def main() -> None:
    X, y = load_data(str(ROOT / 'data/raw/Telco-Customer-Churn.csv'))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42,
    )

    pipe = Pipeline([
        ('pre', build_preprocessor()),
        ('clf', XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=2,
            eval_metric='auc',
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
        )),
    ])
    pipe.fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_te)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    metrics = {
        'roc_auc':              float(roc_auc_score(y_te, proba)),
        'avg_precision':        float(average_precision_score(y_te, proba)),
        'classification_report': classification_report(y_te, pred, output_dict=True),
        'confusion_matrix':     confusion_matrix(y_te, pred).tolist(),
        'class_balance_train':  {'no': int((y_tr == 0).sum()),
                                 'yes': int((y_tr == 1).sum())},
        'class_balance_test':   {'no': int((y_te == 0).sum()),
                                 'yes': int((y_te == 1).sum())},
    }

    print(f'ROC-AUC:        {metrics["roc_auc"]:.3f}')
    print(f'Avg Precision:  {metrics["avg_precision"]:.3f}')
    print(classification_report(y_te, pred, target_names=['No', 'Yes']))

    # ROC curve
    RocCurveDisplay.from_predictions(y_te, proba, color='#4096ee')
    plt.title('Telco churn — ROC curve', weight='bold')
    plt.savefig(FIG / 'roc.png')
    plt.savefig(ART / 'roc.png')
    plt.close()

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_te, pred, display_labels=['No', 'Yes'], cmap='magma',
    )
    plt.title('Telco churn — Confusion matrix (threshold 0.5)', weight='bold')
    plt.savefig(FIG / 'confusion.png')
    plt.savefig(ART / 'confusion.png')
    plt.close()

    # Persist model + metrics
    joblib.dump(pipe, ART / 'model.pkl')
    (ART / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    print(f'\nSaved {ART/"model.pkl"} ({(ART/"model.pkl").stat().st_size/1024:.0f} KB)')


if __name__ == '__main__':
    main()
