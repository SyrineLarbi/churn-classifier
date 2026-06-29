"""
Generate global + local SHAP explanations for the trained model.
Run as: python -m src.explain
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap

from .features import load_data

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)

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
    pipe = joblib.load(ART / 'model.pkl')
    X, y = load_data(str(ROOT / 'data/raw/Telco-Customer-Churn.csv'))

    pre = pipe.named_steps['pre']
    clf = pipe.named_steps['clf']

    # Transform the data once; SHAP works on the transformed matrix
    X_t = pre.transform(X)
    feature_names = pre.get_feature_names_out()

    # Subset for speed (SHAP on 7k × 46 is fine, but 1.5k makes plots cleaner)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_t), size=1500, replace=False)
    X_sample = X_t[idx]

    # Tree explainer is fast for XGBoost
    explainer = shap.Explainer(clf, X_sample, feature_names=feature_names)
    shap_values = explainer(X_sample)

    # ─── Global summary plot ────────────────────────────────────────────
    plt.figure()
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        plot_type='dot',
        max_display=15,
        show=False,
    )
    plt.title('Telco churn — SHAP global feature importance', weight='bold')
    plt.savefig(FIG / 'shap-summary.png')
    plt.savefig(ART / 'shap-summary.png')
    plt.close()
    print(f'Saved {FIG/"shap-summary.png"}')

    # ─── Bar chart (mean |SHAP|) ────────────────────────────────────────
    plt.figure()
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        plot_type='bar',
        max_display=15,
        show=False,
    )
    plt.title('Telco churn — Mean |SHAP| per feature', weight='bold')
    plt.savefig(FIG / 'shap-bar.png')
    plt.savefig(ART / 'shap-bar.png')
    plt.close()
    print(f'Saved {FIG/"shap-bar.png"}')

    # ─── Local example (waterfall) ──────────────────────────────────────
    # Pick the test sample with the highest predicted churn proba — most "interesting"
    proba = clf.predict_proba(X_sample)[:, 1]
    high_risk = int(np.argmax(proba))

    plt.figure()
    shap.plots.waterfall(shap_values[high_risk], max_display=10, show=False)
    plt.title(
        f'Local explanation — customer with p(churn) = {proba[high_risk]:.0%}',
        weight='bold',
    )
    plt.savefig(FIG / 'shap-waterfall.png')
    plt.savefig(ART / 'shap-waterfall.png')
    plt.close()
    print(f'Saved {FIG/"shap-waterfall.png"}')

    # Print top 5 mean-|SHAP| features for the README
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top = sorted(zip(feature_names, mean_abs), key=lambda x: -x[1])[:5]
    print('\nTop-5 features by mean |SHAP|:')
    for name, val in top:
        print(f'  {name:<35} {val:.3f}')


if __name__ == '__main__':
    main()
