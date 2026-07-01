"""
Streamlit live demo for the Telco churn classifier.

Run:    streamlit run app.py
Deploy: Streamlit Community Cloud (Step 8)
"""
from __future__ import annotations

import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'

st.set_page_config(
    page_title='Telco Churn — Live Demo',
    page_icon='📉',
    layout='wide',
)

# ─── Header ─────────────────────────────────────────────────────────────
st.title('Telco Customer Churn — Live Demo')
st.caption(
    'Drag the sliders, watch the prediction. Trained XGBoost (ROC-AUC 0.84). '
    '[Model card](https://github.com/SyrineLarbi/churn-classifier/blob/main/MODEL_CARD.md) · '
    '[Repo](https://github.com/SyrineLarbi/churn-classifier) · '
    '[Portfolio](https://syrine.dev/?persona=data).'
)

# ─── Load the trained pipeline (cached) ─────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(ART / 'model.pkl')

pipe = load_model()

# ─── Inputs ─────────────────────────────────────────────────────────────
st.subheader('Customer profile')
col_a, col_b, col_c = st.columns(3)

with col_a:
    tenure  = st.slider('Tenure (months)', 0, 72, 12, help='How long the customer has been on the platform.')
    monthly = st.slider('Monthly charges (USD)', 18.0, 120.0, 70.0, step=0.5)
    total   = st.number_input('Total charges (USD)', 0.0, 12_000.0, float(tenure * monthly), step=10.0)

with col_b:
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    payment  = st.selectbox('Payment method', [
        'Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)',
    ])
    paperless = st.selectbox('Paperless billing', ['Yes', 'No'])

with col_c:
    internet = st.selectbox('Internet service', ['DSL', 'Fiber optic', 'No'])
    online_sec = st.selectbox('Online security', ['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox('Tech support', ['Yes', 'No', 'No internet service'])

# Optional: collapse the demographics — they barely move the needle, but the
# pipeline expects them so we set defaults.
with st.expander('Demographics (rarely change the prediction)', expanded=False):
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior = st.selectbox('Senior citizen', [0, 1])
    partner = st.selectbox('Partner', ['No', 'Yes'])
    deps = st.selectbox('Dependents', ['No', 'Yes'])
    phone_service = st.selectbox('Phone service', ['Yes', 'No'])
    multi_lines = st.selectbox('Multiple lines', ['No', 'Yes', 'No phone service'])
    online_backup = st.selectbox('Online backup', ['No', 'Yes', 'No internet service'])
    device_protection = st.selectbox('Device protection', ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox('Streaming movies', ['No', 'Yes', 'No internet service'])

# ─── Build a 1-row DataFrame matching the training schema ───────────────
row = pd.DataFrame([{
    'gender':            gender,
    'SeniorCitizen':     senior,
    'Partner':           partner,
    'Dependents':        deps,
    'tenure':            tenure,
    'PhoneService':      phone_service,
    'MultipleLines':     multi_lines,
    'InternetService':   internet,
    'OnlineSecurity':    online_sec,
    'OnlineBackup':      online_backup,
    'DeviceProtection':  device_protection,
    'TechSupport':       tech_support,
    'StreamingTV':       streaming_tv,
    'StreamingMovies':   streaming_movies,
    'Contract':          contract,
    'PaperlessBilling':  paperless,
    'PaymentMethod':     payment,
    'MonthlyCharges':    monthly,
    'TotalCharges':      total,
}])

proba = float(pipe.predict_proba(row)[:, 1][0])
risk_label, risk_color = (
    ('LOW',    '#39ced6') if proba < 0.33
    else ('MEDIUM', '#f5b942') if proba < 0.66
    else ('HIGH',   '#d6399f')
)

# ─── Prediction ─────────────────────────────────────────────────────────
st.subheader('Prediction')
m1, m2 = st.columns(2)
m1.metric('Churn probability', f'{proba:.0%}')
m2.markdown(
    f'<div style="font-size:1.2rem; color:{risk_color}; font-weight:700;">'
    f'Risk bucket: {risk_label}</div>',
    unsafe_allow_html=True,
)

# ─── Local SHAP explanation ─────────────────────────────────────────────
st.subheader('Why this prediction?')

pre = pipe.named_steps['pre']
clf = pipe.named_steps['clf']
row_t = pre.transform(row)
feature_names = pre.get_feature_names_out()

# TreeExplainer uses the tree structure and needs no background dataset, so
# it gives a correct single-row explanation. (shap.Explainer(clf, row_t) would
# use the one row as its own background and collapse every contribution to ~0.)
explainer = shap.TreeExplainer(clf)
sv = explainer(row_t)
sv.feature_names = list(feature_names)

# Render the waterfall to a PNG ourselves and show it via st.image. With the
# correct (non-degenerate) explainer above, bbox_inches='tight' now fits every
# element — long left labels and the full x-axis — without clipping.
plt.close('all')
shap.plots.waterfall(sv[0], max_display=10, show=False)
fig = plt.gcf()
fig.set_size_inches(10, 5)
buf = io.BytesIO()
fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close('all')
st.image(buf.getvalue(), use_container_width=True)

st.caption(
    'Each bar shows how a feature pushed this prediction higher (red, toward '
    'churn) or lower (blue, toward staying). The total lands at the predicted '
    f'log-odds, which corresponds to {proba:.0%} probability.'
)
