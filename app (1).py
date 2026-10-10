import json
import pickle
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Churn Risk Advisor', page_icon='📉',
                   layout='wide')

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
RAW_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',
            'TotalCharges']
SERVICES = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies']
PAYMENTS = ['Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)']
SENIOR_AGE = 65          # age at which a customer is flagged SeniorCitizen

RED, AMBER, GREEN, TEAL, SLATE = '#e5533d', '#f0a23b', '#2f9e5b', '#1b7f8e', '#5b6b7a'
BAND_COLOR = {'HIGH': RED, 'WATCH': AMBER, 'LOW': GREEN}

BASE = Path(__file__).parent          # folder that contains app.py
MODEL_FILES = [BASE / 'churn_model.joblib', BASE / 'best_churn_model.pkl']

# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
CSS = """
<style>
.block-container {padding-top: 1.3rem;}
.hero {background: linear-gradient(120deg, #0f4c5c 0%, #1b7f8e 60%, #2aa198 100%);
       color: #fff; padding: 1.2rem 1.6rem; border-radius: 16px;
       margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 1.85rem; color: #fff; padding: 0;}
.hero p {margin: .3rem 0 0; opacity: .92; font-size: .95rem;}
.kpi {border-radius: 14px; padding: .9rem 1.2rem;
      border: 1px solid rgba(128,128,128,.25); border-left: 7px solid var(--c);
      background: rgba(128,128,128,.07); height: 100%;}
.kpi .label {font-size: .78rem; text-transform: uppercase;
             letter-spacing: .07em; opacity: .7;}
.kpi .value {font-size: 2.1rem; font-weight: 700; line-height: 1.25;
             color: var(--c);}
.kpi .sub {font-size: .85rem; opacity: .75;}
.meter-wrap {position: relative; margin: 1.1rem 0 .3rem;}
.meter {display: flex; height: 16px; border-radius: 8px; overflow: hidden;}
.pin {position: absolute; top: -6px; width: 6px; height: 28px;
      margin-left: -3px; border-radius: 3px; background: #111;
      border: 2px solid #fff; box-shadow: 0 0 4px rgba(0,0,0,.4);}
.legend {font-size: .8rem; opacity: .75; margin-top: .45rem;}
.chips {margin: .6rem 0 .2rem;}
.chip {display: inline-block; padding: .2rem .7rem; margin: 0 .35rem .35rem 0;
       border-radius: 999px; font-size: .8rem;
       background: rgba(27,127,142,.14); border: 1px solid rgba(27,127,142,.35);}
.panel {border-radius: 14px; padding: .9rem 1.1rem;
        border: 1px solid rgba(128,128,128,.25);
        background: rgba(128,128,128,.05);}
section[data-testid="stSidebar"] h3 {margin-top: .6rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def kpi(label, value, sub, color):
    return (f'<div class="kpi" style="--c:{color}">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def meter_html(p, t):
    a, b = t / 2 * 100, t * 100
    return (f'<div class="meter-wrap"><div class="meter">'
            f'<div style="width:{a}%;background:{GREEN}"></div>'
            f'<div style="width:{b - a}%;background:{AMBER}"></div>'
            f'<div style="width:{100 - b}%;background:{RED}"></div></div>'
            f'<div class="pin" style="left:{p * 100}%"></div></div>'
            f'<div class="legend">LOW below {a:.0f}%  |  '
            f'WATCH {a:.0f}% to {b:.0f}%  |  HIGH from {b:.0f}%</div>')


# ------------------------------------------------------------------
# Model loading (supports churn_model.joblib or best_churn_model.pkl)
# ------------------------------------------------------------------
@st.cache_resource            # load once per server, not on every click
def load_artifacts(model_path):
    meta = {}
    meta_path = BASE / 'model_meta.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    if model_path.suffix == '.joblib':
        model = joblib.load(model_path)
        columns = meta.get('feature_columns') or list(model.feature_names_in_)
    else:
        with open(model_path, 'rb') as f:
            bundle = pickle.load(f)
        model = bundle['model']
        columns = list(bundle['columns'])

    final = model[-1] if hasattr(model, 'steps') else model
    defaults = {'model_name': type(final).__name__, 'threshold': 0.30,
                'cv_auc': None, 'cv_auc_std': None, 'test_auc': None,
                'training_data': 'IBM Telco Customer Churn, 7,043 customers',
                'version': '1.0'}
    return model, {**defaults, **meta, 'feature_columns': columns}


def prepare_input(raw, columns):
    X = pd.get_dummies(raw[RAW_COLS])            # no drop_first at serving
    return X.reindex(columns=columns, fill_value=0).astype(float)


available = [p for p in MODEL_FILES if p.exists()]
if not available:
    st.error('Model file not found. Put churn_model.joblib (or '
             'best_churn_model.pkl) next to app.py and commit it to GitHub.')
    st.stop()
model, meta = load_artifacts(available[0])
COLS = meta['feature_columns']


def score(raw):
    return model.predict_proba(prepare_input(raw, COLS))[:, 1]


def band_of(p, t):
    return 'HIGH' if p >= t else 'WATCH' if p >= t / 2 else 'LOW'


# ------------------------------------------------------------------
# Sidebar: one customer
# ------------------------------------------------------------------
with st.sidebar:
    st.title('Customer profile')

    st.subheader('👤 Who is the customer?')
    gender = st.radio('Gender', ['Female', 'Male'], horizontal=True)
    age = st.slider('Age (years)', 18, 90, 35)
    senior = age >= SENIOR_AGE
    st.caption(f'Flagged as senior citizen: **{"Yes" if senior else "No"}** '
               f'(age {SENIOR_AGE}+). The model stores only this yes/no '
               f'flag, not exact age.')
    partner = st.toggle('Has partner')
    dependents = st.toggle('Has dependents')

    st.subheader('📅 Account')
    tenure = st.slider('Tenure (months)', 0, 72, 4)
    contract = st.selectbox('Contract',
                            ['Month-to-month', 'One year', 'Two year'])

    st.subheader('📶 Services')
    internet = st.selectbox('Internet service', ['Fiber optic', 'DSL', 'No'])
    services = []
    if internet != 'No':
        services = st.multiselect('Add-on services', SERVICES,
                                  default=['StreamingTV'])
    phone = st.radio('Phone service', ['Yes', 'No'], horizontal=True)
    multi = 'No phone service'
    if phone == 'Yes':
        multi = st.radio('Multiple lines', ['No', 'Yes'], horizontal=True)

    st.subheader('💳 Billing')
    monthly = st.number_input('Monthly charges (USD)', 18.0, 120.0, 85.0,
                              step=0.5)
    payment = st.selectbox('Payment method', PAYMENTS)
    paperless = st.toggle('Paperless billing', value=True)

    st.divider()
    st.subheader('⚙️ Decision rule')
    threshold = st.slider('Contact threshold', 0.05, 0.95,
                          float(meta['threshold']), 0.05)

row = {'gender': gender, 'SeniorCitizen': int(senior),
       'Partner': 'Yes' if partner else 'No',
       'Dependents': 'Yes' if dependents else 'No', 'tenure': tenure,
       'PhoneService': phone, 'MultipleLines': multi,
       'InternetService': internet, 'Contract': contract,
       'PaperlessBilling': 'Yes' if paperless else 'No',
       'PaymentMethod': payment, 'MonthlyCharges': monthly,
       'TotalCharges': tenure * monthly}           # derived, not typed
for s in SERVICES:
    row[s] = ('No internet service' if internet == 'No'
              else 'Yes' if s in services else 'No')
customer = pd.DataFrame([row])

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
if meta['cv_auc'] is not None and meta['cv_auc_std'] is not None:
    auc_text = f"CV AUC {meta['cv_auc']:.3f} ± {meta['cv_auc_std']:.3f}"
else:
    auc_text = 'CV AUC not recorded'
st.markdown(
    f'<div class="hero"><h1>📉 Customer Churn Risk Advisor</h1>'
    f'<p>{meta["model_name"]} &nbsp;|&nbsp; {auc_text} &nbsp;|&nbsp; '
    f'Decision support only, not automation</p></div>',
    unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(['🧑 One customer', '📂 Batch scoring', 'ℹ️ About'])

# ------------------------------------------------------------------
# Tab 1: one customer
# ------------------------------------------------------------------
with tab1:
    p = float(score(customer)[0])
    band = band_of(p, threshold)
    color = BAND_COLOR[band]

    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi('Churn probability', f'{p:.0%}',
                    'chance this customer leaves', color),
                unsafe_allow_html=True)
    k2.markdown(kpi('Risk band', band,
                    f'contact threshold {threshold:.0%}', color),
                unsafe_allow_html=True)
    k3.markdown(kpi('Action', 'Contact now' if band == 'HIGH' else
                    'Monitor' if band == 'WATCH' else 'No action',
                    'recommended next step', color),
                unsafe_allow_html=True)

    st.markdown(meter_html(p, threshold), unsafe_allow_html=True)
    chips = [gender, f'{age} yrs' + (' (senior)' if senior else ''),
             f'{tenure} months', contract, f'Internet: {internet}',
             payment, f'${monthly:,.2f}/month']
    st.markdown('<div class="chips">' +
                ''.join(f'<span class="chip">{c}</span>' for c in chips) +
                '</div>', unsafe_allow_html=True)

    st.write('')
    left, right = st.columns([3, 2])

    with left:
        st.subheader('What would change the risk?')
        changes = [('Contract', c) for c in ['One year', 'Two year']
                   if c != contract]
        if payment == 'Electronic check':
            changes.append(('PaymentMethod', 'Credit card (automatic)'))
        if internet != 'No' and 'TechSupport' not in services:
            changes.append(('TechSupport', 'Yes'))
        rows = []
        for feature, value in changes:
            alt_row = customer.copy()
            alt_row[feature] = value
            q = float(score(alt_row)[0])
            rows.append({'Change': f'{feature}: {value}',
                         'New probability': round(q, 3),
                         'Difference': round(q - p, 3)})
        if rows:
            plot_df = pd.DataFrame(
                [{'Scenario': 'Current', 'Probability': p}] +
                [{'Scenario': r['Change'], 'Probability': r['New probability']}
                 for r in rows])
            bars = alt.Chart(plot_df).mark_bar(cornerRadiusEnd=5).encode(
                y=alt.Y('Scenario:N', sort=None, title=None),
                x=alt.X('Probability:Q', scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format='%'), title='Churn probability'),
                color=alt.condition(alt.datum.Scenario == 'Current',
                                    alt.value(SLATE), alt.value(TEAL)),
                tooltip=['Scenario', alt.Tooltip('Probability:Q',
                                                 format='.1%')])
            labels = bars.mark_text(align='left', dx=4).encode(
                text=alt.Text('Probability:Q', format='.0%'),
                color=alt.value('gray'))
            st.altair_chart((bars + labels).properties(
                height=46 * len(plot_df) + 30), use_container_width=True)
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)
        st.caption('Associations learned from data, not guaranteed effects.')

    with right:
        st.subheader('Why this score?')
        final = model[-1] if hasattr(model, 'steps') else model
        if hasattr(model, 'steps') and hasattr(final, 'coef_'):
            z = model[:-1].transform(prepare_input(customer, COLS))[0]
            contrib = pd.Series(final.coef_[0] * z, index=COLS)
            top = contrib.reindex(
                contrib.abs().sort_values(ascending=False).index).head(8)
            top_df = pd.DataFrame({'Feature': top.index,
                                   'Contribution': top.values})
            st.altair_chart(
                alt.Chart(top_df).mark_bar(cornerRadiusEnd=5).encode(
                    y=alt.Y('Feature:N', sort=None, title=None),
                    x=alt.X('Contribution:Q', title='Contribution'),
                    color=alt.condition(alt.datum.Contribution > 0,
                                        alt.value(RED), alt.value(GREEN)),
                    tooltip=['Feature', alt.Tooltip('Contribution:Q',
                                                    format='.3f')]
                ).properties(height=300), use_container_width=True)
            st.caption('Red pushes toward churn, green pushes away '
                       '(Logistic Regression contributions).')
        else:
            st.markdown(
                '<div class="panel">A per-customer explanation is available '
                'for Logistic Regression only. For forests and XGBoost it '
                'needs SHAP, which arrives in Week 15.</div>',
                unsafe_allow_html=True)

# ------------------------------------------------------------------
# Tab 2: batch scoring
# ------------------------------------------------------------------
with tab2:
    st.subheader('Score a whole customer file')
    st.write('Upload a CSV with the original Telco columns.')
    file = st.file_uploader('Customer file', type='csv')
    if file is not None:
        data = pd.read_csv(file)
        missing = [c for c in RAW_COLS if c not in data.columns]
        if missing:
            st.error(f'Missing columns: {missing}')
        else:
            data['TotalCharges'] = pd.to_numeric(
                data['TotalCharges'], errors='coerce').fillna(0)
            data['p_churn'] = score(data).round(3)
            data['risk_band'] = np.select(
                [data['p_churn'] >= threshold, data['p_churn'] >= threshold / 2],
                ['HIGH', 'WATCH'], default='LOW')
            data['contact'] = np.where(data['p_churn'] >= threshold, 'Yes', '')

            n_high = int((data['contact'] == 'Yes').sum())
            b1, b2, b3 = st.columns(3)
            b1.markdown(kpi('Customers scored', f'{len(data):,}',
                            'rows in the file', TEAL), unsafe_allow_html=True)
            b2.markdown(kpi('Contact now', f'{n_high:,}',
                            f'{n_high / len(data):.0%} are above the threshold',
                            RED), unsafe_allow_html=True)
            b3.markdown(kpi('Average risk', f"{data['p_churn'].mean():.0%}",
                            'mean churn probability', AMBER),
                        unsafe_allow_html=True)

            st.write('')
            c_left, c_right = st.columns([2, 3])
            with c_left:
                counts = (data['risk_band'].value_counts()
                          .reindex(['HIGH', 'WATCH', 'LOW'], fill_value=0)
                          .reset_index())
                counts.columns = ['Band', 'Customers']
                st.altair_chart(
                    alt.Chart(counts).mark_bar(cornerRadiusEnd=5).encode(
                        x=alt.X('Band:N', sort=['HIGH', 'WATCH', 'LOW'],
                                title=None),
                        y=alt.Y('Customers:Q'),
                        color=alt.Color('Band:N', legend=None,
                                        scale=alt.Scale(
                                            domain=['HIGH', 'WATCH', 'LOW'],
                                            range=[RED, AMBER, GREEN])),
                        tooltip=['Band', 'Customers']
                    ).properties(height=260), use_container_width=True)
            with c_right:
                first = ['p_churn', 'risk_band', 'contact']
                ordered = first + [c for c in data.columns if c not in first]
                st.dataframe(
                    data.sort_values('p_churn', ascending=False)
                        .head(50)[ordered],
                    use_container_width=True,
                    column_config={'p_churn': st.column_config.ProgressColumn(
                        'p_churn', min_value=0.0, max_value=1.0,
                        format='%.2f')})
            st.download_button('⬇️ Download scored CSV',
                               data.to_csv(index=False),
                               'scored_customers.csv', 'text/csv')

# ------------------------------------------------------------------
# Tab 3: about
# ------------------------------------------------------------------
with tab3:
    st.subheader('About this model')
    a1, a2, a3 = st.columns(3)
    a1.markdown(kpi('Model', meta['model_name'], f"version {meta['version']}",
                    TEAL), unsafe_allow_html=True)
    a2.markdown(kpi('Features', f'{len(COLS)}', 'encoded columns', SLATE),
                unsafe_allow_html=True)
    a3.markdown(kpi('Threshold', f'{meta["threshold"]:.0%}',
                    'default contact threshold', AMBER),
                unsafe_allow_html=True)
    st.write('')
    st.json({k: v for k, v in meta.items() if k != 'feature_columns'})
    st.markdown(
        '**Limitations:** trained on one US telecom dataset; not validated '
        'for other markets. Gender and age (senior flag) are model inputs, so '
        'check recall across gender and seniors (full audit in Week 15). '
        'Not for credit, pricing, or any decision that denies a service. '
        'See the model card.')
