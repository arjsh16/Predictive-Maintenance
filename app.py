
# Install Dependencies:
# pip install streamlit joblib matplotlib pandas numpy
# To run this app: streamlit run c:/Users/hp694/OneDrive/Documents/Study/Engineering/sem6/ML/Jackfruit/app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🛠️",
    layout="wide"
)

st.title(" Predictive Maintenance — Industrial Equipment")
st.caption("NASA CMAPSS Turbofan Engine Dataset | ML Jackfruit Project")

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():
    models = {}
    try: models['lr']           = joblib.load('models/lr_model.pkl')
    except: pass
    try: models['gnb']          = joblib.load('models/gnb_model.pkl')
    except: pass
    try: models['dt']           = joblib.load('models/dt_model.pkl')
    except: pass
    try: models['rf']           = joblib.load('models/rf_model.pkl')
    except: pass
    try: models['pca']          = joblib.load('models/pca_transform.pkl')
    except: pass
    try: models['pca_scaler']   = joblib.load('models/pca_scaler.pkl')
    except: pass
    try: models['feature_cols'] = joblib.load('models/feature_cols.pkl')
    except: models['feature_cols'] = [f'sensor{i}' for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]] + ['op1','op2']
    try: models['dt_imp']       = joblib.load('models/dt_importances.pkl')
    except: pass
    return models

models = load_models()
feature_cols = models['feature_cols']

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    " Input & Predict",
    " Why This Prediction?",
    " Health Gauge",
    " Model Comparison"
])

# ============================================================
# TAB 1 — INPUT FORM (Archi)
# ============================================================
with tab1:
    st.subheader(" Sensor Input & Prediction")
    st.markdown("Enter sensor readings for an engine snapshot. Adjust sliders or upload a CSV row.")

    upload = st.file_uploader("Upload a single-row CSV (optional)", type="csv")

    if upload:
        row_df = pd.read_csv(upload)
        input_vals = {col: float(row_df[col].iloc[0]) for col in feature_cols if col in row_df.columns}
    else:
        input_vals = {}

    st.markdown("#### Sensor Sliders")
    cols = st.columns(4)
    sensor_input = {}
    for i, feat in enumerate(feature_cols):
        default = float(input_vals.get(feat, 0.5))
        sensor_input[feat] = cols[i % 4].slider(feat, 0.0, 1.0, default, step=0.01)

    X_input = np.array([[sensor_input[f] for f in feature_cols]])

    st.markdown("---")
    st.markdown("#### Predictions from All Models")

    result_cols = st.columns(4)
    model_map = {
        'Logistic Regression': ('lr',  X_input,  result_cols[0]),
        'Naive Bayes':         ('gnb', X_input,  result_cols[1]),
        'Decision Tree':       ('dt',  X_input,  result_cols[2]),
        'Random Forest':       ('rf',  None,      result_cols[3]),
    }

    X_pca_input = None
    if 'pca_scaler' in models and 'pca' in models:
        X_scaled = models['pca_scaler'].transform(X_input)
        X_pca_input = models['pca'].transform(X_scaled)

    for name, (key, X, col) in model_map.items():
        with col:
            if key == 'rf':
                X_use = X_pca_input
            else:
                X_use = X_input

            if key in models and X_use is not None:
                prob = models[key].predict_proba(X_use)[0][1]
                label = "🔴 Near-Failure" if prob >= 0.5 else "🟢 Normal"
                st.metric(name, label, f"{prob:.1%} failure prob")
            else:
                st.info(f"{name} model not loaded")

# ============================================================
# TAB 2 — WHY THIS PREDICTION (Arjun)
# ============================================================
with tab2:
    st.subheader(" Decision Tree: Why This Prediction?")

    if 'dt' in models and 'dt_imp' in models:
        X_use = X_input
        pred  = models['dt'].predict(X_use)[0]
        prob  = models['dt'].predict_proba(X_use)[0][1]

        st.markdown(f"**Decision Tree says:** {' Near-Failure' if pred else ' Normal'} ({prob:.1%} probability)")

        st.markdown("#### Top Sensors Driving This Prediction")
        importances = models['dt_imp']
        top5 = importances.nlargest(5)

        fig, ax = plt.subplots(figsize=(7, 3))
        colors = ['tomato' if sensor_input.get(feat, 0) > 0.5 else 'steelblue' for feat in top5.index]
        top5.sort_values().plot(kind='barh', ax=ax, color=colors[::-1])
        ax.set_title("Top 5 Feature Importances (Red = high sensor value)")
        ax.set_xlabel("Importance")
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Current Values of Top Sensors")
        rows = []
        for feat in top5.index:
            val = sensor_input.get(feat, 0)
            rows.append({
                'Sensor': feat,
                'Current Value': round(val, 3),
                'Importance': round(top5[feat], 4),
                'Status': ' High' if val > 0.7 else (' Normal' if val < 0.4 else ' Medium')
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        try:
            with open('outputs/decision_rules.txt') as f:
                rules = f.read()
            with st.expander(" View Full IF-THEN Decision Rules"):
                st.code(rules[:2000])
        except:
            st.info("decision_rules.txt not found — run notebook first")
    else:
        st.warning("Decision Tree model not loaded. Run the notebook and place dt_model.pkl in models/")

# ============================================================
# TAB 3 — HEALTH GAUGE (Ansh)
# ============================================================
with tab3:
    st.subheader(" Engine Health Gauge")

    if 'rf' in models and X_pca_input is not None:
        prob_rf = models['rf'].predict_proba(X_pca_input)[0][1]

        if prob_rf < 0.33:
            status, color, emoji = "NORMAL", "green", "🟢"
        elif prob_rf < 0.66:
            status, color, emoji = "WARNING", "goldenrod", "🟡"
        else:
            status, color, emoji = "CRITICAL", "red", "🔴"

        st.markdown(f"### {emoji} Engine Status: **:{color}[{status}]**")
        st.markdown(f"**Failure Probability: {prob_rf:.1%}**")
        st.progress(prob_rf)

        # Gauge
        fig, ax = plt.subplots(figsize=(5, 2.5), subplot_kw={'projection': 'polar'})
        zones = [
            (np.linspace(np.pi, np.pi*2/3, 100), 'green'),
            (np.linspace(np.pi*2/3, np.pi*1/3, 100), 'gold'),
            (np.linspace(np.pi*1/3, 0, 100), 'red')
        ]
        for thetas, c in zones:
            ax.fill_between(thetas, 0.6, 1.0, color=c, alpha=0.7)
        needle = np.pi - (prob_rf * np.pi)
        ax.annotate('', xy=(needle, 0.85), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
        ax.set_yticks([]); ax.set_xticks([])
        ax.spines['polar'].set_visible(False)
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        plt.close()

        st.markdown("#### PCA Health Context")
        st.caption("Where this engine sits in the PCA feature space relative to training data.")

        try:
            pca_scatter = plt.imread('outputs/pca_scatter.png')
            st.image(pca_scatter, caption="PCA 2D Projection — Training Data Health Map")
        except:
            st.info("Run the notebook to generate pca_scatter.png")
    else:
        st.warning("Random Forest or PCA model not loaded. Run the notebook and place rf_model.pkl, pca_transform.pkl, pca_scaler.pkl in models/")

# ============================================================
# TAB 4 — MODEL COMPARISON
# ============================================================
with tab4:
    st.subheader("Final Model Comparison")

    try:
        df = pd.read_csv('outputs/model_comparison.csv')
        st.dataframe(df.style.highlight_max(subset=['F1','Precision','Recall','ROC-AUC'],
                                             color='lightgreen'), use_container_width=True)
        best = df.iloc[0]['Model']
        st.success(f"Best model: **{best}** with F1 = {df.iloc[0]['F1']}")
    except:
        st.info("model_comparison.csv not found — run the full notebook first")

    st.markdown("#### Metric Definitions")
    st.markdown("""
| Metric | What it means |
|--------|--------------|
| F1 | Balance of precision and recall — main metric |
| Precision | Of engines flagged as failing, how many actually were |
| Recall | Of engines that actually failed, how many we caught |
| ROC-AUC | Overall discrimination ability across thresholds |
""")
