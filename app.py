# Predictive Maintenance App — NASA CMAPSS Turbofan Dataset
# streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(page_title="Predictive Maintenance", page_icon="🛠️", layout="wide")
st.title("🛠️ Predictive Maintenance — Industrial Equipment")
st.caption("NASA CMAPSS Turbofan Engine Dataset | ML Jackfruit Project")

# ============================================================
# LOAD MODELS — derive feature count directly from each model
# ============================================================
@st.cache_resource
def load_models():
    m = {}
    for key, path in [
        ('lr',  'models/lr_model.pkl'),
        ('gnb', 'models/gnb_model.pkl'),
        ('rf',  'models/rf_model.pkl'),
    ]:
        try:
            m[key] = joblib.load(path)
        except Exception:
            pass

    try:
        m['feature_cols'] = joblib.load('models/feature_cols.pkl')
    except Exception:
        m['feature_cols'] = None

    return m

models = load_models()

# ── Derive n_features from whichever model is available ──────
def n_features_for(key):
    if key not in models:
        return None
    mdl = models[key]
    try:
        return mdl.named_steps['clf'].n_features_in_
    except Exception:
        try:
            return mdl.n_features_in_
        except Exception:
            return None

# Build per-model feature lists from feature_cols pkl (trimmed to model size)
all_feat = models.get('feature_cols') or [f'sensor{i}' for i in range(1, 29)]

def feature_list_for(key):
    n = n_features_for(key)
    if n is None:
        return all_feat
    return all_feat[:n]

# Use LR feature count for slider display (most likely correct for raw features)
display_features = feature_list_for('lr') or feature_list_for('gnb') or all_feat

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔧 Input & Predict",
    "🔍 Why This Prediction?",
    "❤️ Health Gauge",
    "📊 Model Comparison"
])

# ============================================================
# TAB 1 — INPUT & PREDICT
# ============================================================
with tab1:
    st.subheader("🔧 Sensor Input & Prediction")

    upload = st.file_uploader("Upload a single-row CSV (optional)", type="csv")
    input_vals = {}
    if upload:
        row_df = pd.read_csv(upload)
        input_vals = {col: float(row_df[col].iloc[0])
                      for col in display_features if col in row_df.columns}

    st.markdown("#### Sensor Sliders")
    cols = st.columns(4)
    sensor_input = {}
    for i, feat in enumerate(display_features):
        default = float(input_vals.get(feat, 0.5))
        sensor_input[feat] = cols[i % 4].slider(feat, 0.0, 1.0, default, step=0.01)

    st.markdown("---")
    st.markdown("#### Predictions from All Models")

    result_cols = st.columns(3)
    for (name, key), col in zip(
        [("Logistic Regression", "lr"),
         ("Naive Bayes",         "gnb"),
         ("Random Forest",       "rf")],
        result_cols
    ):
        with col:
            if key in models:
                feats = feature_list_for(key)
                X = np.array([[sensor_input.get(f, 0.5) for f in feats]])
                try:
                    prob  = models[key].predict_proba(X)[0][1]
                    label = "🔴 Near-Failure" if prob >= 0.5 else "🟢 Normal"
                    st.metric(name, label, f"{prob:.1%} failure prob")
                except Exception as e:
                    st.error(f"{name} error: {e}")
            else:
                st.info(f"{name} not loaded")

# ============================================================
# TAB 2 — WHY THIS PREDICTION (RF)
# ============================================================
with tab2:
    st.subheader("🔍 Random Forest: Why This Prediction?")

    if 'rf' in models:
        feats = feature_list_for('rf')
        X_rf  = np.array([[sensor_input.get(f, 0.5) for f in feats]])
        try:
            pred = models['rf'].predict(X_rf)[0]
            prob = models['rf'].predict_proba(X_rf)[0][1]

            st.markdown(f"**Random Forest says:** "
                        f"{'🔴 Near-Failure' if pred else '🟢 Normal'} "
                        f"({prob:.1%} probability)")

            st.image(plt.imread('outputs/feature_importances_dt.png'))


        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning("Random Forest model not loaded.")

# ============================================================
# TAB 3 — HEALTH GAUGE
# ============================================================
with tab3:
    st.subheader("❤️ Engine Health Gauge")

    if 'rf' in models:
        feats = feature_list_for('rf')
        X_rf  = np.array([[sensor_input.get(f, 0.5) for f in feats]])
        try:
            prob_rf = models['rf'].predict_proba(X_rf)[0][1]

            if prob_rf < 0.33:
                status, color, emoji = "NORMAL",   "green",     "🟢"
            elif prob_rf < 0.66:
                status, color, emoji = "WARNING",  "goldenrod", "🟡"
            else:
                status, color, emoji = "CRITICAL", "red",       "🔴"

            st.markdown(f"### {emoji} Engine Status: **:{color}[{status}]**")
            st.markdown(f"**Failure Probability: {prob_rf:.1%}**")
            st.progress(prob_rf)

            fig, ax = plt.subplots(figsize=(5, 2.5), subplot_kw={'projection': 'polar'})
            for thetas, c in [
                (np.linspace(np.pi,     np.pi*2/3, 100), 'green'),
                (np.linspace(np.pi*2/3, np.pi*1/3, 100), 'gold'),
                (np.linspace(np.pi*1/3, 0,         100), 'red'),
            ]:
                ax.fill_between(thetas, 0.6, 1.0, color=c, alpha=0.7)
            needle = np.pi - (prob_rf * np.pi)
            ax.annotate('', xy=(needle, 0.85), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
            ax.set_yticks([]); ax.set_xticks([])
            ax.spines['polar'].set_visible(False)
            ax.set_ylim(0, 1)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Gauge error: {e}")
    else:
        st.warning("Random Forest model not loaded.")

# ============================================================
# TAB 4 — MODEL COMPARISON
# ============================================================
with tab4:
    st.subheader("📊 Final Model Comparison")

    st.dataframe(pd.DataFrame([
        {'Model': 'Logistic Regression',  'F1': 0.661, 'Precision': 0.544, 'Recall': 0.843, 'ROC-AUC': 0.990},
        {'Model': 'Gaussian Naive Bayes', 'F1': 0.613, 'Precision': 0.483, 'Recall': 0.837, 'ROC-AUC': 0.988},
        {'Model': 'Random Forest',        'F1': 0.716, 'Precision': 0.727, 'Recall': 0.705, 'ROC-AUC': 0.987},
    ]), use_container_width=True)
    st.caption("All values from held-out test set (RUL_FD001.txt).")
    st.success("Best model: **Random Forest** — highest F1 = 0.716 and Precision = 0.727")

    st.markdown("""
#### Why Random Forest?

1. **Highest F1 and Precision** — fewest false alarms while catching most real failures.
2. **Ensemble robustness** — bagging over many trees reduces variance vs. a single Decision Tree and avoids the linear assumption of Logistic Regression.
3. **Handles correlated sensors** — random feature subsampling at each split naturally handles the high multicollinearity confirmed by VIF scores.

Logistic Regression has better recall but far lower precision (too many false alarms).
Naive Bayes assumes feature independence which is violated here.
""")