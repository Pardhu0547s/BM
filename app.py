"""
OsteoScan — BMD Prediction Dashboard
Premium Clinical SaaS Edition
Inspired by ICarePro / Vento / Lumos aesthetics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    ARTIFACTS_DIR, BINARY_FEATURES, BMD_THRESHOLDS, DATA_PATH,
    METRICS_PATH, NUMERIC_FEATURES, PIPELINE_PATH,
    SHAP_VALUES_PATH, TEST_RESULTS_PATH, VALIDATION_RANGES,
)
from src.utils import classify_bmd, bmd_risk_description, validate_input

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OsteoScan",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────────────
DARK = False
BG          = "#F2F5F8"    # Silver-ish white background
SIDEBAR_BG  = "#FFFFFF"    # White sidebar
CARD        = "#FFFFFF"    # White card
CARD2       = "#EAEFF4"    # Silver card
BORDER      = "rgba(0,0,0,0.15)"
BORDER2     = "rgba(0,0,0,0.25)"
T1          = "#000000"    # Black text
T2          = "#000000"    # Black text
T3          = "#000000"    # Black text
ACCENT      = "#0056D2"    # Blue accent
ACCENT2     = "#0F9D58"    # Green accent
GREEN       = "#0F9D58"    # Green
AMBER       = "#F4B400"    # Kept for medical semantic warning
RED         = "#DB4437"    # Kept for medical semantic danger
CHIP_GREEN  = "rgba(15,157,88,0.15)"
CHIP_AMBER  = "rgba(244,180,0,0.15)"
CHIP_RED    = "rgba(219,68,55,0.15)"
CHIP_BLUE   = "rgba(0,86,210,0.15)"
SHADOW      = "0 2px 10px rgba(0,0,0,0.05)"
SHADOW_SM   = "0 1px 4px rgba(0,0,0,0.05)"
PLOT_TPL    = "plotly_white"
PLOT_PAPER  = "rgba(0,0,0,0)"
GRID_C      = "rgba(0,0,0,0.1)"
MPL_STYLE   = "default"
MAT_BG      = "#FFFFFF"
MAT_T       = "#000000"
MAT_T2      = "#000000"
MAT_GRID    = "#D3D3D3"    # Silver grid

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body {
    font-family: sans-serif;
    color: #000000;
}

/* Simple cards */
.card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid silver;
    margin-bottom: 1rem;
    color: #000000;
}

/* KPI */
.kpi {
    font-size: 1.4rem;
    font-weight: bold;
    color: #000000;
}

.kpi-label {
    font-size: 0.8rem;
    color: #000000;
}

/* Header */
.header {
    margin-bottom: 1rem;
    color: #000000;
}

.title {
    font-size: 1.4rem;
    font-weight: bold;
    color: #000000;
}

.sub {
    font-size: 0.9rem;
    color: #000000;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_PATH)

@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)

@st.cache_data
def load_test_results():
    return pd.read_csv(TEST_RESULTS_PATH)


def label_color(label):
    return {"Normal": GREEN, "Osteopenia": AMBER, "Osteoporosis": RED}.get(label, T2)

def label_chip_cls(label):
    return {"Normal": "green", "Osteopenia": "amber", "Osteoporosis": "red"}.get(label, "blue")

def label_icon(label):
    return {"Normal": "✓", "Osteopenia": "⚠", "Osteoporosis": "✕"}.get(label, "·")

def label_chip_bg(label):
    return {"Normal": CHIP_GREEN, "Osteopenia": CHIP_AMBER, "Osteoporosis": CHIP_RED}.get(label, CHIP_BLUE)


def style_fig(fig, title=None):
    fig.update_layout(
        plot_bgcolor=PLOT_PAPER, paper_bgcolor=PLOT_PAPER,
        font=dict(color=T2, size=11, family="JetBrains Mono, monospace"),
        margin=dict(l=32, r=16, t=50 if title else 24, b=28),
        xaxis=dict(
            gridcolor=GRID_C, showline=False, zeroline=False,
            tickfont=dict(color=T3, size=10, family="JetBrains Mono, monospace"),
            title_font=dict(size=11, color=T2),
        ),
        yaxis=dict(
            gridcolor=GRID_C, showline=False, zeroline=False,
            tickfont=dict(color=T3, size=10, family="JetBrains Mono, monospace"),
            title_font=dict(size=11, color=T2),
        ),
        title=dict(
            text=title or "",
            font=dict(size=13, color=T1, family="Plus Jakarta Sans, sans-serif"),
            x=0, xanchor="left",
        ),
        legend=dict(font=dict(color=T2, size=10), bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    return fig


def kpi(icon, icon_bg, label, value, unit="", delta_text="", delta_cls="up"):
    st.markdown(f"""
    <div class="card">
        <div class="kpi-label">{label} {icon}</div>
        <div class="kpi">{value} {unit}</div>
        <div style="font-size:0.8rem; color:#000000; margin-top:0.5rem;">{delta_text}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown(f"""
    <div class="brand">
        <div class="brand-icon">🦴</div>
        <div>
            <div class="brand-name">OsteoScan</div>
            <div class="brand-tag">BMD · Clinical AI</div>
        </div>
    </div>
    <div class="nav-section">Main</div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Navigation",
        ["🔬  Predict", "📊  Evaluation"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown(f"""
    <div class="model-card">
        <div class="model-card-title">Model Info</div>
        <div class="trow"><span class="trow-label">Algorithm</span><span class="trow-value">GradientBoost</span></div>
        <div class="trow"><span class="trow-label">Dataset</span><span class="trow-value">593 pts (Female)</span></div>
        <div class="trow"><span class="trow-label">Features</span><span class="trow-value">31 markers</span></div>
        <div class="trow"><span class="trow-label">Validation</span><span class="trow-value">5-Fold CV</span></div>
        <div class="trow"><span class="trow-label">Target</span><span class="trow-value"> L1–4 BMD</span></div>
    </div>
    """, unsafe_allow_html=True)

    return page


# ─────────────────────────────────────────────────────────────────────────────
# Page: Predict
# ─────────────────────────────────────────────────────────────────────────────
def page_predict():
    pipeline     = load_pipeline()
    test_results = load_test_results()
    input_data   = {}

    st.markdown("""
    <div class="header">
        <div class="title">OsteoScan</div>
        <div class="sub">Female-Specific Clinical BMD Prediction</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(":blue[01 · Demographics & Anthropometrics]", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1: input_data["Age"]    = st.number_input("Age (yrs)", 18, 120, 55)
        with c2: input_data["Height"] = st.number_input("Height (cm)", 100.0, 220.0, 165.0, step=0.5)
        with c3: input_data["Weight"] = st.number_input("Weight (kg)", 25.0, 200.0, 65.0, step=0.5)
        with c4: input_data["BMI"]    = st.number_input("BMI (kg/m²)", 10.0, 60.0, 23.5, step=0.1)

    with st.expander(":green[02 · Biochemical Markers]", expanded=True):
        lab_features = [
            ("ALT","U/L",1.0,500.0,25.0), ("AST","U/L",1.0,500.0,22.0),
            ("BUN","mmol/L",0.5,50.0,5.5), ("CREA","µmol/L",10.0,1500.0,70.0),
            ("URIC","µmol/L",50.0,900.0,320.0), ("FBG","mmol/L",2.0,30.0,5.2),
            ("HDL-C","mmol/L",0.1,5.0,1.3), ("LDL-C","mmol/L",0.1,10.0,2.8),
            ("Ca","mmol/L",1.0,4.0,2.2), ("P","mmol/L",0.3,3.0,1.1),
            ("Mg","mmol/L",0.3,2.0,0.85),
        ]
        cols = st.columns(5)
        for i, (feat, unit, lo, hi, default) in enumerate(lab_features):
            with cols[i % 5]:
                input_data[feat] = st.number_input(f"{feat} ({unit})", lo, hi, default, step=0.01)

    with st.expander(":orange[03 · Comorbidities]", expanded=False):
        mh_features = ["HTN","COPD","DM","Hyperlipidaemia","Hyperuricemia","AS","VT","VD","CAD","CKD"]
        mh_cols = st.columns(5)
        for i, feat in enumerate(mh_features):
            with mh_cols[i % 5]:
                input_data[feat] = float(st.selectbox(feat, [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"))

    with st.expander(":violet[04 · Medications & Lifestyle]", expanded=False):
        med_features = ["Calsium","Calcitriol","Bisphosphonate","Calcitonin","Smoking","Drinking"]
        med_cols = st.columns(3)
        for i, feat in enumerate(med_features):
            with med_cols[i % 3]:
                input_data[feat] = float(st.selectbox(feat, [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key=f"med_{feat}"))

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Run Prediction", width="stretch"):
        validation = validate_input(input_data)
        if validation["warnings"]:
            for w in validation["warnings"]:
                st.warning(f"⚠  {w}")

        input_df   = pd.DataFrame([input_data])
        prediction = pipeline.predict(input_df)[0]
        label, _   = classify_bmd(prediction)
        bmd_pct = ((prediction - 0.3) / (1.6 - 0.3)) * 100

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="card">
                <div class="kpi-label">BMD</div>
                <div class="kpi">{prediction:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="card">
                <div class="kpi-label">Status</div>
                <div class="kpi">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="card">
                <div class="kpi-label">Score</div>
                <div class="kpi">{bmd_pct:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

        left, right = st.columns([2,1])

        with left:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={"text": "Lumbar L1-4"},
                gauge={
                    "axis": {"range": [0.3, 1.6]},
                    "bar": {"color": "black", "thickness": 0.2},
                    "steps": [
                        {"range": [0.30, 0.80], "color": "rgba(255, 99, 71, 0.4)"},
                        {"range": [0.80, 1.00], "color": "rgba(255, 165, 0, 0.4)"},
                        {"range": [1.00, 1.60], "color": "rgba(60, 179, 113, 0.4)"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 2}, "thickness": 0.75, "value": prediction},
                },
            ))
            fig_g.update_layout(height=260, margin=dict(l=20, r=20, t=70, b=4))
            st.plotly_chart(fig_g, width="stretch")

        with right:
            st.markdown(f"""
            <div class="card">
                <b>Summary</b><br><br>
                Result: {label}<br>
                BMD: {prediction:.3f}<br><br>
                Advice:<br>
                Maintain healthy lifestyle
            </div>
            """, unsafe_allow_html=True)

        # SHAP
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="sec-label">Feature Attribution · SHAP Waterfall</div>', unsafe_allow_html=True)
        try:
            import shap, matplotlib.pyplot as plt, matplotlib
            matplotlib.use("Agg")
            gb_model     = pipeline.named_steps["model"]
            preprocessor = pipeline.named_steps["preprocessor"]
            feat_names   = []
            for name, _, cols in preprocessor.transformers_:
                feat_names.extend(cols)

            X_t       = preprocessor.transform(input_df)
            explainer = shap.TreeExplainer(gb_model)
            shap_vals = explainer.shap_values(X_t)
            base_val  = explainer.expected_value
            base_val  = float(base_val[0]) if hasattr(base_val, "__len__") and len(base_val) > 0 else float(base_val)
            sv = np.array(shap_vals[0]).flatten()
            xv = np.array(X_t[0]).flatten()
            exp = shap.Explanation(values=sv, base_values=base_val, feature_names=feat_names, data=xv)

            plt.style.use(MPL_STYLE)
            fig_s, ax = plt.subplots(figsize=(10, 5))
            fig_s.patch.set_facecolor(MAT_BG); fig_s.patch.set_alpha(0.0)
            ax.set_facecolor(MAT_BG)
            shap.plots.waterfall(exp, max_display=10, show=False)
            ax = plt.gca(); fig_s = plt.gcf()
            for t in fig_s.findobj(matplotlib.text.Text):
                t.set_color(MAT_T); t.set_fontsize(9.5); t.set_fontfamily("monospace")
            ax.tick_params(colors=MAT_T2)
            for spine in ax.spines.values(): spine.set_color(MAT_GRID)
            plt.title("SHAP Feature Contributions", color=MAT_T, pad=14, size=12, weight="bold", fontfamily="monospace")
            plt.tight_layout()
            st.pyplot(fig_s)
            plt.close("all")
        except Exception as e:
            st.warning(f"SHAP explainer unavailable: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def page_evaluation():
    metrics      = load_metrics()
    test_results = load_test_results()

    st.markdown(f"""
    <div class="pg-header">
        <div class="pg-title">Model Evaluation</div>
        <div class="pg-sub">Performance metrics across held-out test data and cross-validation folds</div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    cv = metrics.get("cv_scores", [])
    with k1: kpi("🎯", CHIP_GREEN, "R² Score",  f"{metrics['r2_score']:.3f}", "",       "98% variance explained", "up")
    with k2: kpi("📉", CHIP_BLUE,  "MAE",       f"{metrics['mae']:.4f}",      "g/cm²", "Mean absolute error",    "up")
    with k3: kpi("📐", CHIP_BLUE,  "RMSE",      f"{metrics['rmse']:.4f}",     "g/cm²", "Root mean square error", "up")
    with k4: kpi("🔄", CHIP_AMBER, "CV Mean R²", f"{np.mean(cv):.3f}" if cv else "—", "", f"σ = {np.std(cv):.4f}" if cv else "", "up")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f'<div class="sec-label">Actual vs. Predicted</div>', unsafe_allow_html=True)
        fig = px.scatter(test_results, x="Actual", y="Predicted",
                         labels={"Actual": "Actual BMD (g/cm²)", "Predicted": "Predicted BMD (g/cm²)"},
                         trendline="ols", template=PLOT_TPL, color_discrete_sequence=[ACCENT])
        mn = min(test_results["Actual"].min(), test_results["Predicted"].min())
        mx = max(test_results["Actual"].max(), test_results["Predicted"].max())
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                  line=dict(dash="dot", color=T3, width=1), showlegend=False))
        st.plotly_chart(style_fig(fig, "Actual vs. Predicted BMD"), width="stretch")

    with c2:
        importances = metrics.get("feature_importances", {})
        if importances:
            imp_df = (pd.DataFrame({"Feature": list(importances.keys()), "Importance": list(importances.values())})
                      .sort_values("Importance").tail(10))
            fig_imp = px.bar(imp_df, y="Feature", x="Importance", orientation="h",
                             template=PLOT_TPL, color="Importance",
                             color_continuous_scale=[[0, ACCENT2], [1, ACCENT]])
            fig_imp.update_coloraxes(showscale=False)
            st.markdown(f'<div class="sec-label">Feature Importance</div>', unsafe_allow_html=True)
            st.plotly_chart(style_fig(fig_imp, "Top 10 Predictive Features"), width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        test_results["Residuals"] = test_results["Actual"] - test_results["Predicted"]
        fig_r = px.scatter(test_results, x="Predicted", y="Residuals", template=PLOT_TPL,
                           labels={"Predicted": "Predicted BMD (g/cm²)", "Residuals": "Residual Error"},
                           color_discrete_sequence=[AMBER])
        fig_r.add_hline(y=0, line_dash="dot", line_color=T3, line_width=1)
        st.markdown(f'<div class="sec-label">Residuals</div>', unsafe_allow_html=True)
        st.plotly_chart(style_fig(fig_r, "Residual Distribution"), width="stretch")

    with c4:
        if cv:
            cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(len(cv))], "R²": cv})
            fig_cv = px.bar(cv_df, x="Fold", y="R²", template=PLOT_TPL, text="R²",
                            color_discrete_sequence=[GREEN])
            fig_cv.update_traces(texttemplate="%{text:.3f}", textposition="inside",
                                 textfont=dict(family="JetBrains Mono, monospace", color="#000000", size=10),
                                 marker_line_width=0, marker_opacity=0.9)
            fig_cv.update_layout(yaxis_range=[0.8, 1.0])
            st.markdown(f'<div class="sec-label">Cross-Validation</div>', unsafe_allow_html=True)
            st.plotly_chart(style_fig(fig_cv, "5-Fold CV R² Scores"), width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    page = render_sidebar()

    if not os.path.exists(PIPELINE_PATH):
        st.error("⚠  Model artifacts not found. Run `python -m src.train` to compile the pipeline.")
        return

    if "Predict" in page:
        page_predict()
    elif "Evaluation" in page:
        page_evaluation()



if __name__ == "__main__":
    main()