"""
BMD (Bone Mineral Density) Prediction Dashboard

A professional, clinical-grade Streamlit dashboard with:
- Model performance metrics & visualizations
- Patient prediction with SHAP explanations
- Clinical insights & methodology

Usage:
    streamlit run app.py
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    ARTIFACTS_DIR,
    BINARY_FEATURES,
    BMD_THRESHOLDS,
    DATA_PATH,
    METRICS_PATH,
    NUMERIC_FEATURES,
    PIPELINE_PATH,
    SHAP_VALUES_PATH,
    TEST_RESULTS_PATH,
    VALIDATION_RANGES,
)
from src.utils import classify_bmd, bmd_risk_description, validate_input

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BMD Predictor — Clinical Dashboard",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — premium dark glassmorphism theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Header styling */
.main-header {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1a3e 50%, #0d1b3e 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    border: 1px solid rgba(79, 195, 247, 0.15);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    margin-bottom: 1.5rem;
}
.main-header h1 {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #a8d0e6;
    font-size: 1rem;
    margin: 0.5rem 0 0 0;
    font-weight: 300;
}

/* Glassmorphism card */
.glass-card {
    background: linear-gradient(135deg, rgba(26,26,62,0.8) 0%, rgba(13,27,62,0.6) 100%);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(79, 195, 247, 0.12);
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 40px rgba(79, 195, 247, 0.1);
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0f1a3a 0%, #162447 100%);
    border: 1px solid rgba(79,195,247,0.15);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.metric-card .metric-label {
    color: #78909c;
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}
.metric-card .metric-value {
    color: #ffffff;
    font-size: 1.8rem;
    font-weight: 700;
}
.metric-card .metric-unit {
    color: #4fc3f7;
    font-size: 0.85rem;
    font-weight: 400;
}

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.5px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e27 0%, #0d1b3e 100%);
}

/* Section divider */
.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(79,195,247,0.3), transparent);
    margin: 1.5rem 0;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Expander styling */
.streamlit-expanderHeader {
    font-weight: 600;
    color: #a8d0e6;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plotly theme constants
# ─────────────────────────────────────────────────────────────────────────────
PLOT_TEMPLATE = "plotly_dark"
PLOT_BG = "rgba(0,0,0,0)"
PAPER_BG = "rgba(0,0,0,0)"
ACCENT_BLUE = "#4fc3f7"
ACCENT_AMBER = "#ffc13b"
ACCENT_CORAL = "#ff6e40"
ACCENT_GREEN = "#4caf50"
COLOR_SCALE = ["#0d47a1", "#1565c0", "#1e88e5", "#42a5f5", "#4fc3f7", "#80deea"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    """Load the trained pipeline."""
    return joblib.load(PIPELINE_PATH)


@st.cache_data
def load_metrics():
    """Load metrics JSON."""
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_test_results():
    """Load test-set actual vs predicted."""
    return pd.read_csv(TEST_RESULTS_PATH)


@st.cache_resource
def load_shap_data():
    """Load pre-computed SHAP data."""
    if os.path.exists(SHAP_VALUES_PATH):
        return joblib.load(SHAP_VALUES_PATH)
    return None


@st.cache_data
def load_raw_data():
    """Load the original CSV for reference ranges."""
    return pd.read_csv(DATA_PATH)


def render_metric_card(label: str, value: str, unit: str = ""):
    """Render a styled metric card."""
    unit_html = f'<div class="metric-unit">{unit}</div>' if unit else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {unit_html}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Model Performance
# ─────────────────────────────────────────────────────────────────────────────
def page_performance():
    """Model performance metrics and visualizations."""
    metrics = load_metrics()
    test_results = load_test_results()

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("R² Score", f"{metrics['r2_score']:.4f}")
    with c2:
        render_metric_card("MAE", f"± {metrics['mae']:.4f}", "g/cm²")
    with c3:
        render_metric_card("RMSE", f"{metrics['rmse']:.4f}", "g/cm²")
    with c4:
        render_metric_card("CV R² (5-fold)", f"{metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.3f}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Plots row 1 ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Actual vs Predicted BMD")
        fig = px.scatter(
            test_results, x="Actual", y="Predicted",
            labels={"Actual": "Actual BMD (g/cm²)", "Predicted": "Predicted BMD (g/cm²)"},
            trendline="ols",
            template=PLOT_TEMPLATE,
            color_discrete_sequence=[ACCENT_BLUE],
        )
        # Perfect prediction line
        mn = min(test_results["Actual"].min(), test_results["Predicted"].min())
        mx = max(test_results["Actual"].max(), test_results["Predicted"].max())
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines", line=dict(dash="dash", color="rgba(255,255,255,0.3)", width=1),
            showlegend=False,
        ))
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20), height=450,
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Top Predictive Features")
        importances = metrics.get("feature_importances", {})
        if importances:
            imp_df = pd.DataFrame({
                "Feature": list(importances.keys()),
                "Importance": list(importances.values()),
            }).sort_values("Importance", ascending=True).tail(15)

            fig_imp = px.bar(
                imp_df, y="Feature", x="Importance", orientation="h",
                template=PLOT_TEMPLATE,
                color="Importance",
                color_continuous_scale=["#0d47a1", "#4fc3f7", "#ffc13b"],
            )
            fig_imp.update_layout(
                margin=dict(l=20, r=20, t=30, b=20), height=450,
                plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Plots row 2 ──
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("Residual Distribution")
        test_results["Residuals"] = test_results["Actual"] - test_results["Predicted"]
        fig_res = px.scatter(
            test_results, x="Predicted", y="Residuals",
            template=PLOT_TEMPLATE,
            labels={"Predicted": "Predicted BMD", "Residuals": "Residual (Actual − Predicted)"},
            color_discrete_sequence=[ACCENT_AMBER],
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
        fig_res.update_layout(
            margin=dict(l=20, r=20, t=30, b=20), height=400,
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        )
        st.plotly_chart(fig_res, use_container_width=True)

    with col_right2:
        st.subheader("Cross-Validation Scores")
        cv_scores = metrics.get("cv_scores", [])
        if cv_scores:
            cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(len(cv_scores))], "R²": cv_scores})
            fig_cv = px.bar(
                cv_df, x="Fold", y="R²",
                template=PLOT_TEMPLATE,
                color_discrete_sequence=[ACCENT_BLUE],
                text="R²",
            )
            fig_cv.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_cv.update_layout(
                margin=dict(l=20, r=20, t=30, b=20), height=400,
                plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
                yaxis_range=[0, 1],
            )
            st.plotly_chart(fig_cv, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Make a Prediction
# ─────────────────────────────────────────────────────────────────────────────
def page_prediction():
    """Patient prediction with SHAP explanations."""
    pipeline = load_pipeline()
    test_results = load_test_results()
    df_ref = load_raw_data()

    st.markdown("Enter patient clinical data to predict Bone Mineral Density (L1-4).")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    input_data = {}

    # ── Patient Biometrics ──
    with st.expander("👤  Patient Biometrics", expanded=True):
        bc1, bc2, bc3, bc4, bc5 = st.columns(5)
        with bc1:
            gender = st.selectbox(
                "Gender", options=[1, 2],
                format_func=lambda x: "♂ Male" if x == 1 else "♀ Female",
                key="gender_input",
            )
            input_data["Gender"] = float(gender)
        with bc2:
            input_data["Age"] = st.number_input("Age (years)", 18, 120, 55, key="age_input")
        with bc3:
            input_data["Height"] = st.number_input("Height (cm)", 100.0, 220.0, 165.0, step=0.5, key="height_input")
        with bc4:
            input_data["Weight"] = st.number_input("Weight (kg)", 25.0, 200.0, 65.0, step=0.5, key="weight_input")
        with bc5:
            input_data["BMI"] = st.number_input("BMI (kg/m²)", 10.0, 60.0, 23.5, step=0.1, key="bmi_input")

    # ── Biochemical Markers ──
    with st.expander("🧪  Biochemical Markers", expanded=True):
        lab_cols = st.columns(5)
        lab_features = [
            ("ALT", "U/L", 1.0, 500.0, 25.0),
            ("AST", "U/L", 1.0, 500.0, 22.0),
            ("BUN", "mmol/L", 0.5, 50.0, 5.5),
            ("CREA", "µmol/L", 10.0, 1500.0, 70.0),
            ("URIC", "µmol/L", 50.0, 900.0, 320.0),
            ("FBG", "mmol/L", 2.0, 30.0, 5.2),
            ("HDL-C", "mmol/L", 0.1, 5.0, 1.3),
            ("LDL-C", "mmol/L", 0.1, 10.0, 2.8),
            ("Ca", "mmol/L", 1.0, 4.0, 2.2),
            ("P", "mmol/L", 0.3, 3.0, 1.1),
            ("Mg", "mmol/L", 0.3, 2.0, 0.85),
        ]
        for i, (feat, unit, lo, hi, default) in enumerate(lab_features):
            with lab_cols[i % 5]:
                input_data[feat] = st.number_input(
                    f"{feat} ({unit})", lo, hi, default, step=0.01,
                    key=f"lab_{feat}",
                )

    # ── Medical History ──
    with st.expander("🏥  Medical History & Comorbidities", expanded=False):
        mh_cols = st.columns(5)
        mh_features = ["HTN", "COPD", "DM", "Hyperlipidaemia", "Hyperuricemia",
                        "AS", "VT", "VD", "CAD", "CKD"]
        for i, feat in enumerate(mh_features):
            with mh_cols[i % 5]:
                val = st.selectbox(
                    feat, options=[0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    key=f"mh_{feat}",
                )
                input_data[feat] = float(val)

    # ── Medications ──
    with st.expander("💊  Current Medications", expanded=False):
        med_cols = st.columns(4)
        med_features = ["Calsium", "Calcitriol", "Bisphosphonate", "Calcitonin"]
        for i, feat in enumerate(med_features):
            with med_cols[i]:
                val = st.selectbox(
                    feat, options=[0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    key=f"med_{feat}",
                )
                input_data[feat] = float(val)

    # ── Lifestyle ──
    with st.expander("🚬  Lifestyle Factors", expanded=False):
        lf_cols = st.columns(2)
        for i, feat in enumerate(["Smoking", "Drinking"]):
            with lf_cols[i]:
                val = st.selectbox(
                    feat, options=[0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes",
                    key=f"lf_{feat}",
                )
                input_data[feat] = float(val)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Predict ──
    if st.button("🔬  Predict BMD", type="primary", use_container_width=True):
        # Validate
        validation = validate_input(input_data)
        if validation["warnings"]:
            for w in validation["warnings"]:
                st.warning(f"⚠️ {w}")

        # Build input DataFrame
        input_df = pd.DataFrame([input_data])
        prediction = pipeline.predict(input_df)[0]

        label, color = classify_bmd(prediction)
        description = bmd_risk_description(label)

        # ── Results row ──
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        res_left, res_right = st.columns([1, 1])

        with res_left:
            st.markdown("### Prediction Result")

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                number={"suffix": " g/cm²", "font": {"size": 28, "color": "#ffffff"}},
                title={"text": "Predicted BMD (L1-4)", "font": {"size": 16, "color": "#a8d0e6"}},
                gauge={
                    "axis": {"range": [0.3, 1.5], "tickcolor": "#546e7a",
                             "tickfont": {"color": "#78909c"}},
                    "bar": {"color": color, "thickness": 0.3},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0.3, 0.8], "color": "rgba(244,67,54,0.15)"},
                        {"range": [0.8, 1.0], "color": "rgba(255,152,0,0.15)"},
                        {"range": [1.0, 1.5], "color": "rgba(76,175,80,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "#ffffff", "width": 3},
                        "thickness": 0.8,
                        "value": prediction,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=300,
                margin=dict(l=30, r=30, t=60, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#ffffff"},
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Classification badge
            st.markdown(f"""
            <div style="text-align: center; margin-top: -1rem;">
                <span class="risk-badge" style="background: {color}22; color: {color}; border: 1px solid {color}44;">
                    {label}
                </span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="glass-card" style="margin-top: 1rem;">
                <p style="color: #b0bec5; font-size: 0.9rem; line-height: 1.6; margin: 0;">{description}</p>
            </div>
            """, unsafe_allow_html=True)

        with res_right:
            st.markdown("### SHAP Explanation")
            st.caption("How each feature influenced this prediction")

            try:
                import shap
                import matplotlib.pyplot as plt
                import matplotlib

                matplotlib.use("Agg")

                gb_model = pipeline.named_steps["model"]
                preprocessor = pipeline.named_steps["preprocessor"]

                # Feature names after transformation
                feature_names = []
                for name, _, cols in preprocessor.transformers_:
                    feature_names.extend(cols)

                X_transformed = preprocessor.transform(input_df)
                explainer = shap.TreeExplainer(gb_model)
                shap_vals = explainer.shap_values(X_transformed)

                # Handle SHAP 0.51+ where expected_value may be an ndarray
                base_val = explainer.expected_value
                if hasattr(base_val, "__len__"):
                    base_val = float(base_val[0]) if len(base_val) > 0 else float(base_val)
                else:
                    base_val = float(base_val)

                # Get single-sample values as 1D array
                sv = np.array(shap_vals[0]).flatten()
                xv = np.array(X_transformed[0]).flatten()

                # Create waterfall plot
                explanation = shap.Explanation(
                    values=sv,
                    base_values=base_val,
                    feature_names=feature_names,
                    data=xv,
                )

                fig_shap, ax = plt.subplots(figsize=(8, 6))
                fig_shap.patch.set_facecolor("#0a0e27")
                ax.set_facecolor("#0a0e27")

                shap.plots.waterfall(explanation, max_display=12, show=False)

                # Re-grab the current axes (SHAP creates its own)
                ax = plt.gca()
                fig_shap = plt.gcf()
                fig_shap.patch.set_facecolor("#0a0e27")
                ax.set_facecolor("#0a0e27")

                # Style the matplotlib plot for dark theme
                for text in fig_shap.findobj(matplotlib.text.Text):
                    text.set_color("#e0e0e0")
                ax.tick_params(colors="#b0bec5")
                for spine in ax.spines.values():
                    spine.set_color("#37474f")
                ax.xaxis.label.set_color("#a8d0e6")
                ax.yaxis.label.set_color("#a8d0e6")
                if ax.get_title():
                    ax.title.set_color("#ffffff")

                plt.tight_layout()
                st.pyplot(fig_shap)
                plt.close("all")

            except ImportError:
                st.info("💡 Install `shap` to see feature-level explanations.")
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

        # ── Distribution context ──
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("Patient Position in BMD Distribution")

        fig_dist = px.histogram(
            test_results, x="Actual", nbins=30,
            template=PLOT_TEMPLATE,
            color_discrete_sequence=["#1e88e5"],
            labels={"Actual": "BMD (g/cm²)", "count": "Patients"},
        )
        fig_dist.add_vline(
            x=prediction, line_width=3, line_dash="dash", line_color=ACCENT_CORAL,
            annotation_text=f"Patient: {prediction:.4f}",
            annotation_position="top right",
            annotation_font_color=ACCENT_CORAL,
        )
        # Zone shading
        fig_dist.add_vrect(x0=0, x1=0.8, fillcolor="rgba(244,67,54,0.05)", line_width=0)
        fig_dist.add_vrect(x0=0.8, x1=1.0, fillcolor="rgba(255,152,0,0.05)", line_width=0)
        fig_dist.add_vrect(x0=1.0, x1=2.0, fillcolor="rgba(76,175,80,0.05)", line_width=0)
        fig_dist.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        )
        st.plotly_chart(fig_dist, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Clinical Insights
# ─────────────────────────────────────────────────────────────────────────────
def page_insights():
    """Clinical significance and methodology."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #4fc3f7; margin-top: 0;">🎯 Clinical Significance</h3>
            <p style="color: #b0bec5; line-height: 1.8;">
                <strong style="color: #ffffff;">Dual-energy X-ray Absorptiometry (DXA)</strong> scans
                are the gold standard for diagnosing osteoporosis, but they are <strong style="color: #ffc13b;">
                expensive, radiation-emitting, and not widely accessible</strong> in primary care settings.
            </p>
            <p style="color: #b0bec5; line-height: 1.8;">
                This model predicts BMD from <strong style="color: #4caf50;">routine blood markers</strong>
                and clinical data that are already collected during standard check-ups. It serves as a
                <strong style="color: #ffffff;">pre-screening tool</strong> to identify patients who
                would most benefit from a confirmatory DXA scan.
            </p>
            <p style="color: #b0bec5; line-height: 1.8;">
                By targeting DXA referrals to high-risk patients, healthcare systems can:
            </p>
            <ul style="color: #b0bec5; line-height: 2;">
                <li>Reduce unnecessary imaging costs</li>
                <li>Improve early detection of osteoporosis</li>
                <li>Enable timely pharmacological intervention</li>
                <li>Decrease fracture-related morbidity in aging populations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #ffc13b; margin-top: 0;">⚙️ Methodology</h3>
            <p style="color: #b0bec5; line-height: 1.8;">
                <strong style="color: #ffffff;">Model:</strong> Gradient Boosting Regressor with
                hyperparameters optimized via 3-fold GridSearchCV.
            </p>
            <p style="color: #b0bec5; line-height: 1.8;">
                <strong style="color: #ffffff;">Pipeline Architecture:</strong>
            </p>
            <ol style="color: #b0bec5; line-height: 2;">
                <li><strong style="color: #4fc3f7;">Imputation</strong> — Median for numeric, mode for binary</li>
                <li><strong style="color: #4fc3f7;">Scaling</strong> — RobustScaler (outlier-resistant)</li>
                <li><strong style="color: #4fc3f7;">Model</strong> — GradientBoostingRegressor</li>
            </ol>
            <p style="color: #b0bec5; line-height: 1.8;">
                <strong style="color: #ffffff;">Explainability:</strong> SHAP (TreeExplainer) provides
                per-patient feature attribution, showing exactly which markers drove the prediction.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #ff6e40; margin-top: 0;">⚠️ Limitations & Disclaimers</h3>
        <ul style="color: #b0bec5; line-height: 2.2;">
            <li>This tool is a <strong style="color: #ffffff;">screening aid</strong>, NOT a diagnostic device.
                Clinical decisions must involve a licensed physician.</li>
            <li>Predictions are based on a dataset of ~1,500 patients. External validation on a separate
                cohort is recommended before clinical deployment.</li>
            <li>BMD classification thresholds used here are approximate and based on L1-4 absolute values
                rather than standardized T-scores.</li>
            <li>The model does not account for medication history duration, dietary patterns, or genetic factors
                beyond the provided features.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Features used
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("📋 Features Used in Prediction")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.markdown("**🔢 Numeric (Continuous)**")
        for f in NUMERIC_FEATURES:
            rng = VALIDATION_RANGES.get(f)
            if rng:
                st.markdown(f"- `{f}` — {rng[0]}–{rng[1]} {rng[2]}")
            else:
                st.markdown(f"- `{f}`")
    with fc2:
        st.markdown("**🏥 Comorbidities (Binary)**")
        for f in ["HTN", "COPD", "DM", "Hyperlipidaemia", "Hyperuricemia", "AS", "VT", "VD", "CAD", "CKD"]:
            st.markdown(f"- `{f}`")
    with fc3:
        st.markdown("**💊 Medications & Lifestyle**")
        for f in ["Calsium", "Calcitriol", "Bisphosphonate", "Calcitonin", "Smoking", "Drinking"]:
            st.markdown(f"- `{f}`")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🦴 BMD Clinical Prediction Dashboard</h1>
        <p>Bone Mineral Density estimation from routine blood markers — powered by Gradient Boosting & SHAP</p>
    </div>
    """, unsafe_allow_html=True)

    # Check for model
    if not os.path.exists(PIPELINE_PATH):
        st.error("⚠️ Trained pipeline not found. Run `python -m src.train` first.")
        st.code("cd BM && python -m src.train --fast", language="bash")
        return

    # Sidebar navigation
    st.sidebar.markdown("## 🧭 Navigation")
    page = st.sidebar.radio(
        "Select page",
        ["📊 Model Performance", "🔬 Make a Prediction", "📖 Clinical Insights"],
        label_visibility="collapsed",
    )

    if page == "📊 Model Performance":
        page_performance()
    elif page == "🔬 Make a Prediction":
        page_prediction()
    else:
        page_insights()

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="color: #546e7a; font-size: 0.8rem; line-height: 1.6;">
        <strong>BMD Predictor v2.0</strong><br>
        Clinical-grade ML pipeline<br>
        GBR + SHAP Explainability<br><br>
        ⚕️ For research use only
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
