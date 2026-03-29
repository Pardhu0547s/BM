import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os

# Set page config
st.set_page_config(
    page_title="BMD Model Dashboard",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium dark theme
st.markdown("""
    <style>
    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .stMetric label {
        color: #a8d0e6 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Plotly dark template matching Streamlit's dark theme
PLOT_TEMPLATE = "plotly_dark"
PLOT_BG = "rgba(0,0,0,0)"
PAPER_BG = "rgba(0,0,0,0)"

@st.cache_resource
def load_data_and_model():
    base_path = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_path, 'bmd_model.pkl')
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    metrics_path = os.path.join(base_path, 'metrics.json')
    test_results_path = os.path.join(base_path, 'test_results.csv')
    feature_importances_path = os.path.join(base_path, 'feature_importances.json')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    test_results = pd.read_csv(test_results_path)

    feature_importances = None
    if os.path.exists(feature_importances_path):
        with open(feature_importances_path, 'r') as f:
            feature_importances = json.load(f)

    # Also get UA.csv path for the prediction tool
    data_path = os.path.join(base_path, 'UA.csv')

    return model, scaler, metrics, test_results, feature_importances, data_path

def main():
    st.title("🦴 BMD (Bone Mineral Density) Prediction Dashboard")
    st.markdown("---")

    base_path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(base_path, 'bmd_model.pkl')):
        st.error("Model files not found. Please run `train_bmd_model.py` first.")
        return

    try:
        model, scaler, metrics, test_results, feature_importances, data_path = load_data_and_model()
    except Exception as e:
        st.error(f"Error loading model data: {e}")
        return

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Model Performance", "Make a Prediction"])

    if page == "Model Performance":
        # Metrics Section
        st.header("📈 Model Performance Metrics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Best Model", metrics['best_model_name'])
        with col2:
            st.metric("R² Score", f"{metrics['r2_score']:.4f}")
        with col3:
            st.metric("RMSE", f"{metrics['rmse']:.4f}")

        st.markdown("---")

        # Visualization Section
        col_plot1, col_plot2 = st.columns(2)

        with col_plot1:
            st.subheader("Actual vs Predicted BMD (L1-4)")
            fig = px.scatter(
                test_results,
                x="Actual",
                y="Predicted",
                labels={'Actual': 'Actual BMD', 'Predicted': 'Predicted BMD'},
                trendline="ols",
                template=PLOT_TEMPLATE,
                color_discrete_sequence=['#4fc3f7']
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                height=500,
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=PAPER_BG,
            )
            st.plotly_chart(fig, width='stretch')

        with col_plot2:
            if feature_importances:
                st.subheader("Key Predictive Features")
                importance_df = pd.DataFrame({
                    'Feature': list(feature_importances.keys()),
                    'Importance': list(feature_importances.values())
                }).sort_values(by='Importance', ascending=True)

                fig_imp = px.bar(
                    importance_df,
                    y='Feature',
                    x='Importance',
                    orientation='h',
                    template=PLOT_TEMPLATE,
                    color='Importance',
                    color_continuous_scale='OrRd',
                )
                fig_imp.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=500,
                    plot_bgcolor=PLOT_BG,
                    paper_bgcolor=PAPER_BG,
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_imp, width='stretch')
            else:
                st.info("Feature importance not available for this model type.")

        # Residuals Plot
        st.markdown("---")
        st.subheader("Residual Plot")
        test_results['Residuals'] = test_results['Actual'] - test_results['Predicted']
        fig_res = px.scatter(
            test_results,
            x='Predicted',
            y='Residuals',
            template=PLOT_TEMPLATE,
            labels={'Predicted': 'Predicted Value', 'Residuals': 'Residual (Actual - Predicted)'},
            color_discrete_sequence=['#ffc13b']
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="white")
        fig_res.update_layout(
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=PAPER_BG,
        )
        st.plotly_chart(fig_res, width='stretch')

    else:
        st.header("🔮 Make a Prediction")
        st.markdown("Enter patient details below to predict their BMD (L1-4).")

        # Load sample data to get columns and ranges
        df_sample = pd.read_csv(data_path)
        exclude_cols = ['L1-4', 'L1.4T', 'FN', 'FNT', 'TL', 'TLT']
        input_cols = [col for col in df_sample.columns if col not in exclude_cols]

        # Create input form
        input_data = {}
        cols = st.columns(3)
        for i, col_name in enumerate(input_cols):
            with cols[i % 3]:
                min_val = float(df_sample[col_name].min())
                max_val = float(df_sample[col_name].max())
                avg_val = float(df_sample[col_name].mean())

                # Gender: restrict to 1 (Male) or 2 (Female)
                if col_name.lower() == 'gender':
                    gender_choice = st.selectbox(
                        "Gender",
                        options=[1, 2],
                        format_func=lambda x: "1 - Male" if x == 1 else "2 - Female",
                        index=0
                    )
                    input_data[col_name] = float(gender_choice)
                else:
                    # Determine a reasonable step size
                    val_range = max_val - min_val
                    if val_range > 0:
                        step = round(val_range / 100, 4)
                        if step == 0:
                            step = 0.01
                    else:
                        step = 0.01

                    input_data[col_name] = st.number_input(
                        f"{col_name}",
                        min_value=min_val,
                        max_value=max_val,
                        value=round(avg_val, 4),
                        step=step,
                        format="%.4f"
                    )

        if st.button("Predict BMD"):
            # Prepare data
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]

            st.success(f"### Predicted BMD (L1-4): {prediction:.4f}")

            # Show position relative to training data
            st.markdown("---")
            st.info("Patient's predicted BMD relative to test set distribution:")
            fig_dist = px.histogram(
                test_results,
                x="Actual",
                nbins=30,
                template=PLOT_TEMPLATE,
                title="BMD Distribution (Test Set)",
                color_discrete_sequence=['#a8d0e6']
            )
            fig_dist.add_vline(x=prediction, line_width=4, line_dash="dash", line_color="#ff6e40",
                               annotation_text="Predicted Patient Value",
                               annotation_position="top right")
            fig_dist.update_layout(
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=PAPER_BG,
            )
            st.plotly_chart(fig_dist, width='stretch')

if __name__ == "__main__":
    main()
