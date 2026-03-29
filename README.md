# 🦴 BMD Clinical Prediction Pipeline

**Predicting Bone Mineral Density from routine blood markers using Machine Learning**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)

---

## Clinical Significance

**Dual-energy X-ray Absorptiometry (DXA)** scans are the gold standard for diagnosing osteoporosis, but they are **expensive**, **radiation-emitting**, and **not widely accessible** in primary care.

This project trains a **Gradient Boosting Regressor** to predict lumbar spine BMD (L1-4) from **routine clinical data** — demographics, biochemical markers, comorbidities, and medications — that are already collected during standard check-ups.

**Use case:** Pre-screening tool to identify high-risk patients who would benefit most from a confirmatory DXA scan, enabling targeted referrals and reducing unnecessary imaging costs.

---

## Project Structure

```
BM/
├── app.py                  # Streamlit dashboard (3-page clinical UI)
├── requirements.txt        # Pinned dependencies
├── README.md               # This file
├── UA.csv                  # Source dataset (1,537 patients)
├── src/                    # Source package
│   ├── __init__.py
│   ├── config.py           # Feature definitions, thresholds, hyperparams
│   ├── utils.py            # Validation, classification, data prep helpers
│   └── train.py            # Pipeline training + SHAP computation
└── artifacts/              # Generated model artifacts
    ├── bmd_pipeline.pkl    # Trained sklearn Pipeline (single file)
    ├── metrics.json        # R², MAE, RMSE, CV scores, feature importances
    ├── test_results.csv    # Actual vs predicted (test set)
    └── shap_values.pkl     # Pre-computed SHAP values
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
# Full hyperparameter search (~5-10 min)
python -m src.train

# Fast mode for development (~1-2 min)
python -m src.train --fast
```

### 3. Launch the dashboard

```bash
streamlit run app.py
```

---

## Model Methodology

| Component        | Technology                                                    |
|------------------|---------------------------------------------------------------|
| **Core Model**   | `GradientBoostingRegressor` (tuned via `GridSearchCV`)        |
| **Preprocessing**| `ColumnTransformer` → `SimpleImputer` + `RobustScaler`       |
| **Persistence**  | Single `joblib` file containing entire `Pipeline` object      |
| **Explainability**| SHAP `TreeExplainer` — per-patient waterfall plots           |
| **Frontend**     | Streamlit with `st.cache_resource` for model loading          |

### Feature Groups

- **Biometrics:** Age, Gender, Height, Weight, BMI
- **Biochemical Markers:** ALT, AST, BUN, CREA, URIC, FBG, HDL-C, LDL-C, Ca, P, Mg
- **Comorbidities:** HTN, COPD, DM, Hyperlipidaemia, Hyperuricemia, AS, VT, VD, CAD, CKD
- **Medications:** Calcium, Calcitriol, Bisphosphonate, Calcitonin
- **Lifestyle:** Smoking, Drinking

### Excluded Features (Data Leakage Prevention)

T-scores (`L1.4T`, `FNT`, `TLT`), other BMD sites (`FN`, `TL`), and diagnosis columns (`OP`, `Fracture`) are **removed** from training to prevent data leakage.

---

## Dashboard Pages

1. **📊 Model Performance** — R², MAE, RMSE metrics; Actual vs Predicted scatter; Feature importance; Residuals; Cross-validation scores
2. **🔬 Make a Prediction** — Grouped input form with validation; Gauge chart with risk zones; SHAP waterfall explanation; Distribution context
3. **📖 Clinical Insights** — Clinical significance; Methodology; Limitations; Feature reference

---

## Limitations

- **Screening tool only** — not a diagnostic device. Clinical decisions must involve a licensed physician.
- Trained on ~1,500 patients. External validation recommended before deployment.
- BMD thresholds are approximate L1-4 absolute values, not standardized T-scores.
- Does not account for medication duration, dietary patterns, or genetic factors.

---

## License

For research and educational purposes only.
