"""
Centralized configuration for the BMD Prediction Pipeline.

Defines feature groups, target variables, leak columns,
and clinically valid ranges for input validation.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "BMD_FEMALE.csv")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "bmd_pipeline.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
TEST_RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "test_results.csv")
SHAP_VALUES_PATH = os.path.join(ARTIFACTS_DIR, "shap_values.pkl")

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
TARGET_COL = "L1-4"

# ---------------------------------------------------------------------------
# Leak / exclusion columns  (derived from BMD scan — NOT clinical inputs)
# ---------------------------------------------------------------------------
LEAK_COLS = [
    "L1-4",   # target itself
    "L1.4T",  # T-score at L1-4 (mathematical derivative of BMD)
    "FN",     # Femoral Neck BMD (another scan site)
    "FNT",    # Femoral Neck T-score
    "TL",     # Total Hip BMD (another scan site)
    "TLT",    # Total Hip T-score
    "OP",     # Osteoporosis diagnosis (determined from scan)
    "Fracture",  # Fragility fracture (often linked to scan results)
]

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
# Continuous / numeric clinical features
NUMERIC_FEATURES = [
    "Age", "Height", "Weight", "BMI",
    # Biochemical markers
    "ALT", "AST", "BUN", "CREA", "URIC", "FBG",
    "HDL-C", "LDL-C", "Ca", "P", "Mg",
]

# Binary / categorical features (0/1 flags)
BINARY_FEATURES = [
    # Medications
    "Calsium", "Calcitriol", "Bisphosphonate", "Calcitonin",
    # Comorbidities
    "HTN", "COPD", "DM", "Hyperlipidaemia", "Hyperuricemia",
    "AS", "VT", "VD", "CAD", "CKD",
    # Lifestyle
    "Smoking", "Drinking",
]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES

# ---------------------------------------------------------------------------
# Clinical validation ranges (used in the dashboard input form)
# ---------------------------------------------------------------------------
VALIDATION_RANGES = {
    "Age":    (18, 120,  "years"),
    "Height": (100, 220, "cm"),
    "Weight": (25, 200,  "kg"),
    "BMI":    (10, 60,   "kg/m²"),
    "ALT":    (1, 500,   "U/L"),
    "AST":    (1, 500,   "U/L"),
    "BUN":    (0.5, 50,  "mmol/L"),
    "CREA":   (10, 1500, "µmol/L"),
    "URIC":   (50, 900,  "µmol/L"),
    "FBG":    (2, 30,    "mmol/L"),
    "HDL-C":  (0.1, 5,   "mmol/L"),
    "LDL-C":  (0.1, 10,  "mmol/L"),
    "Ca":     (1.0, 4.0, "mmol/L"),
    "P":      (0.3, 3.0, "mmol/L"),
    "Mg":     (0.3, 2.0, "mmol/L"),
}

# ---------------------------------------------------------------------------
# BMD classification thresholds (approximate L1-4 g/cm²)
# ---------------------------------------------------------------------------
BMD_THRESHOLDS = {
    "normal_min":     1.000,   # ≥ 1.0 → Normal
    "osteopenia_min": 0.800,   # 0.8 – 1.0 → Osteopenia
    # < 0.8 → Osteoporosis
}

# ---------------------------------------------------------------------------
# Hyper-parameter grid for GridSearchCV
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "model__n_estimators":     [100, 200, 300],
    "model__max_depth":        [3, 5, 7],
    "model__learning_rate":    [0.05, 0.1, 0.2],
    "model__min_samples_split": [2, 5, 10],
}

# Use a smaller grid for faster iteration during development
PARAM_GRID_FAST = {
    "model__n_estimators":     [100, 200],
    "model__max_depth":        [3, 5],
    "model__learning_rate":    [0.05, 0.1],
    "model__min_samples_split": [2, 5],
}
