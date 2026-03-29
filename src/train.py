"""
BMD Prediction Pipeline — Training Script

Trains a GradientBoostingRegressor inside a scikit-learn Pipeline
with proper imputation, scaling, and hyperparameter tuning.

Usage:
    python -m src.train            # Full grid search
    python -m src.train --fast     # Smaller grid (dev mode)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

# ── project imports ──────────────────────────────────────────────────────────
# Add project root to path so `src` is importable when run as `python -m src.train`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    ALL_FEATURES,
    ARTIFACTS_DIR,
    DATA_PATH,
    METRICS_PATH,
    PARAM_GRID,
    PARAM_GRID_FAST,
    PIPELINE_PATH,
    SHAP_VALUES_PATH,
    TARGET_COL,
    TEST_RESULTS_PATH,
)
from src.utils import get_feature_columns, prepare_dataframe

warnings.filterwarnings("ignore", category=FutureWarning)


def build_pipeline(numeric_features: list[str], binary_features: list[str]) -> Pipeline:
    """Construct the full sklearn Pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", __import__("sklearn.preprocessing", fromlist=["RobustScaler"]).RobustScaler()),
                ]),
                numeric_features,
            ),
            (
                "binary",
                SimpleImputer(strategy="most_frequent"),
                binary_features,
            ),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(random_state=42)),
    ])


def train(fast: bool = False) -> None:
    """End-to-end training routine."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ── 1. Load & clean ──────────────────────────────────────────────────
    print(f"📂  Loading data from {DATA_PATH} …")
    df = pd.read_csv(DATA_PATH)
    df = prepare_dataframe(df)

    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET_COL])
    print(f"    → {len(df)} rows, {len(df.columns)} columns after cleaning.")

    # ── 2. Feature / target split ────────────────────────────────────────
    numeric_cols, binary_cols = get_feature_columns(df)
    feature_cols = numeric_cols + binary_cols

    print(f"    → {len(numeric_cols)} numeric + {len(binary_cols)} binary features = {len(feature_cols)} total")
    print(f"    → Target: {TARGET_COL}")

    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    print(f"    → Train: {len(X_train)}, Test: {len(X_test)}")

    # ── 3. Build pipeline ────────────────────────────────────────────────
    pipeline = build_pipeline(numeric_cols, binary_cols)

    # ── 4. Hyperparameter tuning ─────────────────────────────────────────
    grid = PARAM_GRID_FAST if fast else PARAM_GRID
    mode_label = "FAST" if fast else "FULL"
    n_fits = 1
    for v in grid.values():
        n_fits *= len(v)
    n_fits *= 3  # 3-fold CV
    print(f"\n🔍  GridSearchCV ({mode_label}): {n_fits} fits …")

    search = GridSearchCV(
        pipeline,
        param_grid=grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    print(f"    → Best params: {search.best_params_}")
    print(f"    → Best CV R²:  {search.best_score_:.4f}")

    # ── 5. Evaluate on held-out test set ─────────────────────────────────
    y_pred = best_pipeline.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n📊  Test-set metrics:")
    print(f"    R²   = {r2:.4f}")
    print(f"    MAE  = {mae:.4f} g/cm²")
    print(f"    RMSE = {rmse:.4f} g/cm²")

    # Cross-validation scores on full training set
    cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring="r2")
    print(f"    5-fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── 6. Feature importance ────────────────────────────────────────────
    gb_model = best_pipeline.named_steps["model"]
    # Get transformed feature names
    preprocessor = best_pipeline.named_steps["preprocessor"]
    transformed_feature_names = numeric_cols + binary_cols
    importances = dict(
        zip(transformed_feature_names, gb_model.feature_importances_.tolist())
    )

    # ── 7. SHAP values ──────────────────────────────────────────────────
    print("\n🧠  Computing SHAP values …")
    try:
        import shap

        X_test_transformed = preprocessor.transform(X_test)
        explainer = shap.TreeExplainer(gb_model)
        shap_values = explainer.shap_values(X_test_transformed)

        shap_data = {
            "shap_values": shap_values,
            "X_test_transformed": X_test_transformed,
            "feature_names": transformed_feature_names,
            "expected_value": float(explainer.expected_value),
        }
        joblib.dump(shap_data, SHAP_VALUES_PATH)
        print("    → SHAP values saved.")
    except ImportError:
        print("    ⚠  shap not installed — skipping SHAP computation.")
    except Exception as e:
        print(f"    ⚠  SHAP computation failed: {e}")

    # ── 8. Persist artefacts ────────────────────────────────────────────
    print("\n💾  Saving artefacts …")

    # Pipeline (single file)
    joblib.dump(best_pipeline, PIPELINE_PATH)
    print(f"    → Pipeline → {PIPELINE_PATH}")

    # Metrics JSON
    metrics = {
        "best_model_name": "GradientBoostingRegressor",
        "best_params": search.best_params_,
        "r2_score": r2,
        "mae": mae,
        "rmse": rmse,
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "cv_scores": cv_scores.tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importances": importances,
        "features_used": transformed_feature_names,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"    → Metrics  → {METRICS_PATH}")

    # Test results CSV
    results_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred,
    })
    results_df.to_csv(TEST_RESULTS_PATH, index=False)
    print(f"    → Test CSV → {TEST_RESULTS_PATH}")

    print("\n✅  Training complete!")


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the BMD prediction pipeline.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a smaller hyperparameter grid for faster iteration.",
    )
    args = parser.parse_args()
    train(fast=args.fast)
