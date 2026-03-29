"""
Utility helpers for the BMD Prediction Pipeline.

Functions for feature extraction, input validation,
and BMD clinical classification.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from src.config import (
    LEAK_COLS,
    NUMERIC_FEATURES,
    BINARY_FEATURES,
    ALL_FEATURES,
    VALIDATION_RANGES,
    BMD_THRESHOLDS,
)


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Return (numeric_cols, binary_cols) present in the dataframe,
    after removing any leak / target columns.
    """
    available = set(df.columns) - set(LEAK_COLS)
    numeric = [c for c in NUMERIC_FEATURES if c in available]
    binary  = [c for c in BINARY_FEATURES  if c in available]
    return numeric, binary


def validate_input(data: Dict[str, float]) -> Dict[str, List[str]]:
    """
    Validate a single patient's input dict against clinical ranges.

    Returns a dict with 'warnings' and 'errors' lists.
    """
    messages: Dict[str, List[str]] = {"warnings": [], "errors": []}

    for feat, value in data.items():
        if feat in VALIDATION_RANGES:
            lo, hi, unit = VALIDATION_RANGES[feat]
            if value < lo or value > hi:
                messages["warnings"].append(
                    f"{feat} = {value} is outside the expected clinical range "
                    f"[{lo}–{hi} {unit}]."
                )

    return messages


def classify_bmd(value: float) -> Tuple[str, str]:
    """
    Classify a predicted BMD value into a WHO-approximate category.

    Returns (label, color_hex).
    """
    if value >= BMD_THRESHOLDS["normal_min"]:
        return "Normal", "#4caf50"
    elif value >= BMD_THRESHOLDS["osteopenia_min"]:
        return "Osteopenia", "#ff9800"
    else:
        return "Osteoporosis", "#f44336"


def bmd_risk_description(label: str) -> str:
    """Return a short clinical description for each BMD category."""
    descriptions = {
        "Normal": (
            "Bone mineral density is within the healthy reference range. "
            "Continue current lifestyle and preventive measures."
        ),
        "Osteopenia": (
            "Bone mineral density is below optimal levels, indicating early "
            "bone loss. Consider dietary calcium, vitamin D, and weight-bearing "
            "exercise. Follow-up DXA scan recommended."
        ),
        "Osteoporosis": (
            "Bone mineral density is significantly reduced, indicating high "
            "fracture risk. Urgent clinical evaluation and pharmacological "
            "intervention may be warranted. DXA scan confirmation recommended."
        ),
    }
    return descriptions.get(label, "")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the raw CSV dataframe.
    
    - Coerce all columns to numeric
    - Drop rows where target is NaN
    """
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
