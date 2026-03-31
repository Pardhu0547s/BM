from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import json
import os
import sys

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Define features manually for input validation
class PredictRequest(BaseModel):
    Age: float = Field(..., ge=18, le=120)
    Height: float = Field(..., ge=100.0, le=220.0)
    Weight: float = Field(..., ge=25.0, le=200.0)
    BMI: float = Field(..., ge=10.0, le=60.0)
    ALT: float = Field(..., ge=1.0, le=500.0)
    AST: float = Field(..., ge=1.0, le=500.0)
    BUN: float = Field(..., ge=0.5, le=50.0)
    CREA: float = Field(..., ge=10.0, le=1500.0)
    URIC: float = Field(..., ge=50.0, le=900.0)
    FBG: float = Field(..., ge=2.0, le=30.0)
    # pydantic requires valid field names (no hyphens) using alias
    HDL_C: float = Field(..., alias='HDL-C', ge=0.1, le=5.0)
    LDL_C: float = Field(..., alias='LDL-C', ge=0.1, le=10.0)
    Ca: float = Field(..., ge=1.0, le=4.0)
    P: float = Field(..., ge=0.3, le=3.0)
    Mg: float = Field(..., ge=0.3, le=2.0)
    Calsium: float = Field(..., ge=0, le=1)
    Calcitriol: float = Field(..., ge=0, le=1)
    Bisphosphonate: float = Field(..., ge=0, le=1)
    Calcitonin: float = Field(..., ge=0, le=1)
    HTN: float = Field(..., ge=0, le=1)
    COPD: float = Field(..., ge=0, le=1)
    DM: float = Field(..., ge=0, le=1)
    Hyperlipidaemia: float = Field(..., ge=0, le=1)
    Hyperuricemia: float = Field(..., ge=0, le=1)
    AS: float = Field(..., ge=0, le=1)
    VT: float = Field(..., ge=0, le=1)
    VD: float = Field(..., ge=0, le=1)
    CAD: float = Field(..., ge=0, le=1)
    CKD: float = Field(..., ge=0, le=1)
    Smoking: float = Field(..., ge=0, le=1)
    Drinking: float = Field(..., ge=0, le=1)

app = FastAPI(title="OsteoScan API")

# Setup CORS to allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, lock down to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "bmd_pipeline.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
TEST_RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "test_results.csv")
SHAP_VALUES_PATH = os.path.join(ARTIFACTS_DIR, "shap_values.pkl")

# Load compiled model safely
pipeline = None
if os.path.exists(PIPELINE_PATH):
    try:
        pipeline = joblib.load(PIPELINE_PATH)
    except Exception as e:
        print(f"Failed to load pipeline: {e}")

@app.get("/")
def health_check():
    return {"status": "ok", "service": "OsteoScan API"}

@app.get("/api/metrics")
def get_metrics():
    """Retrieve full evaluation metrics."""
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="Metrics file not found. Run model training first.")
    with open(METRICS_PATH, "r") as f:
        data = json.load(f)
        
    if os.path.exists(TEST_RESULTS_PATH):
        df = pd.read_csv(TEST_RESULTS_PATH)
        df["Residual"] = df["Actual"] - df["Predicted"]
        data["test_results"] = df.to_dict(orient="records")
    else:
        data["test_results"] = []
        
    return data

@app.post("/api/predict")
def predict_bmd(request: PredictRequest):
    """Predict BMD securely using standard pipeline structure."""
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline model unavailable.")

    # Format the payload identically to pandas df
    input_data = request.model_dump(by_alias=True)
    input_df = pd.DataFrame([input_data])
    
    # Run absolute prediction
    prediction = pipeline.predict(input_df)[0]
    
    # Classify clinical outcome
    if prediction >= 1.000:
        label = "Normal"
        color = "#0F9D58"
    elif prediction >= 0.800:
        label = "Osteopenia"
        color = "#F4B400"
    else:
        label = "Osteoporosis"
        color = "#DB4437"
        
    # Standard formula to map linear continuous space to a percentage representation (0-100 gauge map)
    bmd_pct = ((prediction - 0.3) / (1.6 - 0.3)) * 100
    
    # Generate generic active SHAP Waterfall points for this exact Patient
    shap_attr = []
    expected_value = 0
    if SHAP_AVAILABLE:
        try:
            gb_model = pipeline.named_steps["model"]
            preprocessor = pipeline.named_steps["preprocessor"]
            
            feat_names = []
            for name, _, cols in preprocessor.transformers_:
                feat_names.extend(cols)
                
            X_t = preprocessor.transform(input_df)
            explainer = shap.TreeExplainer(gb_model)
            shap_vals = explainer.shap_values(X_t)
            base_val = explainer.expected_value
            expected_value = float(base_val[0]) if hasattr(base_val, "__len__") else float(base_val)
            
            # Zip dynamically
            sv = list(np.array(shap_vals[0]).flatten())
            xv = list(np.array(X_t[0]).flatten())
            
            for i, val in enumerate(sv):
                shap_attr.append({
                    "feature": feat_names[i],
                    "value": xv[i],
                    "contribution": val,
                    "importance": abs(val) # to sort for the top N waterfall view
                })
                
            shap_attr = sorted(shap_attr, key=lambda x: x["importance"], reverse=True)[:10]

        except Exception as e:
            print(f"SHAP attribution failed: {e}")
    else:
        print("SHAP is not installed, skipping feature attribution.")
        
    return {
        "prediction_bmd": float(prediction),
        "label": label,
        "color": color,
        "risk_score_pct": max(0, min(100, bmd_pct)),
        "shap_expected_value": expected_value,
        "shap_waterfall": shap_attr
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
