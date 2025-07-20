# api/app.py

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Football Win Probability API")

# ────────────────────────────────────────────────────────────────────────────────
# Pydantic model: define the 17 input features
# ────────────────────────────────────────────────────────────────────────────────
class MatchInput(BaseModel):
    HomeForm: float
    AwayForm: float
    HomeGoalsFor: float
    HomeGoalsAgainst: float
    AwayGoalsFor: float
    AwayGoalsAgainst: float
    DaysSinceHome: float
    DaysSinceAway: float
    ImpH: float
    ImpD: float
    ImpA: float
    H2H_Count: int
    H2H_HomeWinRate: float
    H2H_AwayWinRate: float
    H2H_DrawRate: float
    H2H_HomeGoalsAvg: float
    H2H_AwayGoalsAvg: float

    class Config:
        extra = "ignore"

# ────────────────────────────────────────────────────────────────────────────────
# Globals for loaded models and feature order
# ────────────────────────────────────────────────────────────────────────────────
models = {}
feature_order = []

# ────────────────────────────────────────────────────────────────────────────────
# Load models once at startup
# ────────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def load_models():
    global models, feature_order
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for name in ["lr_baseline", "rf_model", "xgb_model"]:
        path = os.path.join(base_dir, "models", f"{name}.pkl")
        models[name] = joblib.load(path)
    feature_order = list(models["lr_baseline"].feature_names_in_)


# ────────────────────────────────────────────────────────────────────────────────
# Ensemble prediction endpoint
# ────────────────────────────────────────────────────────────────────────────────
@app.post("/predict-ensemble")
def predict_ensemble(match: MatchInput):
    # 1. Build DataFrame from request
    df_in = pd.DataFrame([match.dict()])

    # 2. Reindex to the exact feature names used during training
    X = df_in.reindex(columns=feature_order, fill_value=0)

    # 3. Validate columns
    missing = set(feature_order) - set(X.columns)
    extra   = set(X.columns) - set(feature_order)
    if missing or extra:
        details = []
        if missing:
            details.append(f"Missing features: {sorted(missing)}")
        if extra:
            details.append(f"Unexpected features: {sorted(extra)}")
        raise HTTPException(status_code=400, detail="; ".join(details))

    # 4. Predict probabilities
    try:
        probs = {
            name: mdl.predict_proba(X)[:, 1][0]
            for name, mdl in models.items()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 5. Compute ensemble as simple average
    ensemble_prob = sum(probs.values()) / len(probs)

    return {
        "individual_probs": probs,
        "ensemble_prob": ensemble_prob
    }
