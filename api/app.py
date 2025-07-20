# api/app.py

import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Football Win Probability API")

# ────────────────────────────────────────────────────────────────────────────────
# 1. Define the 17 features your ensemble expects
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
        extra = "ignore"  # ignore any unexpected fields in the payload

# ────────────────────────────────────────────────────────────────────────────────
# 2. Load all models once at startup
# ────────────────────────────────────────────────────────────────────────────────
models = {}
feature_order = None

@app.on_event("startup")
def load_models():
    global models, feature_order

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_names = ["lr_baseline", "rf_model", "xgb_model"]

    for name in model_names:
        path = os.path.join(base, "models", f"{name}.pkl")
        models[name] = joblib.load(path)

    # Cache the feature names in the order used during training
    feature_order = list(models["lr_baseline"].feature_names_in_)

# ────────────────────────────────────────────────────────────────────────────────
# 3. Ensemble prediction endpoint
# ────────────────────────────────────────────────────────────────────────────────
@app.post("/predict-ensemble")
def predict_ensemble(match: MatchInput):
    # Turn the incoming JSON into a one-row DataFrame
    df_in = pd.DataFrame([match.dict()])

    # Reindex to ensure exactly the columns the model saw at fit time
    X = df_in.reindex(columns=feature_order, fill_value=0)

    # Sanity check: if something’s still missing or extra, report it
    missing = set(feature_order) - set(X.columns)
    extra   = set(X.columns) - set(feature_order)
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"Missing features: {sorted(missing)}")
        if extra:
            detail.append(f"Unexpected features: {sorted(extra)}")
        raise HTTPException(status_code=400, detail="; ".join(detail))

    # Predict each model’s home-win probability
    try:
        probs = {
            name: mdl.predict_proba(X)[:, 1][0]
            for name, mdl in models.items()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Average them for an ensemble
    ensemble_prob = sum(probs.values()) / len(probs)

    return {"individual_probs": probs, "ensemble_prob": ensemble_prob}
