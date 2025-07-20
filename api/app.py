# api/app.py

import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Football Win Probability API")

# ────────────────────────────────────────────────────────────────────────────────
# Feature list (must match training)
# ────────────────────────────────────────────────────────────────────────────────
FEATURES = [
    "HomeForm","AwayForm","HomeGoalsFor","HomeGoalsAgainst",
    "AwayGoalsFor","AwayGoalsAgainst","DaysSinceHome","DaysSinceAway",
    "ImpH","ImpD","ImpA",
    "H2H_Count","H2H_HomeWinRate","H2H_AwayWinRate",
    "H2H_DrawRate","H2H_HomeGoalsAvg","H2H_AwayGoalsAvg"
]

# ────────────────────────────────────────────────────────────────────────────────
# Load models on startup
# ────────────────────────────────────────────────────────────────────────────────
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_names = ["lr_baseline", "rf_model", "xgb_model"]
models = {}
for name in model_names:
    path = os.path.join(BASE, "models", f"{name}.pkl")
    models[name] = joblib.load(path)

# ────────────────────────────────────────────────────────────────────────────────
# Pydantic model for POST body
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

# ────────────────────────────────────────────────────────────────────────────────
# Ensemble prediction endpoint
# ────────────────────────────────────────────────────────────────────────────────
@app.post("/predict-ensemble")
def predict_ensemble(match: MatchInput):
    payload = match.dict()
    X = pd.DataFrame([{f: payload[f] for f in FEATURES}])

    try:
        probs = {name: mdl.predict_proba(X)[:, 1][0] for name, mdl in models.items()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    ensemble_prob = sum(probs.values()) / len(probs)

    return {"individual_probs": probs, "ensemble_prob": ensemble_prob}
