# scripts/train_models.py

import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sys

# ────────────────────────────────────────────────────────────────────────────────
# 1. Paths & features
# ────────────────────────────────────────────────────────────────────────────────
ROLL_CSV   = Path("data/processed/features.csv")
H2H_CSV    = Path("data/processed/features_h2h.csv")
MODELDIR   = Path("models")
MODELDIR.mkdir(exist_ok=True, parents=True)

ROLL_FEATURES = [
    "HomeForm","AwayForm",
    "HomeGoalsFor","HomeGoalsAgainst",
    "AwayGoalsFor","AwayGoalsAgainst",
    "DaysSinceHome","DaysSinceAway",
    "ImpH","ImpD","ImpA",
]

H2H_FEATURES = [
    "H2H_Count","H2H_HomeWinRate",
    "H2H_AwayWinRate","H2H_DrawRate",
    "H2H_HomeGoalsAvg","H2H_AwayGoalsAvg"
]

FEATURES = ROLL_FEATURES + H2H_FEATURES

# ────────────────────────────────────────────────────────────────────────────────
# 2. Load & merge
# ────────────────────────────────────────────────────────────────────────────────
print("Loading rolling features with low_memory=False...")
df_roll = pd.read_csv(ROLL_CSV, parse_dates=["Date"], low_memory=False)

print("Loading head-to-head features...")
df_h2h = pd.read_csv(H2H_CSV, parse_dates=["Date"])[
    ["Date","HomeTeam","AwayTeam"] + H2H_FEATURES
]

print("Merging on Date, HomeTeam, AwayTeam...")
df = pd.merge(
    df_roll, df_h2h,
    on=["Date","HomeTeam","AwayTeam"],
    how="inner"
)
print(f"After merge: {len(df):,} rows, {df.shape[1]} columns")

# ────────────────────────────────────────────────────────────────────────────────
# 3. Identify label column
# ────────────────────────────────────────────────────────────────────────────────
for candidate in ("FTR","Result","HTR"):
    if candidate in df.columns:
        label_col = candidate
        break
else:
    print("ERROR: No full-time result column found in merged DataFrame.")
    print("Available columns:", df.columns.tolist())
    sys.exit(1)

print(f"Using '{label_col}' as the target label.")

# ────────────────────────────────────────────────────────────────────────────────
# 4. Prepare X and y
# ────────────────────────────────────────────────────────────────────────────────
X = df[FEATURES].fillna(0)

# home-win label: 1 if full-time result == 'H'
y = (df[label_col] == "H").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ────────────────────────────────────────────────────────────────────────────────
# 5. Train & save models
# ────────────────────────────────────────────────────────────────────────────────
models = {
    "lr_baseline": LogisticRegression(max_iter=1000, n_jobs=-1),
    "rf_model":     RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "xgb_model":    xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=-1)
}

for name, mdl in models.items():
    print(f"Training {name}...")
    mdl.fit(X_train, y_train)
    preds = mdl.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"  → {name} AUC: {auc:.4f}")
    joblib.dump(mdl, MODELDIR / f"{name}.pkl")
    print(f"Saved {name}.pkl")

print("✅ All models trained and saved!")
