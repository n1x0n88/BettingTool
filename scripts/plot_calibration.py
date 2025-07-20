import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# 1. Paths
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJ_ROOT, "data", "processed", "features.csv")
MODEL_DIR = os.path.join(PROJ_ROOT, "models")

# 2. Load data
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

features = [
    "HomeForm", "AwayForm",
    "HomeGoalsFor", "HomeGoalsAgainst",
    "AwayGoalsFor", "AwayGoalsAgainst"
]
X = df[features].fillna(0)
y = df["HomeWin"]

# 3. Load models
lr    = joblib.load(os.path.join(MODEL_DIR, "lr_baseline.pkl"))
platt = joblib.load(os.path.join(MODEL_DIR, "lr_platt.pkl"))
iso   = joblib.load(os.path.join(MODEL_DIR, "lr_iso.pkl"))

# 4. Predict probabilities
probs = {
    "LR":        lr.predict_proba(X)[:, 1],
    "Platt-LR":  platt.predict_proba(X)[:, 1],
    "Isotonic-LR": iso.predict_proba(X)[:, 1]
}

# 5. Compute calibration curves
plt.figure(figsize=(8, 6))
for name, p in probs.items():
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10)
    plt.plot(mean_pred, frac_pos, marker="o", label=name)

# 6. Perfect calibration reference
plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")

plt.title("Calibration Curves")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 7. Show or save
plt.show()
# plt.savefig("calibration_curve.png", dpi=300)
