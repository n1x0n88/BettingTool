import os
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# 1. Paths
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJ_ROOT, "data", "processed", "features.csv")

def main():
    # 2. Load and sort data
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 3. Select pre-match features and target
    features = [
        "HomeForm", "AwayForm",
        "HomeGoalsFor", "HomeGoalsAgainst",
        "AwayGoalsFor", "AwayGoalsAgainst"
    ]
    X = df[features].fillna(0)
    y = df["HomeWin"]

    # 4. Define a time-series splitter
    tscv = TimeSeriesSplit(n_splits=5)

    # 5. Run cross-validation (negative log loss)
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    scores = cross_val_score(
        model, X, y,
        cv=tscv,
        scoring="neg_log_loss",
        n_jobs=-1
    )

    # 6. Report mean log loss
    mean_log_loss = -scores.mean()
    print(f"TimeSeries CV Log Loss (5 folds): {mean_log_loss:.4f}")

if __name__ == "__main__":
    main()
