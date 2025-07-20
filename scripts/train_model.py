import os
import pandas as pd
import joblib
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Paths
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJ_ROOT, "data", "processed", "features.csv")
MODEL_DIR = os.path.join(PROJ_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)

    # 2. Select only pre-match features (no GoalDiff or actual result columns)
    features = [
        "HomeForm",
        "AwayForm",
        "HomeGoalsFor",
        "HomeGoalsAgainst",
        "AwayGoalsFor",
        "AwayGoalsAgainst",
        "DaysSinceHome",
        "DaysSinceAway",
        "ImpH",
        "ImpD",
        "ImpA"
    ]
    X = df[features].fillna(0)
    y = df["HomeWin"]  # target: did the home team win?

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4. Baseline Logistic Regression
    lr = LogisticRegression(max_iter=1000, solver="liblinear")
    lr.fit(X_train, y_train)

    prob_test = lr.predict_proba(X_test)[:, 1]
    print("Baseline Metrics:")
    print(f"  Log Loss   : {log_loss(y_test, prob_test):.4f}")
    print(f"  Brier Score: {brier_score_loss(y_test, prob_test):.4e}")
    print(f"  ROC AUC    : {roc_auc_score(y_test, prob_test):.4f}")

    # Save baseline model
    joblib.dump(lr, os.path.join(MODEL_DIR, "lr_baseline.pkl"))
    print("✅ Saved baseline model to models/lr_baseline.pkl")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Probability Calibration
    # ─────────────────────────────────────────────────────────────────────────

    # 5a. Platt Scaling (sigmoid)
    platt = CalibratedClassifierCV(lr, cv=5, method="sigmoid")
    platt.fit(X_train, y_train)
    p_platt = platt.predict_proba(X_test)[:, 1]
    print("\nPlatt Calibration Metrics:")
    print(f"  Log Loss   : {log_loss(y_test, p_platt):.4f}")
    print(f"  Brier Score: {brier_score_loss(y_test, p_platt):.4e}")
    joblib.dump(platt, os.path.join(MODEL_DIR, "lr_platt.pkl"))
    print("✅ Saved Platt‐calibrated model to models/lr_platt.pkl")

    # 5b. Isotonic Calibration
    iso = CalibratedClassifierCV(lr, cv=5, method="isotonic")
    iso.fit(X_train, y_train)
    p_iso = iso.predict_proba(X_test)[:, 1]
    print("\nIsotonic Calibration Metrics:")
    print(f"  Log Loss   : {log_loss(y_test, p_iso):.4f}")
    print(f"  Brier Score: {brier_score_loss(y_test, p_iso):.4e}")
    joblib.dump(iso, os.path.join(MODEL_DIR, "lr_iso.pkl"))
    print("✅ Saved isotonic‐calibrated model to models/lr_iso.pkl")

    # ─────────────────────────────────────────────────────────────────────────────
    # 6. Random Forest Baseline
    # ─────────────────────────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    p_rf = rf.predict_proba(X_test)[:, 1]

    print("\nRandom Forest Metrics:")
    print(f"  Log Loss : {log_loss(y_test, p_rf):.4f}")
    print(f"  ROC AUC  : {roc_auc_score(y_test, p_rf):.4f}")

    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_baseline.pkl"))
    print("✅ Saved RF baseline model to models/rf_baseline.pkl")


    # ─────────────────────────────────────────────────────────────────────────────
    # 7. Hyperparameter Tuning (Randomized Search)
    # ─────────────────────────────────────────────────────────────────────────────
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 8, 12],
        "min_samples_leaf": [1, 5, 10, 20],
        "max_features": ["sqrt", "log2", None]
    }

    rs = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_dist,
        n_iter=20,
        scoring="neg_log_loss",
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    rs.fit(X_train, y_train)
    best_rf = rs.best_estimator_
    p_best_rf = best_rf.predict_proba(X_test)[:, 1]

    print("\nTuned RF Metrics:")
    print(f"  Best Params    : {rs.best_params_}")
    print(f"  Log Loss       : {log_loss(y_test, p_best_rf):.4f}")
    print(f"  ROC AUC        : {roc_auc_score(y_test, p_best_rf):.4f}")

    joblib.dump(best_rf, os.path.join(MODEL_DIR, "rf_tuned.pkl"))
    print("✅ Saved tuned RF model to models/rf_tuned.pkl")

    # ─────────────────────────────────────────────────────────────────────────────
    # 8. XGBoost Baseline
    # ─────────────────────────────────────────────────────────────────────────────
    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb_clf.fit(X_train, y_train)
    p_xgb = xgb_clf.predict_proba(X_test)[:, 1]

    print("\nXGBoost Metrics:")
    print(f"  Log Loss : {log_loss(y_test, p_xgb):.4f}")
    print(f"  ROC AUC  : {roc_auc_score(y_test, p_xgb):.4f}")

    joblib.dump(xgb_clf, os.path.join(MODEL_DIR, "xgb_baseline.pkl"))
    print("✅ Saved XGBoost model to models/xgb_baseline.pkl")

if __name__ == "__main__":
    main()
