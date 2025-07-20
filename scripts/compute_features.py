# scripts/compute_features.py

import warnings
import pandas as pd
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────────
# Silence the “Could not infer format” warning
# ────────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings(
    "ignore",
    message="Could not infer format, so each element will be parsed individually"
)

# ────────────────────────────────────────────────────────────────────────────────
# 1. Paths & required columns
# ────────────────────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
OUT_CSV       = Path("data/processed/features_h2h.csv")
REQUIRED_COLS = {"Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"}

files = sorted(RAW_DIR.glob("*.csv"))
if not files:
    raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

df_parts = []

# ────────────────────────────────────────────────────────────────────────────────
# 2. Loop through each file
# ────────────────────────────────────────────────────────────────────────────────
for fp in files:
    # 2a. Read header with encoding fallback
    try:
        cols = pd.read_csv(fp, nrows=0, encoding="utf-8").columns.tolist()
    except UnicodeDecodeError:
        cols = pd.read_csv(fp, nrows=0, encoding="latin1").columns.tolist()

    missing = REQUIRED_COLS - set(cols)
    if missing:
        print(f"Skipping {fp.name}: missing columns {sorted(missing)}")
        continue

    # 2b. Read only needed cols, skip bad lines, with encoding fallback
    read_kwargs = dict(usecols=list(REQUIRED_COLS), on_bad_lines="skip")
    try:
        part = pd.read_csv(fp, encoding="utf-8", **read_kwargs)
    except UnicodeDecodeError:
        part = pd.read_csv(fp, encoding="latin1", **read_kwargs)

    # 2c. Parse Date (adjust dayfirst=True if needed)
    part["Date"] = pd.to_datetime(
        part["Date"],
        dayfirst=False,
        errors="coerce"
    )
    bad = part["Date"].isna().sum()
    if bad:
        print(f"  · Dropping {bad} invalid dates in {fp.name}")
        part = part.dropna(subset=["Date"])

    df_parts.append(part)

# ────────────────────────────────────────────────────────────────────────────────
# 3. Concatenate
# ────────────────────────────────────────────────────────────────────────────────
df = pd.concat(df_parts, ignore_index=True)
print(f"Loaded {len(df_parts)} files → {len(df)} total matches")

# ────────────────────────────────────────────────────────────────────────────────
# 4. Compute H2H features
# ────────────────────────────────────────────────────────────────────────────────
def compute_h2h(df):
    df = df.sort_values("Date").reset_index(drop=True)
    for col in [
        "H2H_Count", "H2H_HomeWinRate", "H2H_AwayWinRate",
        "H2H_DrawRate", "H2H_HomeGoalsAvg", "H2H_AwayGoalsAvg"
    ]:
        df[col] = 0.0

    for idx, row in df.iterrows():
        home, away, date = row.HomeTeam, row.AwayTeam, row.Date
        mask = (
            ((df.HomeTeam == home) & (df.AwayTeam == away)) |
            ((df.HomeTeam == away) & (df.AwayTeam == home))
        ) & (df.Date < date)
        hist = df.loc[mask]
        n = len(hist)
        if n == 0:
            continue

        home_wins = ((hist.HomeTeam == home) & (hist.FTR == "H")).sum()
        away_wins = ((hist.AwayTeam == home) & (hist.FTR == "A")).sum()
        draws     = (hist.FTR == "D").sum()

        goals_for = hist.apply(
            lambda r: r.FTHG if r.HomeTeam == home else r.FTAG, axis=1
        ).sum()
        goals_against = hist.apply(
            lambda r: r.FTAG if r.HomeTeam == home else r.FTHG, axis=1
        ).sum()

        df.at[idx, "H2H_Count"]        = n
        df.at[idx, "H2H_HomeWinRate"]  = home_wins / n
        df.at[idx, "H2H_AwayWinRate"]  = away_wins / n
        df.at[idx, "H2H_DrawRate"]     = draws / n
        df.at[idx, "H2H_HomeGoalsAvg"] = goals_for / n
        df.at[idx, "H2H_AwayGoalsAvg"] = goals_against / n

    return df

df = compute_h2h(df)
print("Computed head-to-head features.")

# ────────────────────────────────────────────────────────────────────────────────
# 5. Save enriched file
# ────────────────────────────────────────────────────────────────────────────────
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"Wrote enriched features to {OUT_CSV}")
