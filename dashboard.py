# dashboard.py

import os
import math
import streamlit as st
import pandas as pd
import requests

# Use /predict-ensemble by default
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict-ensemble")

st.set_page_config(page_title="Football Win Probability", layout="wide")
st.title("⚽ Football Win Probability Dashboard")

@st.cache_data
def load_data(path="data/processed/features_h2h.csv"):
    """Load H2H-enriched features, including raw match stats and head-to-head columns."""
    return pd.read_csv(path, parse_dates=["Date"])

# 1. Load data and team list
df = load_data()
teams = sorted(df["HomeTeam"].unique())

# 2. Sidebar inputs: home, away, date
home = st.sidebar.selectbox("Home Team", teams, key="home_team")
away = st.sidebar.selectbox("Away Team", [t for t in teams if t != home], key="away_team")
match_date = st.sidebar.date_input(
    "Match Date",
    value=df["Date"].max().date(),
    key="match_date"
)

# 3. Filter all matches strictly before the selected date
game_dt = pd.to_datetime(match_date)
df_pre = df[df["Date"] < game_dt]

# Validate enough history
if df_pre.empty:
    st.error("No data exists before that date. Pick an earlier date.")
    st.stop()

df_home = df_pre[df_pre["HomeTeam"] == home]
df_away = df_pre[df_pre["AwayTeam"] == away]
if df_home.empty or df_away.empty:
    st.error("No historical matches for one of the teams before that date. Adjust your choices.")
    st.stop()

# 4. Extract last rolling features for each team
home_stats = df_home.sort_values("Date").iloc[-1].fillna(0.0)
away_stats = df_away.sort_values("Date").iloc[-1].fillna(0.0)

# 5. Dynamically compute head-to-head history
mask_h2h = (
    ((df_pre.HomeTeam == home) & (df_pre.AwayTeam == away)) |
    ((df_pre.HomeTeam == away) & (df_pre.AwayTeam == home))
)
h2h_hist = df_pre.loc[mask_h2h]
n = len(h2h_hist)

if n > 0:
    home_wins    = ((h2h_hist.HomeTeam == home) & (h2h_hist.FTR == "H")).sum()
    away_wins    = ((h2h_hist.AwayTeam == home) & (h2h_hist.FTR == "A")).sum()
    draws        = (h2h_hist.FTR == "D").sum()
    goals_for    = h2h_hist.apply(lambda r: r.FTHG if r.HomeTeam == home else r.FTAG, axis=1).sum()
    goals_against= h2h_hist.apply(lambda r: r.FTAG if r.HomeTeam == home else r.FTHG, axis=1).sum()

    H2H_Count          = n
    H2H_HomeWinRate    = home_wins / n
    H2H_AwayWinRate    = away_wins / n
    H2H_DrawRate       = draws     / n
    H2H_HomeGoalsAvg   = goals_for    / n
    H2H_AwayGoalsAvg   = goals_against/ n
else:
    # No prior meetings
    H2H_Count = H2H_HomeWinRate = H2H_AwayWinRate = 0.0
    H2H_DrawRate = H2H_HomeGoalsAvg = H2H_AwayGoalsAvg = 0.0

# 6. Build the full 17-field payload
payload = {
    "HomeForm":         float(home_stats.HomeForm),
    "AwayForm":         float(away_stats.AwayForm),
    "HomeGoalsFor":     float(home_stats.HomeGoalsFor),
    "HomeGoalsAgainst": float(home_stats.HomeGoalsAgainst),
    "AwayGoalsFor":     float(away_stats.AwayGoalsFor),
    "AwayGoalsAgainst": float(away_stats.AwayGoalsAgainst),
    "DaysSinceHome":    float(home_stats.DaysSinceHome),
    "DaysSinceAway":    float(away_stats.DaysSinceAway),
    "ImpH":             float(home_stats.ImpH),
    "ImpD":             float(home_stats.ImpD),
    "ImpA":             float(home_stats.ImpA),
    "H2H_Count":        int(H2H_Count),
    "H2H_HomeWinRate":  float(H2H_HomeWinRate),
    "H2H_AwayWinRate":  float(H2H_AwayWinRate),
    "H2H_DrawRate":     float(H2H_DrawRate),
    "H2H_HomeGoalsAvg": float(H2H_HomeGoalsAvg),
    "H2H_AwayGoalsAvg": float(H2H_AwayGoalsAvg)
}

# Sanitize any NaNs (shouldn’t be any)
for k, v in payload.items():
    if isinstance(v, float) and math.isnan(v):
        payload[k] = 0.0

# 7. Display inputs
with st.expander("Input Features"):
    st.write("Match Date:", match_date)
    st.write(pd.DataFrame([payload]).T.rename(columns={0: "Value"}))

# 8. Call API on button click
if st.sidebar.button("Get Prediction", key="predict_btn"):
    st.write("🔍 Payload:", payload)
    try:
        resp = requests.post(API_URL, json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        # Our endpoint returns 'ensemble_prob'
        if "ensemble_prob" in data:
            st.metric("Home Win Probability", f"{data['ensemble_prob']:.1%}")
            st.write("Individual model probabilities:", data["individual_probs"])
        else:
            st.error(f"Unexpected response format: {data}")

    except requests.exceptions.HTTPError:
        try:
            err = resp.json()
        except:
            err = resp.text
        st.error(f"API returned {resp.status_code} error:\n{err}")
    except Exception as e:
        st.error(f"Request failed: {e}")
