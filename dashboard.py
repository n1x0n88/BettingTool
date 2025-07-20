# dashboard.py

import os
import math
import streamlit as st
import pandas as pd
import requests

# Use environment variable if set (for Docker), else localhost
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(page_title="Football Win Probability", layout="wide")
st.title("⚽ Football Win Probability Dashboard")

@st.cache_data
def load_data(path="data/processed/features.csv"):
    """Load processed features with parsed dates."""
    return pd.read_csv(path, parse_dates=["Date"])

# 1. Load data and team list
df = load_data()
teams = sorted(df["HomeTeam"].unique())

# 2. Sidebar: choose home, away, date
home = st.sidebar.selectbox("Home Team", teams, key="home_team")
away = st.sidebar.selectbox(
    "Away Team",
    [t for t in teams if t != home],
    key="away_team"
)
match_date = st.sidebar.date_input(
    "Match Date",
    value=df["Date"].max().date(),
    key="match_date"
)

# 3. Filter stats before the selected date
game_dt = pd.to_datetime(match_date)
df_pre = df[df["Date"] < game_dt]

df_home = df_pre[df_pre["HomeTeam"] == home]
df_away = df_pre[df_pre["AwayTeam"] == away]

if df_home.empty or df_away.empty:
    st.error("No historical data for those teams before that date. Pick an earlier date.")
    st.stop()

# Get the last available stats and fill any NaNs
home_stats = df_home.sort_values("Date").iloc[-1].fillna(0.0)
away_stats = df_away.sort_values("Date").iloc[-1].fillna(0.0)

# 4. Build the payload
payload = {
    "HomeTeam": home,
    "AwayTeam": away,
    "HomeForm": float(home_stats.HomeForm),
    "AwayForm": float(away_stats.AwayForm),
    "HomeGoalsFor": float(home_stats.HomeGoalsFor),
    "HomeGoalsAgainst": float(home_stats.HomeGoalsAgainst),
    "AwayGoalsFor": float(away_stats.AwayGoalsFor),
    "AwayGoalsAgainst": float(away_stats.AwayGoalsAgainst),
    "DaysSinceHome": float(home_stats.DaysSinceHome),
    "DaysSinceAway": float(away_stats.DaysSinceAway),
    "ImpH": float(home_stats.ImpH),
    "ImpD": float(home_stats.ImpD),
    "ImpA": float(home_stats.ImpA),
}

# 5. Sanitize payload: replace any NaN with 0.0
for k, v in payload.items():
    if isinstance(v, float) and math.isnan(v):
        payload[k] = 0.0

# 6. Show input summary
with st.expander("Input Features"):
    st.write("Match Date:", match_date)
    st.write(pd.DataFrame([payload]).T.rename(columns={0: "Value"}))

# 7. Button to call the API
if st.sidebar.button("Get Prediction", key="get_prediction_button"):
    st.write("🔍 Payload:", payload)
    try:
        resp = requests.post(API_URL, json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        prob = data.get("HomeWin_Prob")
        if prob is None:
            st.error(f"Unexpected response:\n{data}")
        else:
            st.metric("Home Win Probability", f"{prob:.1%}")
    except requests.exceptions.HTTPError:
        # Show API’s error message
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        st.error(f"API returned {resp.status_code} error:\n{err}")
    except Exception as e:
        st.error(f"Request failed: {e}")
