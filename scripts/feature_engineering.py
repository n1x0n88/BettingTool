import os
import pandas as pd

def safe_read_csv(path):
    """
    Try to read CSV with utf-8, then latin-1.
    On ParserError, fallback to python engine skipping bad lines.
    """
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            print(f"⚠️ {os.path.basename(path)} not {enc}, retrying...")
        except pd.errors.ParserError:
            print(f"⚠️ ParserError on {os.path.basename(path)}, falling back to python engine")
            break

    for enc in ("latin-1", "utf-8"):
        try:
            return pd.read_csv(
                path,
                encoding=enc,
                engine="python",
                on_bad_lines="skip",
                skip_blank_lines=True
            )
        except Exception as e:
            print(f"⚠️ python-engine & {enc} failed for {os.path.basename(path)}: {e}")

    raise ValueError(f"Unable to read {path} with any engine/encoding combo")


# 1. Directory setup
SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")
PROC_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)


# 2. Load & concatenate raw CSVs
def load_raw(seasons, leagues):
    dfs = []
    for season in seasons:
        for league in leagues:
            fname = f"{season}_{league}.csv"
            path  = os.path.join(RAW_DIR, fname)
            if os.path.isfile(path):
                df = safe_read_csv(path)
                df["Season"] = season
                df["League"] = league
                dfs.append(df)
            else:
                print(f"⚠️ Missing file, skipping: {fname}")
    if not dfs:
        raise RuntimeError("No data loaded. Check seasons & leagues lists.")
    return pd.concat(dfs, ignore_index=True)


# 3. Preprocessing + implied probabilities + rest days
def preprocess(df):
    # a) Parse dates
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    # b) Drop invalid rows
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    # c) Standardize columns
    df = df.rename(columns={
        "FTHG": "HomeGoals",
        "FTAG": "AwayGoals",
        "FTR":  "Result"   # H, D, A
    })

    # d) Implied probabilities from average odds
    #    BbAvH = avg home-win odds, BbAvD = draw, BbAvA = away-win
    df["ImpH_raw"] = 1 / df["BbAvH"]
    df["ImpD_raw"] = 1 / df["BbAvD"]
    df["ImpA_raw"] = 1 / df["BbAvA"]
    sum_imp = df["ImpH_raw"] + df["ImpD_raw"] + df["ImpA_raw"]
    df["ImpH"] = df["ImpH_raw"] / sum_imp
    df["ImpD"] = df["ImpD_raw"] / sum_imp
    df["ImpA"] = df["ImpA_raw"] / sum_imp
    df = df.drop(columns=["ImpH_raw", "ImpD_raw", "ImpA_raw"])

    # e) Rest days: days since each team's last match
    df = df.sort_values("Date")
    df["DaysSinceHome"] = (
        df.groupby("HomeTeam")["Date"]
          .diff()
          .dt.days
          .fillna(7)
    )
    df["DaysSinceAway"] = (
        df.groupby("AwayTeam")["Date"]
          .diff()
          .dt.days
          .fillna(7)
    )

    return df


# 4. Basic match‐level features
def create_basic_features(df):
    df["GoalDiff"] = df["HomeGoals"] - df["AwayGoals"]
    df["HomeWin"]  = (df["Result"] == "H").astype(int)
    df["Draw"]     = (df["Result"] == "D").astype(int)
    df["AwayWin"]  = (df["Result"] == "A").astype(int)
    return df


# 5. Rolling‐window features
def add_rolling_features(df, window=5):
    df = df.sort_values("Date").reset_index(drop=True)

    def home_form(g):
        pts = g["HomeWin"]*3 + g["Draw"]
        return pts.shift().rolling(window).mean().rename("HomeForm")

    def away_form(g):
        pts = g["AwayWin"]*3 + g["Draw"]
        return pts.shift().rolling(window).mean().rename("AwayForm")

    df["HomeForm"] = df.groupby("HomeTeam", group_keys=False).apply(home_form)
    df["AwayForm"] = df.groupby("AwayTeam", group_keys=False).apply(away_form)

    df["HomeGoalsFor"]     = df.groupby("HomeTeam")["HomeGoals"]\
                                  .shift().rolling(window).mean().reset_index(drop=True)
    df["HomeGoalsAgainst"] = df.groupby("HomeTeam")["AwayGoals"]\
                                  .shift().rolling(window).mean().reset_index(drop=True)
    df["AwayGoalsFor"]     = df.groupby("AwayTeam")["AwayGoals"]\
                                  .shift().rolling(window).mean().reset_index(drop=True)
    df["AwayGoalsAgainst"] = df.groupby("AwayTeam")["HomeGoals"]\
                                  .shift().rolling(window).mean().reset_index(drop=True)

    return df


def main():
    # 6. Seasons & leagues to process
    seasons = [
        "0001","0102","0203","0304","0405","0506","0607","0708",
        "0809","0910","1011","1112","1213","1314","1415","1516",
        "1617","1718","1819","1920","2021","2122","2223","2324","2425"
    ]
    leagues = [
        "E0","E1","E2","E3","SC0","SC1","SC2","SC3","D1","D2",
        "I1","I2","SP1","SP2","F1","F2","N1","B1","P1","T1",
        "G1","A1","S1","R1","U1"
    ]

    raw_df      = load_raw(seasons, leagues)
    clean_df    = preprocess(raw_df)
    basic_df    = create_basic_features(clean_df)
    featured_df = add_rolling_features(basic_df)

    out_path = os.path.join(PROC_DIR, "features.csv")
    featured_df.to_csv(out_path, index=False)
    print(f"✅ Saved processed features to {out_path}")


if __name__ == "__main__":
    main()
