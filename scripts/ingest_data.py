import os
import requests

# 1. Configure seasons and league codes
SEASONS = ["0001","0102","0203","0304","0405","0506","0607","0708","0809","0910","1011","1112","1213","1314","1415","1516","1617","1718","1819","1920","2021","2122","2223","2324", "2425"]
LEAGUES = ["E0", "E1", "E2", "E3", "SC0", "SC1", "SC2", "SC3", "D1", "D2", "I1", "I2", "SP1", "SP2", "F1", "F2", "N1", "B1", "P1", "T1", "G1", "A1", "S1", "R1", "U1"]

BASE_URL = "https://www.football-data.co.uk/mmz4281"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# 2. Ensure the raw data directory exists
os.makedirs(RAW_DIR, exist_ok=True)

# 3. Download each CSV and save to disk
for season in SEASONS:
    for league in LEAGUES:
        url = f"{BASE_URL}/{season}/{league}.csv"
        dest = os.path.join(RAW_DIR, f"{season}_{league}.csv")
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.write(resp.content)
            print(f"✔ Downloaded {season}_{league}.csv")
        except Exception as e:
            print(f"✘ Failed {url}: {e}")
