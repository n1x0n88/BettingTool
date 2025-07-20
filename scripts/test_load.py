import pandas as pd

def main():
    # pick one of your downloaded CSVs
    path = "data/raw/2324_E0.csv"
    df = pd.read_csv(path)

    # report what we’ve got
    print("✅ Columns:", df.columns.tolist())
    print("✅ Number of rows:", len(df))
    print("\n🔍 First 5 rows:\n", df.head())

if __name__ == "__main__":
    main()
