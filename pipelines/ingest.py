import pandas as pd
import os

def ingest_data(url: str, output_path: str):
    """
    Downloads data from a URL and saves it to the raw data folder.
    """
    try:
        print(f"Ingesting data from {url}...")
        data = pd.read_csv(url, sep=";")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        return data
    except Exception as e:
        print(f"Error during ingestion: {e}")
        return None

if __name__ == "__main__":
    WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    RAW_DATA_PATH = "data/raw/winequality-red.csv"
    ingest_data(WINE_URL, RAW_DATA_PATH)
