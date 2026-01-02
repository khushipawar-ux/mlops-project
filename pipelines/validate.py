import pandas as pd
import os

def validate_data(input_path: str):
    """
    Validates the ingested data.
    """
    if not os.path.exists(input_path):
        print(f"File {input_path} not found.")
        return False
    
    data = pd.read_csv(input_path)
    
    # Check for null values
    if data.isnull().values.any():
        print("Data contains null values.")
        return False
    
    # Check for expected columns
    expected_columns = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol", "quality"
    ]
    
    if not all(col in data.columns for col in expected_columns):
        print("Missing expected columns.")
        return False
    
    print("Data validation successful.")
    return True

if __name__ == "__main__":
    RAW_DATA_PATH = "data/raw/winequality-red.csv"
    validate_data(RAW_DATA_PATH)
