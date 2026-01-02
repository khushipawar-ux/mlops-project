import pandas as pd
import os
from sklearn.model_selection import train_test_split

def transform_data(input_path: str, processed_dir: str):
    """
    Transforms data and creates train/test sets.
    """
    if not os.path.exists(input_path):
        print(f"File {input_path} not found.")
        return
    
    data = pd.read_csv(input_path)
    
    # Simple transformation: renaming columns to be snake_case (standardizing)
    data.columns = [col.replace(" ", "_") for col in data.columns]
    
    os.makedirs(processed_dir, exist_ok=True)
    
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    
    train.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    
    print(f"Transformed data saved to {processed_dir}")

if __name__ == "__main__":
    RAW_DATA_PATH = "data/raw/winequality-red.csv"
    PROCESSED_DIR = "data/processed"
    transform_data(RAW_DATA_PATH, PROCESSED_DIR)
