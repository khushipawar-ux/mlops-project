import pytest
import os
import pandas as pd
from pipelines.ingest import ingest_data
from pipelines.validate import validate_data

# Constants for testing
WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
TEST_RAW_DATA_PATH = "data/raw/winequality-red.csv"

def test_ingestion():
    """Test if data ingestion correctly downloads and saves data."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(TEST_RAW_DATA_PATH), exist_ok=True)
    
    data = ingest_data(WINE_URL, TEST_RAW_DATA_PATH)
    assert data is not None
    assert os.path.exists(TEST_RAW_DATA_PATH)

def test_data_validation():
    """Test if data validation correctly identifies valid data."""
    assert os.path.exists(TEST_RAW_DATA_PATH)
    is_valid = validate_data(TEST_RAW_DATA_PATH)
    assert is_valid is True

def test_data_load():
    """Test if the saved data can be loaded and has expected structure."""
    data = pd.read_csv(TEST_RAW_DATA_PATH)
    assert not data.empty
    # The columns in the ingested CSV are already normalized by pandas if sep is correct
    # Actually ingest.py saves it with the original column names (with spaces)
    assert "quality" in data.columns
