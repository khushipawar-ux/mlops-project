import pytest
import os
from src.ingestion import ingest_data

def test_ingestion():
    # Test if data ingestion creates the file
    ingest_data()
    assert os.path.exists("data/raw_data.csv")

def test_data_load():
    import pandas as pd
    data = pd.read_csv("data/raw_data.csv")
    assert not data.empty
    assert "quality" in data.columns
