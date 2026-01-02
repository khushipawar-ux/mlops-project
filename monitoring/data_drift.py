import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
import os

def check_data_drift(reference_path, current_path, output_path="monitoring/data_drift_report.html"):
    if not os.path.exists(reference_path) or not os.path.exists(current_path):
        print("Data files missing for drift detection.")
        return

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    snapshot.save_html(output_path)
    print(f"Data drift report generated at {output_path}")

if __name__ == "__main__":
    # Example logic
    REF = "data/processed/test.csv"
    CUR = "data/processed/train.csv" # Just for testing
    check_data_drift(REF, CUR)
