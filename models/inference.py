import joblib
import pandas as pd
import os

class ModelInference:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_path)

    def predict(self, data: pd.DataFrame):
        # Sanitize column names to match training format
        data.columns = [col.replace(" ", "_") for col in data.columns]
        return self.model.predict(data)

if __name__ == "__main__":
    # Example usage
    MODEL_PATH = "models/wine_quality_rf.joblib"
    try:
        inference = ModelInference(MODEL_PATH)
        sample_data = pd.DataFrame([{
            "fixed_acidity": 7.4, "volatile_acidity": 0.70, "citric_acid": 0.00,
            "residual_sugar": 1.9, "chlorides": 0.076, "free_sulfur_dioxide": 11.0,
            "total_sulfur_dioxide": 34.0, "density": 0.9978, "pH": 3.51,
            "sulphates": 0.56, "alcohol": 9.4
        }])
        prediction = inference.predict(sample_data)
        print(f"Prediction: {prediction}")
    except Exception as e:
        print(f"Error: {e}")
