from fastapi import FastAPI, HTTPException
from api.schema import WineFeatures, PredictionResponse
from models.inference import ModelInference
import pandas as pd
import os

app = FastAPI(title="Wine Quality MLOps API")

MODEL_PATH = "models/wine_quality_rf.joblib"
inference = None

@app.on_event("startup")
def load_model():
    global inference
    if os.path.exists(MODEL_PATH):
        inference = ModelInference(MODEL_PATH)
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

@app.get("/")
def read_root():
    return {"status": "online", "model": "RandomForest"}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: WineFeatures):
    if inference is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Convert Pydantic model to DataFrame
        data = pd.DataFrame([features.dict()])
        prediction = inference.predict(data)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
