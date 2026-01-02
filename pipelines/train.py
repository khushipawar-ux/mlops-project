import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import joblib

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_model(train_path: str, test_path: str, model_path: str):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    mlflow.set_experiment("Wine_Quality_Restructured")

    n_estimators = 100
    max_depth = 10

    with mlflow.start_run():
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(train_x, train_y.values.ravel())

        predicted_qualities = rf.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print(f"Random Forest model (n_estimators={n_estimators}, max_depth={max_depth}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Log and register model
        mlflow.sklearn.log_model(rf, "model", registered_model_name="WineQualityModelV2")
        
        # Save model locally for inference component
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(rf, model_path)
        print(f"Model saved locally to {model_path}")

if __name__ == "__main__":
    TRAIN_PATH = "data/processed/train.csv"
    TEST_PATH = "data/processed/test.csv"
    MODEL_PATH = "models/wine_quality_rf.joblib"
    train_model(TRAIN_PATH, TEST_PATH, MODEL_PATH)
