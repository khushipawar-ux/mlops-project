# End-to-End ML Pipeline with MLOps

This project implements a complete Machine Learning pipeline with MLOps best practices, including data ingestion, validation, transformation, model training, versioning, serving, and monitoring.

## Folder Structure
- `data/`: Raw and processed data storage.
- `pipelines/`: Modular scripts for the ML pipeline.
- `models/`: Local storage for the latest model artifacts.
- `api/`: FastAPI implementation for model serving.
- `monitoring/`: Scripts for data and model drift detection.
- `.github/workflows/`: CI/CD pipelines for training and deployment.

## Tech Stack
- **MLflow**: Experiment tracking and model registry.
- **FastAPI**: REST API for model serving.
- **Docker & Docker Compose**: Containerization and local orchestration.
- **Github Actions**: CI/CD for automation.
- **Evidently AI**: Monitoring for drift detection.

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
python pipelines/ingest.py
python pipelines/validate.py
python pipelines/transform.py
python pipelines/train.py
```

### 3. Run the API locally
```bash
uvicorn api.main:app --reload
```

### 4. Run with Docker Compose
```bash
docker-compose up --build
# OR if you have Docker V2
docker compose up --build
```
This will start the API at `localhost:8000` and the MLflow UI at `localhost:5000`.

## Monitoring
Run `python monitoring/data_drift.py` to generate a drift report in the `monitoring/` folder.
