# AuroraML

> Production-Grade Automated Machine Learning Platform

AuroraML is an end-to-end automated machine learning platform that enables businesses to build, deploy, and monitor AI models without data science expertise.

## Features

- **Automated ML Pipeline** — Data ingestion, cleaning, feature engineering, model training, and hyperparameter tuning
- **Multi-Framework** — Scikit-learn, XGBoost, LightGBM (with TensorFlow/PyTorch coming soon)
- **AutoML** — Optuna-powered hyperparameter optimization with cross-validation
- **REST API** — FastAPI with JWT authentication, Swagger docs
- **Model Management** — Version, deploy, monitor, and serve trained models
- **Drift Detection** — Built-in PSI-based data drift monitoring

## Quick Start

### 1. Set up environment

```bash
cd auroraML
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start infrastructure (optional — for PostgreSQL/Redis/MinIO)

```bash
docker-compose up -d
```

### 3. Run the API server

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### 4. Open docs

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

## Project Structure

```
auroraML/
├── backend/
│   └── app/
│       ├── api/endpoints/     # FastAPI route handlers
│       ├── core/              # Config, database, security
│       ├── models/            # SQLAlchemy ORM models
│       ├── schemas/           # Pydantic request/response models
│       ├── services/          # Business logic (training, deployment, monitoring)
│       ├── tasks/             # Celery background tasks
│       └── main.py            # Application entry point
├── config.yaml                # Pipeline configuration
├── docker-compose.yml         # Local dev services
├── Dockerfile                 # Production container
├── requirements.txt           # Python dependencies
└── .env                       # Environment variables
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Register new user |
| POST | `/api/v1/auth/login` | Login (returns JWT) |
| POST | `/api/v1/projects` | Create ML project |
| GET | `/api/v1/projects` | List projects |
| POST | `/api/v1/projects/{id}/datasets` | Upload dataset |
| POST | `/api/v1/projects/{id}/jobs/train` | Start training job |
| GET | `/api/v1/jobs/{id}` | Get job status/results |
| POST | `/api/v1/models/{id}/deploy` | Deploy model |
| POST | `/api/v1/predict` | Make predictions |

## License

MIT
