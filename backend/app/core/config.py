"""
AuroraML Core Configuration
Loads settings from environment variables / .env file.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # App
    APP_NAME: str = "AuroraML"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "sqlite:///./auroraml.db"

    # JWT
    JWT_SECRET_KEY: str = "auroraml-super-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days for MVP

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Storage Mode
    USE_MINIO: bool = False
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET_DATASETS: str = "auroraml-datasets"
    MINIO_BUCKET_MODELS: str = "auroraml-models"
    MINIO_USE_SSL: bool = False

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"

    # Celery
    USE_CELERY: bool = False
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    ALLOW_INPROCESS_JOBS: bool = True
    REQUIRE_DURABLE_WORKERS: bool = False

    # Storage (local fallback when MinIO is unavailable)
    LOCAL_STORAGE_PATH: str = "./storage"

    # LLM / Aurora Copilot
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4.1"
    AURORA_LLM_ENABLED: bool = True

    # Dataset Type Classifier
    DATASET_TYPE_MODEL_PATH: str = "./backend/app/models/dataset_type_model.json"

    # Monitoring
    MONITORING_SNAPSHOT_RESOLUTION_MINUTES: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
