"""ML Model SQLAlchemy Model."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Text, DateTime, ForeignKey, JSON, Float, Boolean, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


class MLModel(Base):
    __tablename__ = "ml_models"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    # Model info
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_type: Mapped[str] = mapped_column(String(100), nullable=False)  # random_forest, xgboost, etc
    framework: Mapped[str] = mapped_column(String(50), nullable=False)  # sklearn, xgboost, lightgbm
    version: Mapped[int] = mapped_column(Integer, default=1)
    # Storage
    file_path: Mapped[str] = mapped_column(Text, nullable=True)  # S3/MinIO path
    artifact_uri: Mapped[str] = mapped_column(Text, nullable=True)  # MLflow artifact URI
    # Performance
    metrics: Mapped[dict] = mapped_column(JSON, nullable=True)
    hyperparameters: Mapped[dict] = mapped_column(JSON, nullable=True)
    feature_importance: Mapped[dict] = mapped_column(JSON, nullable=True)
    # Deployment
    is_deployed: Mapped[bool] = mapped_column(Boolean, default=False)
    deployment_stage: Mapped[str] = mapped_column(
        String(50), nullable=True
    )  # staging, production, archived
    endpoint_url: Mapped[str] = mapped_column(Text, nullable=True)
    # Tracking
    mlflow_model_uri: Mapped[str] = mapped_column(Text, nullable=True)
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    deployed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    job = relationship("Job", back_populates="ml_models")
    project = relationship("Project", back_populates="ml_models")
