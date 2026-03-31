"""Job (Training Job) SQLAlchemy Model."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Text, DateTime, ForeignKey, JSON, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )
    # Job configuration
    job_type: Mapped[str] = mapped_column(String(50), default="training")  # training, tuning
    status: Mapped[str] = mapped_column(
        String(50), default="pending"
    )  # pending, running, completed, failed, cancelled
    target_column: Mapped[str] = mapped_column(String(255), nullable=True)
    problem_type: Mapped[str] = mapped_column(
        String(50), nullable=True
    )  # classification, regression
    model_types: Mapped[dict] = mapped_column(JSON, nullable=True)  # list of model types to try
    config: Mapped[dict] = mapped_column(JSON, nullable=True)  # full pipeline config
    # Results
    best_model_type: Mapped[str] = mapped_column(String(100), nullable=True)
    best_model_id: Mapped[str] = mapped_column(String(36), nullable=True)
    best_score: Mapped[float] = mapped_column(Float, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=True)  # all metrics from best model
    all_results: Mapped[dict] = mapped_column(JSON, nullable=True)  # results from all models
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    # Tracking
    mlflow_run_id: Mapped[str] = mapped_column(String(100), nullable=True)
    celery_task_id: Mapped[str] = mapped_column(String(100), nullable=True)
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    project = relationship("Project", back_populates="jobs")
    dataset = relationship("Dataset", back_populates="jobs")
    ml_models = relationship("MLModel", back_populates="job", cascade="all, delete-orphan")
