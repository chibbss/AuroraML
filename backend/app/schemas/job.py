"""Job Pydantic Schemas."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    dataset_id: str = Field(..., description="ID of dataset to train on")
    target_column: str = Field(..., description="Target column name")
    problem_type: str = Field(
        ..., description="classification or regression", pattern="^(classification|regression)$"
    )
    model_types: Optional[list[str]] = Field(
        None,
        description="Model types to try. Default: all supported",
        examples=[["random_forest", "xgboost", "lightgbm"]],
    )
    config: Optional[dict] = Field(
        None,
        description="Advanced pipeline configuration (cleaning, feature engineering, tuning)",
    )
    auto_deploy: bool = Field(False, description="Auto-deploy best model on completion")


class JobResponse(BaseModel):
    id: str
    project_id: str
    dataset_id: Optional[str]
    job_type: str
    status: str
    target_column: Optional[str]
    problem_type: Optional[str]
    model_types: Optional[list]
    best_model_type: Optional[str]
    best_model_id: Optional[str]
    best_score: Optional[float]
    metrics: Optional[dict]
    all_results: Optional[dict]
    config: Optional[dict]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    model_config = {"from_attributes": True}


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    total: int
