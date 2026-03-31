"""ML Model Pydantic Schemas."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class MLModelResponse(BaseModel):
    id: str
    job_id: str
    project_id: str
    name: str
    model_type: str
    framework: str
    version: int
    metrics: Optional[dict]
    hyperparameters: Optional[dict]
    feature_importance: Optional[dict]
    is_deployed: bool
    deployment_stage: Optional[str]
    endpoint_url: Optional[str]
    report: Optional[dict] = None
    created_at: datetime
    deployed_at: Optional[datetime]

    model_config = {"from_attributes": True}


class FeatureEffectPoint(BaseModel):
    x: float
    y: float


class FeatureEffect(BaseModel):
    feature: str
    method: str
    points: list[FeatureEffectPoint]


class FeatureEffectsResponse(BaseModel):
    model_id: str
    effects: list[FeatureEffect]


class DriftBaselineFeature(BaseModel):
    feature: str
    dtype: str
    missing_pct: float
    mean: Optional[float] = None
    std: Optional[float] = None
    p10: Optional[float] = None
    p90: Optional[float] = None
    top_value: Optional[str] = None
    top_share: Optional[float] = None
    unique_count: Optional[int] = None


class DriftBaselineResponse(BaseModel):
    model_id: str
    generated_at: datetime
    sample_size: int
    features: list[DriftBaselineFeature]


class LocalExplanationRequest(BaseModel):
    data: dict


class LocalExplanationContribution(BaseModel):
    feature: str
    value: Optional[str] = None
    baseline: Optional[str] = None
    delta: float


class LocalExplanationResponse(BaseModel):
    model_id: str
    prediction: float
    prediction_label: Optional[str] = None
    contributions: list[LocalExplanationContribution]


class MLModelListResponse(BaseModel):
    models: list[MLModelResponse]
    total: int


class DeployRequest(BaseModel):
    stage: str = "staging"  # staging or production


class PredictionRequest(BaseModel):
    model_id: str
    data: dict  # feature_name -> value (single) or feature_name -> list[values] (batch)


class PredictionResponse(BaseModel):
    model_id: str
    model_name: str
    predictions: list
    probabilities: Optional[list] = None  # for classification
