"""Dataset Pydantic Schemas."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class DatasetResponse(BaseModel):
    id: str
    project_id: str
    project_name: Optional[str] = None
    filename: str
    original_filename: str
    file_size: int
    file_type: str
    num_rows: Optional[int]
    num_columns: Optional[int]
    column_names: Optional[list]
    column_types: Optional[dict]
    summary_stats: Optional[dict]
    version: int
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetUploadResponse(BaseModel):
    id: str
    filename: str
    file_size: int
    num_rows: int
    num_columns: int
    column_names: list[str]
    column_types: dict
    message: str = "Dataset uploaded successfully"

    model_config = {"from_attributes": True}


class DatasetPreview(BaseModel):
    columns: list[str]
    dtypes: dict[str, str]
    head: list[dict]  # first N rows as list of dicts
    shape: list[int]  # [num_rows, num_columns]
    missing_values: dict[str, int]


class DatasetFinding(BaseModel):
    title: str
    detail: str
    severity: str
    feature: Optional[str] = None


class DatasetRecommendation(BaseModel):
    text: str


class DatasetCorrelatedPair(BaseModel):
    left: str
    right: str
    strength: float


class DatasetFeatureRole(BaseModel):
    name: str
    label: str
    role: str
    confidence: float
    rationale: str


class DatasetFeatureSpotlight(BaseModel):
    name: str
    label: str
    role: str
    quality_score: float
    note: str


class DatasetTargetHealth(BaseModel):
    label: str
    value: str
    status: str


class DatasetDistributionItem(BaseModel):
    label: str
    share: float


class DatasetTargetRelationship(BaseModel):
    feature: str
    strength: float
    direction: str


class DatasetSegmentItem(BaseModel):
    title: str
    feature: str
    cohort: str
    sample_size: int
    share_of_rows: float
    target_signal: Optional[float] = None
    comparison: str
    insight: str


class DatasetResearchSection(BaseModel):
    title: str
    body: str


class DatasetReportOverview(BaseModel):
    rows: int
    columns: int
    numeric_features: int
    categorical_features: int
    datetime_features: int
    quality_score: float
    modeling_readiness_score: float
    dataset_type: str = "training_dataset"
    dataset_type_confidence: float = 0.6
    dataset_type_signals: list[str] = []


class DatasetReportQuality(BaseModel):
    score: float
    missing_cell_ratio: float
    duplicate_ratio: float
    constant_features: list[str]
    id_like_features: list[str] = []
    high_missing_features: list[str]
    identifier_like_features: list[str]
    high_cardinality_features: list[str]
    skewed_features: list[str]
    formula_like_columns: list[str] = []
    template_flags: list[str] = []
    non_dataset_flags: list[str] = []
    dataset_type: str = "training_dataset"
    dataset_type_confidence: float = 0.6
    dataset_type_signals: list[str] = []
    correlated_pairs: list[DatasetCorrelatedPair]


class DatasetModelingReadiness(BaseModel):
    score: float
    status: str
    summary: str


class DatasetTargetAnalysis(BaseModel):
    recommended_target: Optional[str] = None
    problem_type: str
    rationale: str
    target_health: list[DatasetTargetHealth]
    distribution: list[DatasetDistributionItem] = []
    top_relationships: list[DatasetTargetRelationship] = []
    imbalance_ratio: Optional[float] = None


class DatasetReportResponse(BaseModel):
    overview: DatasetReportOverview
    quality: DatasetReportQuality
    modeling_readiness: DatasetModelingReadiness
    target_analysis: DatasetTargetAnalysis
    segments: list[DatasetSegmentItem]
    feature_roles: list[DatasetFeatureRole]
    feature_spotlight: list[DatasetFeatureSpotlight]
    findings: list[DatasetFinding]
    recommendations: list[str]
    analyst_brief: list[str]
    research_report: list[DatasetResearchSection]


class DatasetAskRequest(BaseModel):
    question: str


class DatasetAskResponse(BaseModel):
    answer: str
    citations: list[str]
    provider: str
    grounded: bool = True
    warning: Optional[str] = None


class DatasetTypeOverrideRequest(BaseModel):
    dataset_type: str
