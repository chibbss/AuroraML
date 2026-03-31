"""
Datasets Endpoints — Upload, list, and manage datasets.
"""

import os
import hashlib
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import io
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.project import Project
from app.models.dataset import Dataset
from app.schemas.dataset import (
    DatasetResponse,
    DatasetUploadResponse,
    DatasetPreview,
    DatasetReportResponse,
    DatasetTypeOverrideRequest,
    DatasetAskRequest,
    DatasetAskResponse,
)
from app.services.dataset_service import DatasetService
from app.services.aurora_ai_service import AuroraAIService
from pydantic import BaseModel

router = APIRouter(prefix="/datasets", tags=["Datasets"])

ALLOWED_EXTENSIONS = {".csv", ".parquet", ".json", ".xlsx", ".xls"}


def _get_user_project(project_id: str, user: User, db: Session) -> Project:
    """Helper to fetch a project owned by the user."""
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.owner_id == user.id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project


def _read_dataframe(file_path: str, file_type: str) -> pd.DataFrame:
    """Read a file into a DataFrame based on type."""
    readers = {
        ".csv": pd.read_csv,
        ".parquet": pd.read_parquet,
        ".json": pd.read_json,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
    }
    reader = readers.get(file_type)
    if not reader:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_type}",
        )
    return reader(file_path)


@router.post(
    "/projects/{project_id}",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_dataset(
    project_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload a dataset file (CSV, Parquet, JSON, Excel)."""
    project = _get_user_project(project_id, current_user, db)

    # Validate extension
    _, ext = os.path.splitext(file.filename or "")
    ext = ext.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Save file to local storage
    storage_dir = os.path.join(settings.LOCAL_STORAGE_PATH, "datasets", project_id)
    os.makedirs(storage_dir, exist_ok=True)

    unique_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(storage_dir, unique_filename)

    content = await file.read()
    file_size = len(content)

    # Compute checksum
    checksum = hashlib.sha256(content).hexdigest()

    with open(file_path, "wb") as f:
        f.write(content)

    # Parse dataset metadata
    try:
        df = _read_dataframe(file_path, ext)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse file: {str(e)}",
        )

    column_names = df.columns.tolist()
    column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Basic summary stats
    summary_stats = {}
    try:
        desc = df.describe(include="all").to_dict()
        summary_stats = {
            "describe": {k: {sk: str(sv) for sk, sv in v.items()} for k, v in desc.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": column_types,
        }
        profile = DatasetService.profile_dataset(df)
        report = DatasetService.build_dataset_report(df, profile)
        summary_stats["quality_score"] = report["overview"]["quality_score"]
        summary_stats["readiness_score"] = report["modeling_readiness"]["score"]
        summary_stats["readiness_status"] = report["modeling_readiness"]["status"]
        summary_stats["dataset_type"] = report["overview"].get("dataset_type")
        summary_stats["dataset_type_confidence"] = report["overview"].get("dataset_type_confidence")
        summary_stats["dataset_type_signals"] = report["overview"].get("dataset_type_signals")
    except Exception:
        pass

    # Create DB record
    dataset = Dataset(
        project_id=project.id,
        filename=unique_filename,
        original_filename=file.filename or "unknown",
        file_path=file_path,
        file_size=file_size,
        file_type=ext.lstrip("."),
        num_rows=len(df),
        num_columns=len(df.columns),
        column_names=column_names,
        column_types=column_types,
        summary_stats=summary_stats,
        checksum=checksum,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return DatasetUploadResponse(
        id=dataset.id,
        filename=dataset.original_filename,
        file_size=dataset.file_size,
        num_rows=dataset.num_rows,
        num_columns=dataset.num_columns,
        column_names=column_names,
        column_types=column_types,
    )


@router.get("", response_model=list[DatasetResponse])
def list_all_datasets(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all datasets across all projects belonging to the current user."""
    # We join with Project to ensure we only get datasets where owner_id matches
    datasets = (
        db.query(Dataset)
        .join(Project)
        .filter(Project.owner_id == current_user.id)
        .all()
    )
    
    # Enforce project_name onto the response objects
    for d in datasets:
        d.project_name = d.project.name
        
    return datasets


@router.get("/projects/{project_id}", response_model=list[DatasetResponse])
def list_datasets(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all datasets for a project."""
    _get_user_project(project_id, current_user, db)
    datasets = db.query(Dataset).filter(Dataset.project_id == project_id).all()
    return datasets


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get dataset details."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    # Verify ownership
    _get_user_project(dataset.project_id, current_user, db)
    return dataset


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
def preview_dataset(
    dataset_id: str,
    rows: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Preview first N rows of a dataset."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    _get_user_project(dataset.project_id, current_user, db)

    try:
        df = _read_dataframe(dataset.file_path, f".{dataset.file_type}")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read dataset: {str(e)}",
        )

    return DatasetPreview(
        columns=df.columns.tolist(),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        head=df.head(rows).to_dict(orient="records"),
        shape=[len(df), len(df.columns)],
        missing_values=df.isnull().sum().to_dict(),
    )


@router.get("/{dataset_id}/profile")
def get_dataset_profile(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get the full data profile (Pandas profiling) of a dataset."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    _get_user_project(dataset.project_id, current_user, db)

    try:
        df = _read_dataframe(dataset.file_path, f".{dataset.file_type}")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read dataset: {str(e)}",
        )

    profile = DatasetService.profile_dataset(df)
    
    # Auto-detect likely target column using scored inference.
    recommended_target = DatasetService.recommend_target_column(df)

    profile["recommended_target"] = recommended_target
    
    # Auto-detect problem type if target is found
    if recommended_target:
        profile["problem_type"] = DatasetService.detect_problem_type(df, recommended_target)
    else:
        profile["problem_type"] = "classification" # Default
    
    return profile


@router.get("/{dataset_id}/report", response_model=DatasetReportResponse)
def get_dataset_report(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a structured intelligence report for a dataset."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    _get_user_project(dataset.project_id, current_user, db)

    try:
        df = _read_dataframe(dataset.file_path, f".{dataset.file_type}")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read dataset: {str(e)}",
        )

    profile = DatasetService.profile_dataset(df)
    if dataset.summary_stats and dataset.summary_stats.get("dataset_type_override"):
        profile["dataset_type_override"] = dataset.summary_stats.get("dataset_type_override")
    report = DatasetService.build_dataset_report(df, profile)
    try:
        summary = dict(dataset.summary_stats) if dataset.summary_stats else {}
        summary["quality_score"] = report["overview"]["quality_score"]
        summary["readiness_score"] = report["modeling_readiness"]["score"]
        summary["readiness_status"] = report["modeling_readiness"]["status"]
        summary["dataset_type"] = report["overview"].get("dataset_type")
        summary["dataset_type_confidence"] = report["overview"].get("dataset_type_confidence")
        summary["dataset_type_signals"] = report["overview"].get("dataset_type_signals")
        dataset.summary_stats = summary
        db.commit()
    except Exception:
        db.rollback()
    return DatasetReportResponse.model_validate(report)


@router.post("/{dataset_id}/dataset-type", status_code=status.HTTP_200_OK)
def set_dataset_type_override(
    dataset_id: str,
    payload: DatasetTypeOverrideRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Override dataset type classification."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    _get_user_project(dataset.project_id, current_user, db)

    summary = dict(dataset.summary_stats) if dataset.summary_stats else {}
    summary["dataset_type_override"] = payload.dataset_type
    dataset.summary_stats = summary
    db.commit()
    return {"status": "ok", "dataset_type_override": payload.dataset_type}


@router.post("/{dataset_id}/ask", response_model=DatasetAskResponse)
def ask_aurora_about_dataset(
    dataset_id: str,
    payload: DatasetAskRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Ask Aurora a grounded question about a dataset."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    _get_user_project(dataset.project_id, current_user, db)

    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty")

    try:
        df = _read_dataframe(dataset.file_path, f".{dataset.file_type}")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read dataset: {str(e)}",
        )

    profile = DatasetService.profile_dataset(df)
    report = DatasetService.build_dataset_report(df, profile)
    answer = AuroraAIService.answer_dataset_question(
        dataset_name=dataset.original_filename,
        question=question,
        report=report,
        profile=profile,
    )
    return DatasetAskResponse.model_validate(answer)


class CleanConfigRequest(BaseModel):
    target_column: Optional[str] = None
    remove_duplicates: bool = True
    drop_missing: bool = False
    missing_threshold: float = 0.4
    standardize_names: bool = True

@router.post("/{dataset_id}/clean", status_code=status.HTTP_200_OK)
def clean_dataset(
    dataset_id: str,
    config: CleanConfigRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Clean the dataset synchronously based on UI configuration settings."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    project = _get_user_project(dataset.project_id, current_user, db)

    try:
        df = _read_dataframe(dataset.file_path, f".{dataset.file_type}")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read dataset: {str(e)}",
        )

    initial_rows = len(df)
    
    # 1. Standardize Names
    if config.standardize_names:
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
    
    # 2. Drop Missing above threshold
    if config.drop_missing:
        threshold_count = int(len(df) * (1 - config.missing_threshold))
        df = df.dropna(axis=1, thresh=threshold_count)
        # For remaining NA rows, we can drop them or impute. MVP: Drop.
        df = df.dropna(axis=0)

    # 3. Remove Duplicates
    if config.remove_duplicates:
        df = df.drop_duplicates()

    final_rows = len(df)

    # Save cleaned file
    storage_dir = os.path.join(settings.LOCAL_STORAGE_PATH, "datasets", project.id)
    clean_filename = f"cleaned_{dataset.filename}"
    clean_file_path = os.path.join(storage_dir, clean_filename)
    
    # Support multiple formats, save as CSV for now
    df.to_csv(clean_file_path, index=False)
    
    # Update DB record with new location and stats
    dataset.file_path = clean_file_path
    dataset.filename = clean_filename
    dataset.num_rows = final_rows
    dataset.num_columns = len(df.columns)
    dataset.column_names = df.columns.tolist()
    db.commit()

    return {
        "status": "success",
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "removed_rows": initial_rows - final_rows,
        "dataset_id": dataset.id
    }


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a dataset."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    _get_user_project(dataset.project_id, current_user, db)

    # Remove file
    if os.path.exists(dataset.file_path):
        os.remove(dataset.file_path)

    db.delete(dataset)
    db.commit()
