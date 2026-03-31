"""
Models Endpoints — List trained models, deploy/undeploy.
"""

from datetime import datetime, timezone
from typing import Any
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.project import Project
from app.models.ml_model import MLModel
from app.models.job import Job
from app.models.dataset import Dataset
from app.schemas.ml_model import (
    MLModelResponse,
    MLModelListResponse,
    DeployRequest,
    FeatureEffectsResponse,
    FeatureEffect,
    FeatureEffectPoint,
    DriftBaselineResponse,
    DriftBaselineFeature,
    LocalExplanationRequest,
    LocalExplanationResponse,
    LocalExplanationContribution,
)
from app.services.dataset_service import DatasetService
from app.services.model_artifact_service import load_model_artifact

router = APIRouter(tags=["Models"])


def _labelize(value: str) -> str:
    return value.replace("_", " ").replace("cat__", "").replace("num__", "").title()


def _build_per_class_metrics(metrics: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not metrics:
        return []
    confusion = metrics.get("confusion_matrix") or {}
    labels = confusion.get("labels")
    matrix = confusion.get("matrix")
    if not isinstance(labels, list) or not isinstance(matrix, list):
        return []
    class_labels = metrics.get("class_labels") or []

    rows = []
    for idx, label in enumerate(labels):
        if idx >= len(matrix) or not isinstance(matrix[idx], list):
            continue
        row = matrix[idx]
        tp = row[idx] if idx < len(row) and isinstance(row[idx], (int, float)) else 0
        fn = sum(v for j, v in enumerate(row) if j != idx and isinstance(v, (int, float)))
        fp = 0
        for row_idx, matrix_row in enumerate(matrix):
            if row_idx == idx or not isinstance(matrix_row, list) or idx >= len(matrix_row):
                continue
            cell = matrix_row[idx]
            if isinstance(cell, (int, float)):
                fp += cell

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        support = sum(v for v in row if isinstance(v, (int, float)))
        rows.append(
            {
                "label": (
                    str(class_labels[int(label)])
                    if class_labels and str(label).isdigit() and int(label) < len(class_labels)
                    else str(label)
                ),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "support": int(support),
            }
        )
    return rows


def _build_report(model: MLModel) -> dict[str, Any]:
    metrics = model.metrics or {}
    feature_importance = model.feature_importance or {}
    hyperparameters = model.hyperparameters or {}

    sorted_features = sorted(
        (
            {
                "name": str(name),
                "label": _labelize(str(name)),
                "importance": round(float(value), 6),
            }
            for name, value in feature_importance.items()
            if isinstance(value, (int, float))
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )

    primary_score = (
        metrics.get("f1_score")
        or metrics.get("r2_score")
        or metrics.get("accuracy")
        or 0
    )

    return {
        "summary": {
            "problem_type": metrics.get("lineage", {}).get("problem_type") or model.model_type,
            "framework": model.framework,
            "primary_score": round(float(primary_score), 4) if isinstance(primary_score, (int, float)) else 0.0,
            "feature_count": len(feature_importance),
            "parameter_count": len(hyperparameters),
        },
        "top_features": sorted_features[:12],
        "per_class_metrics": _build_per_class_metrics(metrics),
        "validation": metrics.get("validation_strategy") or {},
        "cross_validation": metrics.get("cross_validation") or {},
        "lineage": metrics.get("lineage") or {},
        "class_labels": metrics.get("class_labels") or [],
        "quality_signals": {
            "roc_auc": metrics.get("roc_auc"),
            "average_precision": metrics.get("average_precision"),
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score"),
            "log_loss": metrics.get("log_loss"),
            "matthews_corrcoef": metrics.get("matthews_corrcoef"),
            "r2_score": metrics.get("r2_score"),
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
        },
        "deployment": {
            "is_deployed": model.is_deployed,
            "stage": model.deployment_stage,
            "endpoint_url": model.endpoint_url,
        },
    }


def _serialize_model(model: MLModel) -> dict[str, Any]:
    payload = MLModelResponse.model_validate(model).model_dump(mode="json")
    payload["report"] = _build_report(model)
    return payload


def _get_model_job_dataset(model: MLModel, db: Session) -> tuple[Job, Dataset]:
    job = db.query(Job).filter(Job.id == model.job_id).first()
    if not job or not job.dataset_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training job or dataset not found")
    dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return job, dataset


def _compute_feature_effects(model: MLModel, job: Job, dataset: Dataset) -> list[dict[str, Any]]:
    if not model.file_path:
        return []

    df = DatasetService.read_dataframe(dataset.file_path, dataset.file_type)
    target = job.target_column
    if target and target in df.columns:
        df = df.drop(columns=[target])

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        return []

    importance = model.feature_importance or {}
    ranked = [name for name, _ in sorted(importance.items(), key=lambda item: item[1], reverse=True)]
    candidate_features = [f for f in ranked if f in numeric_cols]
    if not candidate_features:
        candidate_features = numeric_cols
    candidate_features = candidate_features[:4]

    sample = df.sample(n=min(2000, len(df)), random_state=42) if len(df) > 2000 else df.copy()
    pipeline, _ = load_model_artifact(model.file_path)

    effects = []
    for feature in candidate_features:
        try:
            result = partial_dependence(pipeline, sample, [feature], kind="average")
            xs = result["grid_values"][0]
            ys = result["average"][0]
            points = [
                {"x": float(x), "y": float(y)} for x, y in zip(xs, ys)
                if np.isfinite(x) and np.isfinite(y)
            ]
            effects.append({"feature": feature, "method": "model", "points": points})
        except Exception:
            if target and target in dataset.column_names:
                df_full = DatasetService.read_dataframe(dataset.file_path, dataset.file_type)
                if target in df_full.columns and feature in df_full.columns:
                    series = df_full[feature]
                    target_series = df_full[target]
                    target_numeric = pd.to_numeric(target_series, errors="coerce")
                    if target_numeric.isna().all():
                        unique_targets = target_series.dropna().unique()
                        if len(unique_targets) == 2:
                            target_numeric = target_series.map({unique_targets[0]: 0, unique_targets[1]: 1})
                        else:
                            target_numeric = pd.Series([], dtype=float)

                    if not target_numeric.empty:
                        bins = pd.qcut(series, q=8, duplicates="drop")
                        grouped = target_numeric.groupby(bins).mean().reset_index()
                        points = []
                        for _, row in grouped.iterrows():
                            bin_range = row[feature]
                            center = float(bin_range.mid) if hasattr(bin_range, "mid") else float(bin_range.left)
                            points.append({"x": center, "y": float(row[target])})
                        effects.append({"feature": feature, "method": "data", "points": points})

    return effects


def _compute_drift_baseline(model: MLModel, job: Job, dataset: Dataset) -> list[dict[str, Any]]:
    df = DatasetService.read_dataframe(dataset.file_path, dataset.file_type)
    target = job.target_column
    if target and target in df.columns:
        df = df.drop(columns=[target])

    importance = model.feature_importance or {}
    ranked = [name for name, _ in sorted(importance.items(), key=lambda item: item[1], reverse=True)]
    if not ranked:
        ranked = df.columns.tolist()

    features = []
    for feature in ranked[:8]:
        if feature not in df.columns:
            continue
        series = df[feature]
        missing_pct = float(series.isna().mean() * 100)
        dtype = str(series.dtype)
        entry = {
            "feature": feature,
            "dtype": dtype,
            "missing_pct": round(missing_pct, 2),
        }
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                entry.update(
                    {
                        "mean": float(numeric.mean()),
                        "std": float(numeric.std()),
                        "p10": float(numeric.quantile(0.1)),
                        "p90": float(numeric.quantile(0.9)),
                    }
                )
        else:
            vc = series.dropna().astype(str).value_counts()
            if not vc.empty:
                top_value = vc.index[0]
                top_share = float(vc.iloc[0] / max(len(series.dropna()), 1))
                entry.update(
                    {
                        "top_value": str(top_value),
                        "top_share": round(top_share, 4),
                        "unique_count": int(series.dropna().nunique()),
                    }
                )
        features.append(entry)
    return features


def _build_default_row(df: pd.DataFrame) -> dict[str, Any]:
    defaults = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            defaults[col] = float(pd.to_numeric(series, errors="coerce").median())
        else:
            mode_series = series.dropna().astype(str).mode()
            defaults[col] = mode_series.iloc[0] if not mode_series.empty else ""
    return defaults


def _predict_value(
    pipeline,
    row_df: pd.DataFrame,
    class_labels: list[str] | None = None,
) -> tuple[float, str | None]:
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(row_df)[0]
        max_idx = int(np.argmax(proba))
        if len(proba) == 2:
            if class_labels and len(class_labels) == 2:
                return float(proba[max_idx]), class_labels[max_idx]
            return float(proba[1]), "positive"
        if class_labels and max_idx < len(class_labels):
            return float(proba[max_idx]), class_labels[max_idx]
        return float(proba[max_idx]), f"class_{max_idx}"
    pred = pipeline.predict(row_df)[0]
    return float(pred), None


def _get_user_project(project_id: str, user: User, db: Session) -> Project:
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.owner_id == user.id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project


@router.get("/projects/{project_id}/models", response_model=MLModelListResponse)
def list_models(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all trained models for a project."""
    _get_user_project(project_id, current_user, db)
    models = (
        db.query(MLModel)
        .filter(MLModel.project_id == project_id)
        .order_by(MLModel.created_at.desc())
        .all()
    )
    return MLModelListResponse(models=[_serialize_model(model) for model in models], total=len(models))


@router.get("/models/{model_id}", response_model=MLModelResponse)
def get_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get model details."""
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    _get_user_project(model.project_id, current_user, db)
    return _serialize_model(model)


@router.get("/models/{model_id}/feature-effects", response_model=FeatureEffectsResponse)
def get_feature_effects(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get model-based feature effects (partial dependence for numeric features)."""
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    _get_user_project(model.project_id, current_user, db)
    job, dataset = _get_model_job_dataset(model, db)
    effects = _compute_feature_effects(model, job, dataset)
    return FeatureEffectsResponse(
        model_id=model.id,
        effects=[FeatureEffect(feature=e["feature"], method=e["method"], points=[FeatureEffectPoint(**p) for p in e["points"]]) for e in effects],
    )


@router.get("/models/{model_id}/drift-baseline", response_model=DriftBaselineResponse)
def get_drift_baseline(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get baseline feature distribution statistics from training data."""
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    _get_user_project(model.project_id, current_user, db)
    job, dataset = _get_model_job_dataset(model, db)
    features = _compute_drift_baseline(model, job, dataset)
    return DriftBaselineResponse(
        model_id=model.id,
        generated_at=datetime.now(timezone.utc),
        sample_size=dataset.num_rows or 0,
        features=[DriftBaselineFeature(**feature) for feature in features],
    )


@router.post("/models/{model_id}/explain", response_model=LocalExplanationResponse)
def explain_prediction(
    model_id: str,
    payload: LocalExplanationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate local feature contribution explanations for a single prediction."""
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    _get_user_project(model.project_id, current_user, db)
    if not model.file_path:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model artifact not available")

    job, dataset = _get_model_job_dataset(model, db)
    df = DatasetService.read_dataframe(dataset.file_path, dataset.file_type)
    if job.target_column and job.target_column in df.columns:
        df = df.drop(columns=[job.target_column])

    defaults = _build_default_row(df)
    input_data = payload.data or {}
    row = defaults.copy()
    for key, value in input_data.items():
        if key in row:
            row[key] = value

    row_df = pd.DataFrame([row])
    pipeline, artifact_metadata = load_model_artifact(model.file_path)
    class_labels = [str(label) for label in artifact_metadata.get("class_labels", [])]
    base_pred, base_label = _predict_value(pipeline, row_df, class_labels)

    contributions = []
    for feature, value in input_data.items():
        if feature not in row:
            continue
        modified = row.copy()
        modified[feature] = defaults.get(feature)
        modified_df = pd.DataFrame([modified])
        modified_pred, _ = _predict_value(pipeline, modified_df, class_labels)
        delta = base_pred - modified_pred
        contributions.append(
            LocalExplanationContribution(
                feature=feature,
                value=str(value),
                baseline=str(defaults.get(feature)),
                delta=float(delta),
            )
        )

    contributions.sort(key=lambda c: abs(c.delta), reverse=True)
    return LocalExplanationResponse(
        model_id=model.id,
        prediction=base_pred,
        prediction_label=base_label,
        contributions=contributions[:12],
    )


@router.post("/models/{model_id}/deploy", response_model=MLModelResponse)
def deploy_model(
    model_id: str,
    deploy_req: DeployRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Deploy a trained model to staging or production."""
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    _get_user_project(model.project_id, current_user, db)

    if not model.file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model has no saved artifact to deploy",
        )

    model.is_deployed = True
    model.deployment_stage = deploy_req.stage
    model.deployed_at = datetime.now(timezone.utc)
    model.endpoint_url = f"/api/v1/predict?model_id={model.id}"

    db.commit()
    db.refresh(model)
    return _serialize_model(model)


@router.post("/models/{model_id}/undeploy", response_model=MLModelResponse)
def undeploy_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Undeploy a model."""
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    _get_user_project(model.project_id, current_user, db)

    model.is_deployed = False
    model.deployment_stage = "archived"
    model.endpoint_url = None

    db.commit()
    db.refresh(model)
    return _serialize_model(model)
@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a trained model and its artifacts."""
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    _get_user_project(model.project_id, current_user, db)

    # Note: In a real app we'd also delete the file from disk/MinIO here.
    db.delete(model)
    db.commit()
    return None
