"""
Predictions Endpoint — Make predictions using deployed models.
"""

import os
import time
import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.ml_model import MLModel
from app.models.project import Project
from app.models.prediction_event import PredictionEvent
from app.schemas.ml_model import PredictionRequest, PredictionResponse
from app.services.model_artifact_service import load_model_artifact

router = APIRouter(tags=["Predictions"])


def _map_predictions(predictions: list, class_labels: list[str]) -> list:
    if not class_labels:
        return predictions
    mapped = []
    for value in predictions:
        if isinstance(value, (int, np.integer)) and 0 <= int(value) < len(class_labels):
            mapped.append(class_labels[int(value)])
        else:
            mapped.append(value)
    return mapped


def _map_probabilities(probabilities: np.ndarray | None, class_labels: list[str]) -> list | None:
    if probabilities is None:
        return None
    rows = probabilities.tolist()
    if not class_labels or len(class_labels) != probabilities.shape[1]:
        return rows
    return [
        [{"label": class_labels[idx], "probability": round(float(prob), 6)} for idx, prob in enumerate(row)]
        for row in probabilities
    ]


@router.post("/predict", response_model=PredictionResponse)
def make_prediction(
    request: PredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Make predictions using a deployed model."""
    model = (
        db.query(MLModel)
        .join(Project, Project.id == MLModel.project_id)
        .filter(MLModel.id == request.model_id, Project.owner_id == current_user.id)
        .first()
    )
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    if not model.is_deployed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model is not deployed. Deploy it first.",
        )

    if not model.file_path or not os.path.exists(model.file_path):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model artifact not found on disk",
        )

    # Load the model pipeline
    try:
        pipeline, artifact_metadata = load_model_artifact(model.file_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )

    # Build input DataFrame
    try:
        input_df = pd.DataFrame(request.data)
        if input_df.ndim == 1 or len(input_df) == 0:
            # Single prediction — data was {feature: value} not {feature: [value]}
            input_df = pd.DataFrame([request.data])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input data: {str(e)}",
        )

    if input_df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input data must contain at least one row",
        )

    # Make predictions
    started = time.perf_counter()
    try:
        raw_predictions = pipeline.predict(input_df).tolist()

        # Get probabilities for classifiers
        probability_array = None
        if hasattr(pipeline, "predict_proba"):
            try:
                probability_array = pipeline.predict_proba(input_df)
            except Exception:
                pass

        class_labels = [str(label) for label in artifact_metadata.get("class_labels", [])]
        predictions = _map_predictions(raw_predictions, class_labels)
        probabilities = _map_probabilities(probability_array, class_labels)

        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        db.add(
            PredictionEvent(
                model_id=model.id,
                project_id=model.project_id,
                user_id=current_user.id,
                request_rows=int(len(input_df)),
                latency_ms=latency_ms,
                success=True,
                status_code=200,
                request_payload={
                    "records": input_df.head(100).replace({np.nan: None}).to_dict(orient="records"),
                    "truncated": len(input_df) > 100,
                },
                prediction_payload={
                    "predictions": predictions[:100],
                    "probabilities": probabilities[:100] if probabilities else None,
                },
            )
        )
        db.commit()

        return PredictionResponse(
            model_id=model.id,
            model_name=model.name,
            predictions=predictions,
            probabilities=probabilities,
        )
    except Exception as e:
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        db.add(
            PredictionEvent(
                model_id=model.id,
                project_id=model.project_id,
                user_id=current_user.id,
                request_rows=int(len(input_df)),
                latency_ms=latency_ms,
                success=False,
                status_code=500,
                request_payload={
                    "records": input_df.head(100).replace({np.nan: None}).to_dict(orient="records"),
                    "truncated": len(input_df) > 100,
                },
                error_message=str(e),
            )
        )
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )
