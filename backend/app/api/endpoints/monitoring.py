from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict
import pandas as pd
import numpy as np

from app.core.database import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.project import Project
from app.models.ml_model import MLModel
from app.models.monitoring_snapshot import MonitoringSnapshot
from app.services.monitoring_service import MonitoringService

router = APIRouter(tags=["Monitoring"])


def _get_user_project(project_id: str, user: User, db: Session) -> Project:
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.owner_id == user.id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project

@router.get("/monitoring/models/{model_id}/drift")
def get_model_drift(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get data drift report for a specific model."""
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    _get_user_project(model.project_id, current_user, db)
        
    try:
        service = MonitoringService(db)
        return service.calculate_live_drift(model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/models/{model_id}/drift-live")
def get_model_live_drift(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get live drift scores vs baseline (MVP simulated)."""
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    _get_user_project(model.project_id, current_user, db)

    try:
        service = MonitoringService(db)
        return service.calculate_live_drift(model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitoring/models/{model_id}/health")
def get_model_health(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get health and reliability metrics for a model."""
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    _get_user_project(model.project_id, current_user, db)
    service = MonitoringService(db)
    return service.get_health_stats(model_id)


@router.get("/monitoring/models/{model_id}/snapshots/latest")
def get_model_latest_snapshot(
    model_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the latest materialized monitoring snapshot for a model."""
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    _get_user_project(model.project_id, current_user, db)

    snapshot = (
        db.query(MonitoringSnapshot)
        .filter(MonitoringSnapshot.model_id == model_id)
        .order_by(MonitoringSnapshot.window_start.desc())
        .first()
    )
    if not snapshot:
        service = MonitoringService(db)
        service.get_health_stats(model_id)
        snapshot = (
            db.query(MonitoringSnapshot)
            .filter(MonitoringSnapshot.model_id == model_id)
            .order_by(MonitoringSnapshot.window_start.desc())
            .first()
        )
    if not snapshot:
        return {"model_id": model_id, "snapshot": None}

    return {
        "model_id": model_id,
        "snapshot": {
            "window_start": snapshot.window_start,
            "window_end": snapshot.window_end,
            "resolution_minutes": snapshot.resolution_minutes,
            "source": snapshot.source,
            "request_count": snapshot.request_count,
            "row_count": snapshot.row_count,
            "latency_avg_ms": snapshot.latency_avg_ms,
            "latency_p95_ms": snapshot.latency_p95_ms,
            "error_rate": snapshot.error_rate,
            "uptime_pct": snapshot.uptime_pct,
            "drift_share": snapshot.drift_share,
            "drift_detected": snapshot.drift_detected,
            "payload": snapshot.payload or {},
        },
    }

@router.get("/monitoring/projects/{project_id}/summary")
def get_project_monitoring_summary(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get monitoring overview for all models in a project."""
    _get_user_project(project_id, current_user, db)
    models = db.query(MLModel).filter(MLModel.project_id == project_id, MLModel.is_deployed == True).all()
    
    service = MonitoringService(db)
    summary = []
    
    for m in models:
        health = service.get_health_stats(m.id)
        summary.append({
            "model_id": m.id,
            "model_name": m.name,
            "stage": m.deployment_stage,
            "health": health
        })
        
    return summary
@router.get("/monitoring/summary")
def get_global_monitoring_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get monitoring overview for ALL deployed models of the user across projects."""
    # Find all deployed models belonging to projects owned by the user
    models = (
        db.query(MLModel)
        .join(MLModel.project)
        .filter(Project.owner_id == current_user.id, MLModel.is_deployed == True)
        .all()
    )
    
    service = MonitoringService(db)
    summary = []
    
    for m in models:
        health = service.get_health_stats(m.id)
        summary.append({
            "model_id": m.id,
            "model_name": m.name,
            "project_name": m.project.name,
            "stage": m.deployment_stage,
            "health": health
        })
        
    return summary
