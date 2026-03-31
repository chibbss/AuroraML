"""Monitoring Celery Tasks — Periodic rollups and drift refresh."""

import logging

from app.tasks.training_tasks import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="monitoring.refresh_model_snapshot")
def refresh_model_snapshot(self, model_id: str):
    """Refresh monitoring rollups for a deployed model."""
    from app.core.database import SessionLocal
    from app.models.ml_model import MLModel
    from app.services.monitoring_service import MonitoringService

    db = SessionLocal()
    try:
        model = db.query(MLModel).filter(MLModel.id == model_id).first()
        if not model:
            raise ValueError(f"Model {model_id} not found")
        service = MonitoringService(db)
        health = service.get_health_stats(model_id)
        drift = service.calculate_live_drift(model_id)
        logger.info(f"Refreshed monitoring snapshot for model {model_id}")
        return {"health": health, "drift": drift}
    finally:
        db.close()


@celery_app.task(name="monitoring.refresh_all_snapshots")
def refresh_all_snapshots():
    """Refresh monitoring rollups for all deployed models."""
    from app.core.database import SessionLocal
    from app.models.ml_model import MLModel
    from app.services.monitoring_service import MonitoringService

    db = SessionLocal()
    try:
        models = db.query(MLModel).filter(MLModel.is_deployed == True).all()
        service = MonitoringService(db)
        refreshed = 0
        for model in models:
            service.get_health_stats(model.id)
            service.calculate_live_drift(model.id)
            refreshed += 1
        logger.info(f"Refreshed monitoring rollups for {refreshed} models")
        return {"refreshed_models": refreshed}
    finally:
        db.close()
