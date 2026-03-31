"""
Training Celery Tasks — Background job execution.
When Redis/Celery is available, these tasks handle long-running training jobs.
"""

import logging
from celery import Celery
from app.core.config import settings

logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery(
    "auroraml",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3500,
)


@celery_app.task(bind=True, name="training.run_training_job")
def run_training_job(self, job_id: str):
    """Execute a training job as a Celery task."""
    from app.core.database import SessionLocal
    from app.services.training_service import TrainingService

    logger.info(f"Starting training job: {job_id}")

    db = SessionLocal()
    try:
        service = TrainingService(db)
        service.run_training_job(job_id)
        logger.info(f"Training job {job_id} completed")
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")
        raise
    finally:
        db.close()


@celery_app.task(bind=True, name="training.run_batch_prediction")
def run_batch_prediction(self, model_id: str, input_file_path: str, output_file_path: str):
    """Execute batch predictions as a Celery task."""
    import pandas as pd

    logger.info(f"Starting batch prediction for model: {model_id}")

    from app.core.database import SessionLocal
    from app.models.ml_model import MLModel
    from app.services.model_artifact_service import load_model_artifact

    db = SessionLocal()
    try:
        model = db.query(MLModel).filter(MLModel.id == model_id).first()
        if not model or not model.file_path:
            raise ValueError(f"Model {model_id} not found or has no artifact")

        pipeline, artifact_metadata = load_model_artifact(model.file_path)
        input_df = pd.read_csv(input_file_path)
        predictions = pipeline.predict(input_df).tolist()
        class_labels = [str(label) for label in artifact_metadata.get("class_labels", [])]
        if class_labels:
            predictions = [
                class_labels[int(value)] if isinstance(value, (int, float)) and int(value) < len(class_labels) else value
                for value in predictions
            ]
        input_df["prediction"] = predictions
        input_df.to_csv(output_file_path, index=False)

        logger.info(f"Batch prediction completed: {len(predictions)} predictions saved")
    finally:
        db.close()
