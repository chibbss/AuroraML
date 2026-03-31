"""
Jobs Endpoints — Start training jobs, check status, get results.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import settings
from app.api.deps import get_current_user
from app.models.user import User
from app.models.project import Project
from app.models.dataset import Dataset
from app.models.job import Job
from app.schemas.job import JobCreate, JobResponse, JobListResponse

router = APIRouter(tags=["Jobs"])


def _get_user_project(project_id: str, user: User, db: Session) -> Project:
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.owner_id == user.id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project


@router.post(
    "/projects/{project_id}/jobs/train",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
)
def start_training_job(
    project_id: str,
    job_data: JobCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Start a new training job."""
    project = _get_user_project(project_id, current_user, db)

    # Verify dataset exists and belongs to project
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == job_data.dataset_id, Dataset.project_id == project_id)
        .first()
    )
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found in this project",
        )

    # Validate target column exists
    if dataset.column_names and job_data.target_column not in dataset.column_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{job_data.target_column}' not found. "
            f"Available: {dataset.column_names}",
        )

    # Default model types
    model_types = job_data.model_types or ["random_forest", "xgboost", "lightgbm"]

    if settings.REQUIRE_DURABLE_WORKERS and not settings.USE_CELERY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Durable workers are required, but Celery is disabled.",
        )
    if not settings.USE_CELERY and not settings.ALLOW_INPROCESS_JOBS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="In-process training is disabled. Enable Celery workers to run jobs.",
        )

    job = Job(
        project_id=project.id,
        dataset_id=dataset.id,
        target_column=job_data.target_column,
        problem_type=job_data.problem_type,
        model_types=model_types,
        config={
            **(job_data.config or {}),
            "runtime": {
                "dispatch_mode": "celery" if settings.USE_CELERY else "inprocess",
                "durable_workers_required": settings.REQUIRE_DURABLE_WORKERS,
            },
        },
        status="pending",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        if settings.USE_CELERY:
            from app.tasks.training_tasks import run_training_job as training_task

            task = training_task.delay(job.id)
            job.celery_task_id = task.id
        else:
            from app.services.training_service import TrainingService
            import threading

            def run_job():
                from app.core.database import SessionLocal
                session = SessionLocal()
                try:
                    service = TrainingService(session)
                    service.run_training_job(job.id)
                finally:
                    session.close()

            thread = threading.Thread(target=run_job, daemon=True)
            thread.start()
        job.status = "running"
        db.commit()
        db.refresh(job)
    except Exception as exc:
        job.status = "failed"
        job.error_message = f"Failed to dispatch training job: {exc}"
        db.commit()
        db.refresh(job)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to dispatch training job",
        )

    return job


@router.get("/projects/{project_id}/jobs", response_model=JobListResponse)
def list_jobs(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all jobs for a project."""
    _get_user_project(project_id, current_user, db)
    jobs = (
        db.query(Job)
        .filter(Job.project_id == project_id)
        .order_by(Job.created_at.desc())
        .all()
    )
    return JobListResponse(jobs=jobs, total=len(jobs))


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get job details and results."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    _get_user_project(job.project_id, current_user, db)
    return job


@router.post("/jobs/{job_id}/cancel", response_model=JobResponse)
def cancel_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Cancel a pending or running job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    _get_user_project(job.project_id, current_user, db)

    if job.status not in ("pending", "running"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job with status: {job.status}",
        )

    job.status = "cancelled"
    db.commit()
    db.refresh(job)
    return job
