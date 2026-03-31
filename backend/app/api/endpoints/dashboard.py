"""
Dashboard Endpoints — Global statistics and system health.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.database import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.project import Project
from app.models.dataset import Dataset
from app.models.job import Job
from app.models.ml_model import MLModel

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/stats")
def get_dashboard_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get global statistics for the current user's dashboard."""
    # Counts
    projects_count = db.query(Project).filter(Project.owner_id == current_user.id).count()
    
    # For datasets, models and jobs, we filter by projects owned by the user
    project_ids = [p.id for p in db.query(Project.id).filter(Project.owner_id == current_user.id).all()]
    
    datasets_count = db.query(Dataset).filter(Dataset.project_id.in_(project_ids)).count() if project_ids else 0
    models_count = db.query(MLModel).filter(MLModel.project_id.in_(project_ids)).count() if project_ids else 0
    jobs_count = db.query(Job).filter(Job.project_id.in_(project_ids)).count() if project_ids else 0
    
    # Running jobs
    running_jobs = db.query(Job).filter(
        Job.project_id.in_(project_ids), 
        Job.status.in_(["pending", "running"])
    ).count() if project_ids else 0

    return {
        "projects_count": projects_count,
        "datasets_count": datasets_count,
        "models_count": models_count,
        "jobs_count": jobs_count,
        "running_jobs_count": running_jobs,
        "system_status": "healthy",
        "storage_usage_pct": 5,  # Mock for now until we add real quota tracking
    }
