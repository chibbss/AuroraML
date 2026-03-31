"""
Deployment Service — Model deployment and serving management.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session
from app.models.ml_model import MLModel
from app.models.job import Job

logger = logging.getLogger(__name__)


class DeploymentService:
    """Manages model deployment lifecycle."""

    def __init__(self, db: Session):
        self.db = db

    def deploy_model(self, model_id: str, stage: str = "staging") -> MLModel:
        """Deploy a model to the specified stage."""
        model = self.db.query(MLModel).filter(MLModel.id == model_id).first()
        if not model:
            raise ValueError(f"Model {model_id} not found")

        if not model.file_path or not os.path.exists(model.file_path):
            raise ValueError("Model artifact not found")

        model.is_deployed = True
        model.deployment_stage = stage
        model.deployed_at = datetime.now(timezone.utc)
        model.endpoint_url = f"/api/v1/predict?model_id={model.id}"

        self.db.commit()
        self.db.refresh(model)

        logger.info(f"Model {model.name} deployed to {stage}")
        return model

    def undeploy_model(self, model_id: str) -> MLModel:
        """Undeploy a model."""
        model = self.db.query(MLModel).filter(MLModel.id == model_id).first()
        if not model:
            raise ValueError(f"Model {model_id} not found")

        model.is_deployed = False
        model.deployment_stage = "archived"
        model.endpoint_url = None

        self.db.commit()
        self.db.refresh(model)

        logger.info(f"Model {model.name} undeployed")
        return model

    def promote_model(self, model_id: str) -> MLModel:
        """Promote a model from staging to production."""
        model = self.db.query(MLModel).filter(MLModel.id == model_id).first()
        if not model:
            raise ValueError(f"Model {model_id} not found")

        if model.deployment_stage != "staging":
            raise ValueError("Can only promote models in staging")

        model.deployment_stage = "production"
        self.db.commit()
        self.db.refresh(model)

        logger.info(f"Model {model.name} promoted to production")
        return model

    def get_deployed_models(self, project_id: str) -> list[MLModel]:
        """Get all deployed models for a project."""
        return (
            self.db.query(MLModel)
            .filter(MLModel.project_id == project_id, MLModel.is_deployed == True)
            .all()
        )
