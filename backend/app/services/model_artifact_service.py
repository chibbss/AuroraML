"""Utilities for saving and loading model artifacts with lineage metadata."""

from __future__ import annotations

from typing import Any

import joblib
import os


def save_model_artifact(file_path: str, pipeline: Any, metadata: dict[str, Any]) -> None:
    artifact = {
        "pipeline": pipeline,
        "metadata": metadata or {},
    }
    joblib.dump(artifact, file_path)


def load_model_artifact(file_path: str) -> tuple[Any, dict[str, Any]]:
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    artifact = joblib.load(file_path)
    if isinstance(artifact, dict) and "pipeline" in artifact:
        return artifact["pipeline"], dict(artifact.get("metadata") or {})
    return artifact, {}
