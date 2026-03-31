from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import logging
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.dataset import Dataset
from app.models.ml_model import MLModel
from app.models.monitoring_snapshot import MonitoringSnapshot
from app.models.prediction_event import PredictionEvent
from app.services.dataset_service import DatasetService

logger = logging.getLogger(__name__)


class MonitoringService:
    def __init__(self, db: Session):
        self.db = db

    def _current_snapshot_window(self, resolution_minutes: int) -> tuple[datetime, datetime]:
        now = datetime.now(timezone.utc)
        minute_bucket = (now.minute // resolution_minutes) * resolution_minutes
        window_start = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minute_bucket)
        window_end = window_start + timedelta(minutes=resolution_minutes)
        return window_start, window_end

    def _load_reference_df(self, model: MLModel) -> pd.DataFrame:
        job = model.job
        if not job or not job.dataset_id:
            raise ValueError("No reference dataset found for this model")
        dataset = self.db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
        if not dataset:
            raise ValueError("Dataset not found for reference")
        df = DatasetService.read_dataframe(dataset.file_path, dataset.file_type)
        if job.target_column and job.target_column in df.columns:
            df = df.drop(columns=[job.target_column])
        return df

    def _load_live_records(self, model_id: str, lookback_hours: int = 24, max_rows: int = 2000) -> pd.DataFrame:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        events = (
            self.db.query(PredictionEvent)
            .filter(
                PredictionEvent.model_id == model_id,
                PredictionEvent.created_at >= cutoff,
                PredictionEvent.success == True,
            )
            .order_by(PredictionEvent.created_at.desc())
            .limit(200)
            .all()
        )

        records: list[dict[str, Any]] = []
        for event in events:
            payload = event.request_payload or {}
            event_records = payload.get("records") or []
            if isinstance(event_records, list):
                for row in event_records:
                    if isinstance(row, dict):
                        records.append(row)
                        if len(records) >= max_rows:
                            break
            if len(records) >= max_rows:
                break

        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    def _get_events_for_window(self, model_id: str, window_start: datetime, window_end: datetime) -> list[PredictionEvent]:
        return (
            self.db.query(PredictionEvent)
            .filter(
                PredictionEvent.model_id == model_id,
                PredictionEvent.created_at >= window_start,
                PredictionEvent.created_at < window_end,
            )
            .all()
        )

    def _psi(self, ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
        ref_clean = pd.to_numeric(ref, errors="coerce").dropna()
        cur_clean = pd.to_numeric(cur, errors="coerce").dropna()
        if ref_clean.empty or cur_clean.empty:
            return 0.0
        counts, bin_edges = np.histogram(ref_clean, bins=bins)
        ref_perc = counts / max(len(ref_clean), 1)
        cur_counts, _ = np.histogram(cur_clean, bins=bin_edges)
        cur_perc = cur_counts / max(len(cur_clean), 1)
        epsilon = 1e-6
        psi = np.sum((ref_perc - cur_perc) * np.log((ref_perc + epsilon) / (cur_perc + epsilon)))
        return float(psi)

    def calculate_live_drift(self, model_id: str) -> dict[str, Any]:
        """Calculate drift scores against the training baseline using logged inference traffic."""
        model = self.db.query(MLModel).filter(MLModel.id == model_id).first()
        if not model:
            raise ValueError("Model not found")

        try:
            ref_df = self._load_reference_df(model)
            current_df = self._load_live_records(model_id)
            if current_df.empty or len(current_df) < 20:
                return {
                    "drift_share": 0.0,
                    "number_of_columns": 0,
                    "number_of_drifted_columns": 0,
                    "dataset_drift": False,
                    "features": [],
                    "source": "prediction_events",
                    "window_hours": 24,
                    "sample_rows": int(len(current_df)),
                    "message": "Not enough recent prediction traffic to compute live drift.",
                }

            common_cols = [col for col in ref_df.columns if col in current_df.columns]
            if not common_cols:
                return {
                    "drift_share": 0.0,
                    "number_of_columns": 0,
                    "number_of_drifted_columns": 0,
                    "dataset_drift": False,
                    "features": [],
                    "source": "prediction_events",
                    "window_hours": 24,
                    "sample_rows": int(len(current_df)),
                    "message": "No overlapping features between training data and live requests.",
                }

            features = []
            drifted = 0
            for feature in common_cols:
                ref_series = ref_df[feature]
                cur_series = current_df[feature]
                if pd.api.types.is_numeric_dtype(ref_series):
                    score = self._psi(ref_series, cur_series)
                    metric = "PSI"
                    drift_detected = score > 0.2
                    baseline = {
                        "mean": float(pd.to_numeric(ref_series, errors="coerce").mean()),
                        "p10": float(pd.to_numeric(ref_series, errors="coerce").quantile(0.1)),
                        "p90": float(pd.to_numeric(ref_series, errors="coerce").quantile(0.9)),
                    }
                    current = {
                        "mean": float(pd.to_numeric(cur_series, errors="coerce").mean()),
                        "p10": float(pd.to_numeric(cur_series, errors="coerce").quantile(0.1)),
                        "p90": float(pd.to_numeric(cur_series, errors="coerce").quantile(0.9)),
                    }
                else:
                    ref_counts = ref_series.dropna().astype(str).value_counts(normalize=True)
                    cur_counts = cur_series.dropna().astype(str).value_counts(normalize=True)
                    population = sorted(set(ref_counts.index).union(set(cur_counts.index)))
                    score = 0.0
                    for key in population:
                        ref_share = float(ref_counts.get(key, 0.0))
                        cur_share = float(cur_counts.get(key, 0.0))
                        score += abs(ref_share - cur_share)
                    metric = "CategoryShift"
                    drift_detected = score > 0.3
                    baseline = {
                        "top_value": str(ref_counts.index[0]) if not ref_counts.empty else None,
                        "top_share": float(ref_counts.iloc[0]) if not ref_counts.empty else 0.0,
                    }
                    current = {
                        "top_value": str(cur_counts.index[0]) if not cur_counts.empty else None,
                        "top_share": float(cur_counts.iloc[0]) if not cur_counts.empty else 0.0,
                    }

                if drift_detected:
                    drifted += 1
                features.append(
                    {
                        "feature": feature,
                        "drift_score": round(float(score), 4),
                        "metric": metric,
                        "drift_detected": drift_detected,
                        "baseline": baseline,
                        "current": current,
                    }
                )

            total = len(common_cols)
            drift_share = drifted / total if total else 0.0
            result = {
                "drift_share": round(drift_share, 4),
                "number_of_columns": total,
                "number_of_drifted_columns": drifted,
                "dataset_drift": drift_share > 0.3,
                "features": features,
                "source": "prediction_events",
                "window_hours": 24,
                "sample_rows": int(len(current_df)),
            }
            self._refresh_monitoring_snapshot(model, drift_payload=result)
            return result
        except Exception as exc:
            logger.error(f"Drift calculation failed: {exc}")
            return {"error": str(exc)}

    def get_health_stats(self, model_id: str) -> dict[str, Any]:
        """Get operational health stats from real inference events."""
        model = self.db.query(MLModel).filter(MLModel.id == model_id).first()

        if not model or not model.is_deployed:
            return {
                "status": "offline",
                "latency_avg": 0.0,
                "throughput": 0,
                "error_rate": 1.0,
                "uptime": "0.00%",
                "source": "prediction_events",
                "request_count_1h": 0,
            }

        snapshot = self._refresh_monitoring_snapshot(model)
        if snapshot:
            payload = snapshot.payload or {}
            return {
                "status": payload.get("status", "healthy"),
                "latency_avg": round(float(snapshot.latency_avg_ms or 0.0), 2),
                "throughput": int(snapshot.row_count or 0),
                "error_rate": round(float(snapshot.error_rate or 0.0), 4),
                "uptime": f"{float(snapshot.uptime_pct or 0.0):.2f}%",
                "source": snapshot.source,
                "request_count_1h": int(snapshot.request_count or 0),
            }

        now = datetime.now(timezone.utc)
        hour_cutoff = now - timedelta(hours=1)
        day_cutoff = now - timedelta(days=1)
        hour_events = (
            self.db.query(PredictionEvent)
            .filter(PredictionEvent.model_id == model_id, PredictionEvent.created_at >= hour_cutoff)
            .all()
        )
        day_events = (
            self.db.query(PredictionEvent)
            .filter(PredictionEvent.model_id == model_id, PredictionEvent.created_at >= day_cutoff)
            .all()
        )

        if not day_events:
            return {
                "status": "idle",
                "latency_avg": 0.0,
                "throughput": 0,
                "error_rate": 0.0,
                "uptime": "100.00%",
                "source": "prediction_events",
                "request_count_1h": 0,
            }

        successful_day = [event for event in day_events if event.success]
        latency_values = [float(event.latency_ms) for event in successful_day if event.latency_ms is not None]
        latency_avg = float(np.mean(latency_values)) if latency_values else 0.0
        request_rows_1h = sum(int(event.request_rows or 0) for event in hour_events if event.success)
        total_events_day = len(day_events)
        failed_events_day = sum(1 for event in day_events if not event.success)
        error_rate = failed_events_day / max(total_events_day, 1)
        uptime = max(0.0, 1.0 - error_rate)

        if error_rate > 0.05 or latency_avg > 1500:
            status = "needs_attention"
        elif request_rows_1h == 0:
            status = "idle"
        else:
            status = "healthy" if model.deployment_stage == "production" else "staging"

        return {
            "status": status,
            "latency_avg": round(latency_avg, 2),
            "throughput": request_rows_1h,
            "error_rate": round(float(error_rate), 4),
            "uptime": f"{uptime * 100:.2f}%",
            "source": "prediction_events",
            "request_count_1h": int(sum(1 for event in hour_events if event.success)),
        }

    def _refresh_monitoring_snapshot(
        self,
        model: MLModel,
        drift_payload: dict[str, Any] | None = None,
    ) -> MonitoringSnapshot | None:
        resolution = int(settings.MONITORING_SNAPSHOT_RESOLUTION_MINUTES or 60)
        window_start, window_end = self._current_snapshot_window(resolution)
        events = self._get_events_for_window(model.id, window_start, window_end)
        successful_events = [event for event in events if event.success]
        latency_values = [float(event.latency_ms) for event in successful_events if event.latency_ms is not None]
        request_count = len(successful_events)
        row_count = sum(int(event.request_rows or 0) for event in successful_events)
        total_events = len(events)
        failed_events = sum(1 for event in events if not event.success)
        error_rate = failed_events / max(total_events, 1) if total_events else 0.0
        uptime_pct = max(0.0, 100.0 - (error_rate * 100.0))
        latency_avg = float(np.mean(latency_values)) if latency_values else 0.0
        latency_p95 = float(np.percentile(latency_values, 95)) if latency_values else 0.0

        if error_rate > 0.05 or latency_avg > 1500:
            status = "needs_attention"
        elif row_count == 0:
            status = "idle"
        else:
            status = "healthy" if model.deployment_stage == "production" else "staging"

        latest_snapshot = (
            self.db.query(MonitoringSnapshot)
            .filter(
                MonitoringSnapshot.model_id == model.id,
                MonitoringSnapshot.resolution_minutes == resolution,
                MonitoringSnapshot.window_start == window_start,
            )
            .first()
        )
        if latest_snapshot is None:
            latest_snapshot = MonitoringSnapshot(
                model_id=model.id,
                project_id=model.project_id,
                resolution_minutes=resolution,
                window_start=window_start,
                window_end=window_end,
                source="prediction_events",
            )
            self.db.add(latest_snapshot)

        latest_snapshot.request_count = request_count
        latest_snapshot.row_count = row_count
        latest_snapshot.latency_avg_ms = round(latency_avg, 3)
        latest_snapshot.latency_p95_ms = round(latency_p95, 3)
        latest_snapshot.error_rate = round(float(error_rate), 6)
        latest_snapshot.uptime_pct = round(float(uptime_pct), 4)
        latest_snapshot.payload = {
            **(latest_snapshot.payload or {}),
            "status": status,
            "request_count": request_count,
            "row_count": row_count,
        }

        if drift_payload:
            latest_snapshot.drift_share = float(drift_payload.get("drift_share", 0.0))
            latest_snapshot.drift_detected = bool(drift_payload.get("dataset_drift", False))
            latest_snapshot.payload = {
                **(latest_snapshot.payload or {}),
                "drift_source": drift_payload.get("source"),
                "drift_sample_rows": drift_payload.get("sample_rows"),
                "drift_features_preview": [
                    {
                        "feature": feature.get("feature"),
                        "drift_score": feature.get("drift_score"),
                        "drift_detected": feature.get("drift_detected"),
                    }
                    for feature in (drift_payload.get("features") or [])[:5]
                ],
            }

        self.db.commit()
        self.db.refresh(latest_snapshot)
        return latest_snapshot
