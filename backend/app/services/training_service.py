"""
Training Service — Full ML pipeline: cleaning, feature engineering, training, and tuning.
This is the core engine of AuroraML.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    KFold,
    GroupKFold,
    GroupShuffleSplit,
    TimeSeriesSplit,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    log_loss,
    average_precision_score,
)

# Models
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = LGBMRegressor = None

try:
    import optuna
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sqlalchemy.orm import Session
from app.models.job import Job
from app.models.dataset import Dataset
from app.models.ml_model import MLModel
from app.models.project import Project
from app.models.notification import Notification
from app.core.config import settings
from app.services.model_artifact_service import save_model_artifact

logger = logging.getLogger(__name__)


# ─── Model Registry ──────────────────────────────────────────────────────────

CLASSIFIERS = {
    "random_forest": {
        "class": RandomForestClassifier,
        "framework": "sklearn",
        "default_params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "tuning_params": {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int", 3, 30),
            "min_samples_split": ("int", 2, 20),
            "min_samples_leaf": ("int", 1, 10),
        },
    },
    "gradient_boosting": {
        "class": GradientBoostingClassifier,
        "framework": "sklearn",
        "default_params": {"n_estimators": 100, "random_state": 42},
        "tuning_params": {
            "n_estimators": ("int", 50, 300),
            "max_depth": ("int", 3, 15),
            "learning_rate": ("float", 0.01, 0.3),
            "subsample": ("float", 0.6, 1.0),
        },
    },
    "logistic_regression": {
        "class": LogisticRegression,
        "framework": "sklearn",
        "default_params": {"max_iter": 1000, "random_state": 42},
        "tuning_params": {
            "C": ("float_log", 0.001, 100),
        },
    },
    "svm": {
        "class": SVC,
        "framework": "sklearn",
        "default_params": {"probability": True, "random_state": 42},
        "tuning_params": {
            "C": ("float_log", 0.01, 100),
            "kernel": ("categorical", ["rbf", "linear"]),
        },
    },
}

REGRESSORS = {
    "random_forest": {
        "class": RandomForestRegressor,
        "framework": "sklearn",
        "default_params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "tuning_params": {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int", 3, 30),
            "min_samples_split": ("int", 2, 20),
        },
    },
    "gradient_boosting": {
        "class": GradientBoostingRegressor,
        "framework": "sklearn",
        "default_params": {"n_estimators": 100, "random_state": 42},
        "tuning_params": {
            "n_estimators": ("int", 50, 300),
            "max_depth": ("int", 3, 15),
            "learning_rate": ("float", 0.01, 0.3),
        },
    },
    "ridge": {
        "class": Ridge,
        "framework": "sklearn",
        "default_params": {"random_state": 42},
        "tuning_params": {"alpha": ("float_log", 0.001, 100)},
    },
    "lasso": {
        "class": Lasso,
        "framework": "sklearn",
        "default_params": {"random_state": 42},
        "tuning_params": {"alpha": ("float_log", 0.001, 100)},
    },
}

# Add XGBoost if available
if XGBClassifier:
    CLASSIFIERS["xgboost"] = {
        "class": XGBClassifier,
        "framework": "xgboost",
        "default_params": {
            "n_estimators": 100,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "verbosity": 0,
        },
        "tuning_params": {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int", 3, 15),
            "learning_rate": ("float", 0.01, 0.3),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.5, 1.0),
        },
    }
if XGBRegressor:
    REGRESSORS["xgboost"] = {
        "class": XGBRegressor,
        "framework": "xgboost",
        "default_params": {"n_estimators": 100, "random_state": 42, "verbosity": 0},
        "tuning_params": {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int", 3, 15),
            "learning_rate": ("float", 0.01, 0.3),
            "subsample": ("float", 0.6, 1.0),
        },
    }

# Add LightGBM if available
if LGBMClassifier:
    CLASSIFIERS["lightgbm"] = {
        "class": LGBMClassifier,
        "framework": "lightgbm",
        "default_params": {"n_estimators": 100, "random_state": 42, "verbose": -1},
        "tuning_params": {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int", 3, 15),
            "learning_rate": ("float", 0.01, 0.3),
            "num_leaves": ("int", 15, 127),
            "subsample": ("float", 0.6, 1.0),
        },
    }
if LGBMRegressor:
    REGRESSORS["lightgbm"] = {
        "class": LGBMRegressor,
        "framework": "lightgbm",
        "default_params": {"n_estimators": 100, "random_state": 42, "verbose": -1},
        "tuning_params": {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int", 3, 15),
            "learning_rate": ("float", 0.01, 0.3),
            "num_leaves": ("int", 15, 127),
        },
    }


# ─── Data Cleaning ───────────────────────────────────────────────────────────


class DataCleaner:
    """Automated data cleaning pipeline."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    def clean(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Run only leakage-safe cleanup before train/test split."""
        df = df.copy()

        df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
        target_column_clean = target_column.strip().lower().replace(" ", "_").replace("-", "_")

        n_dups = df.duplicated().sum()
        if n_dups > 0:
            logger.info(f"Removing {n_dups} duplicate rows")
            df = df.drop_duplicates()

        if target_column_clean in df.columns:
            df = df.dropna(subset=[target_column_clean])

        return df, target_column_clean


# ─── Feature Engineering ─────────────────────────────────────────────────────


class FeatureEngineer:
    """Automated feature engineering."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    def engineer(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Apply feature engineering based on config."""
        df = df.copy()

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)

        # Log transform for skewed features
        if self.config.get("advanced_transformations", True):
            for col in numeric_cols:
                skewness = df[col].skew()
                if abs(skewness) > 1.0 and (df[col] > 0).all():
                    df[f"{col}_log"] = np.log1p(df[col])

        # Interaction features
        if self.config.get("interaction_features", False) and len(numeric_cols) >= 2:
            # Create interaction features for top correlated pairs
            if len(numeric_cols) <= 10:
                for i in range(min(len(numeric_cols), 5)):
                    for j in range(i + 1, min(len(numeric_cols), 5)):
                        c1, c2 = numeric_cols[i], numeric_cols[j]
                        df[f"{c1}_x_{c2}"] = df[c1] * df[c2]

        return df


class DataFrameCleanerTransformer(BaseEstimator, TransformerMixin):
    """Fit column-retention rules on training data and reuse them consistently."""

    def __init__(self, missing_threshold: float = 0.5, variance_threshold: float = 0.01):
        self.missing_threshold = missing_threshold
        self.variance_threshold = variance_threshold
        self.columns_to_keep_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        frame = X.copy()
        missing_ratio = frame.isnull().mean()
        keep_cols = [
            col for col in frame.columns
            if float(missing_ratio.get(col, 0.0)) <= float(self.missing_threshold)
        ]

        if self.variance_threshold > 0:
            numeric_cols = frame[keep_cols].select_dtypes(include=["number"]).columns.tolist()
            low_variance_cols = []
            for col in numeric_cols:
                series = pd.to_numeric(frame[col], errors="coerce").dropna()
                if series.empty or float(series.std()) < float(self.variance_threshold):
                    low_variance_cols.append(col)
            keep_cols = [col for col in keep_cols if col not in low_variance_cols]

        self.columns_to_keep_ = keep_cols
        return self

    def transform(self, X: pd.DataFrame):
        frame = X.copy()
        if not self.columns_to_keep_:
            return frame
        return frame.reindex(columns=self.columns_to_keep_)


class DataFrameFeatureEngineer(BaseEstimator, TransformerMixin):
    """Train-fitted feature engineering to avoid leakage into the holdout split."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.log_features_: list[str] = []
        self.interaction_pairs_: list[tuple[str, str]] = []

    def fit(self, X: pd.DataFrame, y=None):
        frame = X.copy()
        numeric_cols = frame.select_dtypes(include=["number"]).columns.tolist()

        if self.config.get("advanced_transformations", True):
            for col in numeric_cols:
                series = pd.to_numeric(frame[col], errors="coerce").dropna()
                if not series.empty and abs(float(series.skew())) > 1.0 and bool((series > 0).all()):
                    self.log_features_.append(col)

        if self.config.get("interaction_features", False) and len(numeric_cols) >= 2:
            candidate_cols = numeric_cols[: min(len(numeric_cols), 5)]
            corr = frame[candidate_cols].corr().abs().fillna(0)
            ranked_pairs: list[tuple[str, str, float]] = []
            for idx, left in enumerate(candidate_cols):
                for right in candidate_cols[idx + 1 :]:
                    ranked_pairs.append((left, right, float(corr.loc[left, right])))
            ranked_pairs.sort(key=lambda item: item[2], reverse=True)
            self.interaction_pairs_ = [(left, right) for left, right, _ in ranked_pairs[:3]]

        return self

    def transform(self, X: pd.DataFrame):
        frame = X.copy()
        for col in self.log_features_:
            if col not in frame.columns:
                continue
            series = pd.to_numeric(frame[col], errors="coerce")
            frame[f"{col}_log"] = np.where(series > 0, np.log1p(series), np.nan)

        for left, right in self.interaction_pairs_:
            if left not in frame.columns or right not in frame.columns:
                continue
            frame[f"{left}_x_{right}"] = pd.to_numeric(frame[left], errors="coerce") * pd.to_numeric(
                frame[right], errors="coerce"
            )

        return frame


# ─── Training Service ────────────────────────────────────────────────────────


class TrainingService:
    """Orchestrates the full training pipeline."""

    def __init__(self, db: Session):
        self.db = db

    @staticmethod
    def _normalize_column_name(column_name: str | None) -> str | None:
        if not column_name:
            return None
        return column_name.strip().lower().replace(" ", "_").replace("-", "_")

    def run_training_job(self, job_id: str) -> None:
        """Execute a complete training job."""
        job = self.db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        try:
            # Update status
            job.status = "running"
            job.started_at = datetime.now(timezone.utc)
            self.db.commit()

            # Load dataset
            dataset = self.db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
            if not dataset:
                raise ValueError("Dataset not found")

            df = self._load_dataset(dataset)
            logger.info(f"Loaded dataset: {df.shape}")

            # Clean data
            cleaner = DataCleaner(job.config.get("data_cleaning", {}))
            df, target_col = cleaner.clean(df, job.target_column)
            logger.info(f"After cleaning: {df.shape}")

            # Prepare X, y
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Encode target for classification
            label_encoder = None
            if job.problem_type == "classification":
                label_encoder = LabelEncoder()
                y = pd.Series(label_encoder.fit_transform(y), name=target_col)
            class_labels = (
                [str(label) for label in label_encoder.classes_]
                if label_encoder is not None
                else [str(label) for label in sorted(pd.Series(y).dropna().unique().tolist())]
                if job.problem_type == "classification"
                else []
            )

            data_cleaning_config = job.config.get("data_cleaning", {})
            fe_config = job.config.get("feature_engineering", {})
            validation_config = job.config.get("validation", {})
            schema_cleaner = DataFrameCleanerTransformer(
                missing_threshold=data_cleaning_config.get("missing_threshold", 0.5),
                variance_threshold=data_cleaning_config.get("variance_threshold", 0.01),
            )
            engineer = DataFrameFeatureEngineer(fe_config)

            transformers = []
            transformers.append(
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]), make_column_selector(dtype_include=np.number))
            )
            transformers.append(
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]), make_column_selector(dtype_include=["object", "category", "bool"]))
            )

            preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

            # Split data
            X_train, X_test, y_train, y_test, validation_metadata, groups_train = self._safe_train_test_split(
                X,
                y,
                problem_type=job.problem_type,
                validation_config=validation_config,
                random_state=42,
            )

            # Train models
            model_registry = CLASSIFIERS if job.problem_type == "classification" else REGRESSORS
            model_types = job.model_types or list(model_registry.keys())

            all_results = {}
            best_score = -np.inf
            best_model_type = None
            best_pipeline = None
            best_metrics = None

            for model_type in model_types:
                if model_type not in model_registry:
                    logger.warning(f"Unknown model type: {model_type}, skipping")
                    continue

                model_info = model_registry[model_type]
                logger.info(f"Training {model_type}...")

                try:
                    # Hyperparameter tuning with Optuna
                    if OPTUNA_AVAILABLE:
                        best_params, cv_metadata = self._tune_hyperparameters(
                            model_info,
                            schema_cleaner,
                            engineer,
                            preprocessor,
                            X_train,
                            y_train,
                            job.problem_type,
                            validation_metadata=validation_metadata,
                            groups_train=groups_train,
                            n_trials=job.config.get("auto_ml", {}).get("max_trials", 20),
                            cv_folds=job.config.get("auto_ml", {}).get("cv_folds", 3),
                        )
                    else:
                        best_params = model_info["default_params"]
                        cv_metadata = None

                    # Build final pipeline with best params
                    model = model_info["class"](**best_params)
                    pipeline = self._build_model_pipeline(schema_cleaner, engineer, preprocessor, model)

                    # Fit
                    pipeline.fit(X_train, y_train)

                    # Evaluate
                    metrics = self._evaluate_model(
                        pipeline, X_test, y_test, job.problem_type
                    )
                    metrics["hyperparameters"] = best_params
                    metrics["validation_strategy"] = {
                        **validation_metadata,
                        "type": f"{validation_metadata.get('type', 'holdout')}_plus_cv",
                        "cv_folds": job.config.get("auto_ml", {}).get("cv_folds", 3),
                    }
                    if cv_metadata:
                        metrics["cross_validation"] = cv_metadata
                    if class_labels:
                        metrics["class_labels"] = class_labels
                    metrics["lineage"] = {
                        "dataset_id": dataset.id,
                        "dataset_checksum": dataset.checksum,
                        "dataset_version": dataset.version,
                        "dataset_rows": len(df),
                        "dataset_columns": len(X.columns),
                        "target_column": target_col,
                        "problem_type": job.problem_type,
                        "train_rows": int(len(X_train)),
                        "test_rows": int(len(X_test)),
                    }

                    # Determine score (primary metric)
                    primary_score = (
                        metrics.get("f1_score", 0)
                        if job.problem_type == "classification"
                        else metrics.get("r2_score", -np.inf)
                    )

                    all_results[model_type] = metrics

                    if primary_score > best_score:
                        best_score = primary_score
                        best_model_type = model_type
                        best_pipeline = pipeline
                        best_metrics = metrics

                    logger.info(f"{model_type} score: {primary_score:.4f}")

                except Exception as e:
                    logger.error(f"Failed to train {model_type}: {e}")
                    all_results[model_type] = {"error": str(e)}

                # [ADDED FOR UI DASHBOARD POLLING] Update DB progressively per model
                from sqlalchemy.orm.attributes import flag_modified
                
                # Update job progress dict safely
                current_config = dict(job.config) if job.config else {}
                current_config["progress"] = {
                    "completed": list(model_types).index(model_type) + 1,
                    "total": len(model_types),
                    "current": model_type,
                    "best_score": float(best_score) if best_score != -np.inf else 0.0,
                    "best_model": best_model_type
                }
                
                job.config = current_config
                job.all_results = dict(all_results)
                job.metrics = all_results.get(best_model_type, {})
                
                flag_modified(job, "config")
                flag_modified(job, "all_results")
                flag_modified(job, "metrics")
                self.db.commit()

            if best_pipeline is None:
                raise ValueError("All models failed to train")

            # Save best model
            model_dir = os.path.join(settings.LOCAL_STORAGE_PATH, "models", job.project_id)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{job.id}_{best_model_type}.joblib")
            artifact_metadata = {
                "class_labels": class_labels,
                "target_column": target_col,
                "problem_type": job.problem_type,
                "dataset_id": dataset.id,
                "dataset_checksum": dataset.checksum,
                "dataset_version": dataset.version,
                "feature_columns": X.columns.tolist(),
                "feature_schema": {col: str(dtype) for col, dtype in X.dtypes.items()},
                "training_manifest": best_metrics.get("lineage", {}) if best_metrics else {},
                "validation_strategy": best_metrics.get("validation_strategy", {}) if best_metrics else {},
                "cross_validation": best_metrics.get("cross_validation", {}) if best_metrics else {},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            save_model_artifact(model_path, best_pipeline, artifact_metadata)

            # Feature importance
            feature_importance = self._get_feature_importance(best_pipeline, X.columns.tolist())

            # Create MLModel record
            ml_model = MLModel(
                job_id=job.id,
                project_id=job.project_id,
                name=f"{best_model_type}_{job.id[:8]}",
                model_type=best_model_type,
                framework=model_registry[best_model_type]["framework"],
                file_path=model_path,
                metrics=best_metrics or all_results.get(best_model_type, {}),
                hyperparameters=all_results.get(best_model_type, {}).get("hyperparameters", {}),
                feature_importance=feature_importance,
            )
            self.db.add(ml_model)
            self.db.flush()

            # Update job
            job.status = "completed"
            job.best_model_type = best_model_type
            job.best_model_id = ml_model.id
            job.best_score = best_score
            job.metrics = best_metrics or all_results.get(best_model_type, {})
            job.all_results = all_results
            job.completed_at = datetime.now(timezone.utc)
            self.db.commit()
            self._create_job_notification(job, status="completed")

            logger.info(
                f"Job {job_id} completed. Best: {best_model_type} ({best_score:.4f})"
            )

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc)
            self.db.commit()
            self._create_job_notification(job, status="failed")

    def _build_model_pipeline(
        self,
        schema_cleaner: DataFrameCleanerTransformer,
        engineer: DataFrameFeatureEngineer,
        preprocessor: ColumnTransformer,
        model,
    ) -> Pipeline:
        return Pipeline([
            ("schema_cleaner", schema_cleaner),
            ("feature_engineer", engineer),
            ("preprocessor", preprocessor),
            ("model", model),
        ])

    def _safe_train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
        validation_config: Optional[dict] = None,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, Any], Optional[pd.Series]]:
        validation_config = validation_config or {}
        requested_strategy = str(validation_config.get("strategy", "auto")).lower()
        time_column = self._normalize_column_name(validation_config.get("time_column"))
        group_column = self._normalize_column_name(validation_config.get("group_column"))
        test_size = float(validation_config.get("test_size", 0.2))
        test_size = min(max(test_size, 0.1), 0.4)

        if requested_strategy == "auto":
            if time_column and time_column in X.columns:
                strategy = "temporal"
            elif group_column and group_column in X.columns:
                strategy = "group"
            elif problem_type == "classification":
                strategy = "stratified"
            else:
                strategy = "random"
        else:
            strategy = requested_strategy

        metadata: dict[str, Any] = {
            "requested_type": requested_strategy,
            "type": strategy,
            "holdout_fraction": test_size,
            "stratified": False,
            "time_column": time_column,
            "group_column": group_column,
        }

        if strategy == "temporal" and time_column and time_column in X.columns:
            time_values = pd.to_datetime(X[time_column], errors="coerce")
            valid_time = time_values.notna()
            if valid_time.sum() >= max(10, int(len(X) * 0.6)):
                ordered_index = time_values.fillna(pd.Timestamp.min).sort_values().index
                split_idx = max(1, min(len(ordered_index) - 1, int(len(ordered_index) * (1 - test_size))))
                train_idx = ordered_index[:split_idx]
                test_idx = ordered_index[split_idx:]
                metadata["type"] = "temporal"
                return (
                    X.loc[train_idx].copy(),
                    X.loc[test_idx].copy(),
                    y.loc[train_idx].copy(),
                    y.loc[test_idx].copy(),
                    metadata,
                    None,
                )
            strategy = "stratified" if problem_type == "classification" else "random"
            metadata["type"] = strategy

        if strategy == "group" and group_column and group_column in X.columns:
            groups = X[group_column].fillna("__missing__").astype(str)
            if groups.nunique() >= 2:
                splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
                train_idx, test_idx = next(splitter.split(X, y, groups=groups))
                metadata["type"] = "group"
                return (
                    X.iloc[train_idx].copy(),
                    X.iloc[test_idx].copy(),
                    y.iloc[train_idx].copy(),
                    y.iloc[test_idx].copy(),
                    metadata,
                    groups.iloc[train_idx].copy(),
                )
            strategy = "stratified" if problem_type == "classification" else "random"
            metadata["type"] = strategy

        stratify = None
        if strategy == "stratified" and problem_type == "classification":
            value_counts = pd.Series(y).value_counts(dropna=False)
            min_count = int(value_counts.min()) if not value_counts.empty else 0
            if len(value_counts) > 1 and min_count >= 2:
                stratify = y
                metadata["stratified"] = True

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )
            metadata["stratified"] = False
            if metadata["type"] == "stratified":
                metadata["type"] = "random"

        return X_train, X_test, y_train, y_test, metadata, None

    def _load_dataset(self, dataset: Dataset) -> pd.DataFrame:
        """Load dataset from file."""
        from app.services.dataset_service import DatasetService
        return DatasetService.read_dataframe(dataset.file_path, dataset.file_type)

    def _create_job_notification(self, job: Job, status: str) -> None:
        project = self.db.query(Project).filter(Project.id == job.project_id).first()
        if not project:
            return
        title = "Training Completed" if status == "completed" else "Training Failed"
        message = (
            f"AutoML training for project '{project.name}' completed successfully. "
            f"Best model: {job.best_model_type or 'N/A'}."
            if status == "completed"
            else f"AutoML training for project '{project.name}' failed. {job.error_message or ''}"
        ).strip()
        notification = Notification(
            user_id=project.owner_id,
            project_id=project.id,
            job_id=job.id,
            notification_type=status,
            title=title,
            message=message,
        )
        self.db.add(notification)
        self.db.commit()

    def _tune_hyperparameters(
        self,
        model_info: dict,
        schema_cleaner: DataFrameCleanerTransformer,
        engineer: DataFrameFeatureEngineer,
        preprocessor: ColumnTransformer,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        problem_type: str,
        validation_metadata: Optional[dict[str, Any]] = None,
        groups_train: Optional[pd.Series] = None,
        n_trials: int = 20,
        cv_folds: int = 3,
    ) -> tuple[dict, dict]:
        """Use Optuna to tune hyperparameters."""
        tuning_params = model_info["tuning_params"]
        default_params = model_info["default_params"].copy()
        scoring = "f1_weighted" if problem_type == "classification" else "r2"
        validation_metadata = validation_metadata or {}
        split_type = validation_metadata.get("type", "random")
        if split_type == "temporal":
            cv_folds = max(2, min(cv_folds, max(2, len(X_train) - 1)))
            cv_strategy = TimeSeriesSplit(n_splits=cv_folds)
            cv_label = "time_series_split"
            cv_groups = None
        elif split_type == "group" and groups_train is not None and groups_train.nunique() >= 2:
            cv_folds = max(2, min(cv_folds, int(groups_train.nunique())))
            cv_strategy = GroupKFold(n_splits=cv_folds)
            cv_label = "group_kfold"
            cv_groups = groups_train
        elif problem_type == "classification":
            cv_folds = max(2, min(cv_folds, len(X_train)))
            cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_label = "stratified_kfold"
            cv_groups = None
        else:
            cv_folds = max(2, min(cv_folds, len(X_train)))
            cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_label = "kfold"
            cv_groups = None

        def objective(trial):
            params = default_params.copy()
            for param_name, param_spec in tuning_params.items():
                param_type = param_spec[0]
                if param_type == "int":
                    params[param_name] = trial.suggest_int(param_name, param_spec[1], param_spec[2])
                elif param_type == "float":
                    params[param_name] = trial.suggest_float(param_name, param_spec[1], param_spec[2])
                elif param_type == "float_log":
                    params[param_name] = trial.suggest_float(param_name, param_spec[1], param_spec[2], log=True)
                elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, param_spec[1])

            model = model_info["class"](**params)
            pipeline = self._build_model_pipeline(schema_cleaner, engineer, preprocessor, model)
            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv_strategy,
                scoring=scoring,
                n_jobs=-1,
                groups=cv_groups,
            )
            return scores.mean()

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = default_params.copy()
        best_params.update(study.best_params)
        return best_params, {
            "metric": scoring,
            "folds": cv_folds,
            "mean_score": round(float(study.best_value), 4),
            "strategy": cv_label,
        }

    def _evaluate_model(
        self,
        pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        problem_type: str,
    ) -> dict:
        """Evaluate model on test set."""
        y_pred = pipeline.predict(X_test)
        metrics = {}

        if problem_type == "classification":
            metrics["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
            metrics["balanced_accuracy"] = round(balanced_accuracy_score(y_test, y_pred), 4)
            metrics["matthews_corrcoef"] = round(matthews_corrcoef(y_test, y_pred), 4)
            avg = "weighted" if len(np.unique(y_test)) > 2 else "binary"
            metrics["precision"] = round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4)
            metrics["recall"] = round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4)
            metrics["f1_score"] = round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4)
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            classes = [str(c) for c in np.unique(y_test)]
            metrics["confusion_matrix"] = {
                "labels": classes,
                "matrix": cm.tolist(),
                "values": [
                    {"actual": classes[i], "predicted": classes[j], "count": int(cm[i, j])}
                    for i in range(len(classes)) for j in range(len(classes))
                ]
            }
            
            try:
                if hasattr(pipeline, "predict_proba"):
                    y_proba = pipeline.predict_proba(X_test)
                    metrics["log_loss"] = round(log_loss(y_test, y_proba), 4)
                    if y_proba.shape[1] == 2:
                        metrics["roc_auc"] = round(roc_auc_score(y_test, y_proba[:, 1]), 4)
                        metrics["average_precision"] = round(average_precision_score(y_test, y_proba[:, 1]), 4)
                    else:
                        metrics["roc_auc"] = round(
                            roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"), 4
                        )
            except Exception:
                pass
        else:
            metrics["mae"] = round(mean_absolute_error(y_test, y_pred), 4)
            metrics["rmse"] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
            metrics["r2_score"] = round(r2_score(y_test, y_pred), 4)

            # Residuals (sample for plotting)
            residuals = y_test - y_pred
            sample_size = min(len(y_test), 500)
            indices = np.random.choice(len(y_test), sample_size, replace=False)
            metrics["residuals"] = [
                {"actual": float(y_test.iloc[i]), "predicted": float(y_pred[i]), "residual": float(residuals.iloc[i])}
                for i in indices
            ]

        return metrics

    def _get_feature_importance(self, pipeline: Pipeline, feature_names: list) -> Optional[dict]:
        """Extract feature importance from the model if available."""
        try:
            model = pipeline.named_steps["model"]
            if hasattr(model, "feature_importances_"):
                # Get transformed feature names
                preprocessor = pipeline.named_steps["preprocessor"]
                try:
                    transformed_names = preprocessor.get_feature_names_out()
                except Exception:
                    transformed_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]

                importance = dict(zip(
                    [str(n) for n in transformed_names],
                    [round(float(v), 6) for v in model.feature_importances_],
                ))
                # Sort by importance
                importance = dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))
                return dict(list(importance.items())[:30])  # Top 30
            elif hasattr(model, "coef_"):
                preprocessor = pipeline.named_steps["preprocessor"]
                try:
                    transformed_names = preprocessor.get_feature_names_out()
                except Exception:
                    coef = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
                    transformed_names = [f"feature_{i}" for i in range(len(coef))]

                coef = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
                importance = dict(zip(
                    [str(n) for n in transformed_names],
                    [round(float(v), 6) for v in coef],
                ))
                importance = dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))
                return dict(list(importance.items())[:30])
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
        return None
