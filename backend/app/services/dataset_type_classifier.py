import json
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "log_rows",
    "log_cols",
    "numeric_ratio",
    "categorical_ratio",
    "datetime_ratio",
    "avg_unique_ratio",
    "median_unique_ratio",
    "high_unique_ratio_pct",
    "low_unique_ratio_pct",
    "missing_cell_ratio",
    "duplicate_ratio",
    "formula_ratio",
    "template_keyword_ratio",
    "log_keyword_ratio",
    "analytics_keyword_ratio",
    "log_value_match_ratio",
    "http_verb_ratio",
    "status_code_ratio",
    "error_value_ratio",
]


FEATURE_LABELS = {
    "log_rows": "row volume",
    "log_cols": "column count",
    "numeric_ratio": "numeric feature share",
    "categorical_ratio": "categorical feature share",
    "datetime_ratio": "datetime feature share",
    "avg_unique_ratio": "average uniqueness",
    "median_unique_ratio": "median uniqueness",
    "high_unique_ratio_pct": "high-uniqueness columns",
    "low_unique_ratio_pct": "low-uniqueness columns",
    "missing_cell_ratio": "missing cell density",
    "duplicate_ratio": "duplicate rows",
    "formula_ratio": "formula-style cells",
    "template_keyword_ratio": "template keyword hits",
    "log_keyword_ratio": "log keyword hits",
    "analytics_keyword_ratio": "analytics keyword hits",
    "log_value_match_ratio": "log-like values",
    "http_verb_ratio": "HTTP verb values",
    "status_code_ratio": "HTTP status codes",
    "error_value_ratio": "error-like values",
}


CLASS_NAMES = [
    "training_dataset",
    "logs_telemetry",
    "template_sheet",
    "metrics_report",
    "analytics_export",
    "reference_table",
]


@dataclass
class DatasetTypePrediction:
    dataset_type: str
    confidence: float
    signals: list[str]
    probabilities: dict[str, float]


class DatasetTypeClassifier:
    def __init__(self, model: dict[str, Any]):
        self.model = model
        self.class_names = model.get("class_names", CLASS_NAMES)
        self.feature_names = model.get("feature_names", FEATURE_NAMES)
        self.weights = np.array(model.get("weights", []), dtype=float)
        self.bias = np.array(model.get("bias", []), dtype=float)
        self.means = np.array(model.get("feature_means", [0.0] * len(self.feature_names)), dtype=float)
        self.stds = np.array(model.get("feature_stds", [1.0] * len(self.feature_names)), dtype=float)

    @classmethod
    def load(cls, model_path: str) -> "DatasetTypeClassifier | None":
        if not model_path or not os.path.exists(model_path):
            return None
        with open(model_path, "r", encoding="utf-8") as handle:
            model = json.load(handle)
        return cls(model)

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores)

    def predict(self, feature_vector: np.ndarray) -> DatasetTypePrediction:
        safe_stds = np.where(self.stds == 0, 1.0, self.stds)
        normalized = (feature_vector - self.means) / safe_stds
        scores = self.weights @ normalized + self.bias
        probs = self._softmax(scores)
        best_idx = int(np.argmax(probs))
        dataset_type = self.class_names[best_idx]
        confidence = float(probs[best_idx])
        contributions = self.weights[best_idx] * normalized
        top_indices = np.argsort(contributions)[::-1][:4]
        signals = [
            FEATURE_LABELS.get(self.feature_names[idx], self.feature_names[idx])
            for idx in top_indices
            if contributions[idx] > 0.05
        ]
        probabilities = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
        return DatasetTypePrediction(
            dataset_type=dataset_type,
            confidence=round(confidence, 3),
            signals=signals[:3],
            probabilities=probabilities,
        )

    @staticmethod
    def extract_features(
        df: pd.DataFrame,
        *,
        template_keywords: list[str],
        log_keywords: list[str],
        analytics_keywords: list[str],
        log_value_patterns: list[Any],
    ) -> dict[str, float]:
        row_count = len(df)
        column_count = len(df.columns)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

        numeric_ratio = len(numeric_cols) / max(column_count, 1)
        categorical_ratio = len(categorical_cols) / max(column_count, 1)
        datetime_ratio = len(datetime_cols) / max(column_count, 1)

        unique_ratios = []
        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                continue
            unique_ratios.append(series.nunique() / max(len(series), 1))
        avg_unique_ratio = float(np.mean(unique_ratios)) if unique_ratios else 0.0
        median_unique_ratio = float(np.median(unique_ratios)) if unique_ratios else 0.0
        high_unique_ratio_pct = (
            sum(1 for ratio in unique_ratios if ratio >= 0.95) / max(len(unique_ratios), 1)
            if unique_ratios
            else 0.0
        )
        low_unique_ratio_pct = (
            sum(1 for ratio in unique_ratios if ratio <= 0.05) / max(len(unique_ratios), 1)
            if unique_ratios
            else 0.0
        )

        missing_cell_ratio = float(df.isnull().sum().sum() / max(row_count * max(column_count, 1), 1))
        duplicate_ratio = float(df.duplicated().sum() / max(row_count, 1))

        formula_ratios = []
        for col in df.columns:
            series = df[col].dropna().astype(str)
            if series.empty:
                continue
            formula_ratios.append(float((series.str.startswith("=")).mean()))
        formula_ratio = float(np.mean(formula_ratios)) if formula_ratios else 0.0

        template_hits = 0
        log_hits = 0
        analytics_hits = 0
        for col in df.columns:
            normalized = col.strip().lower().replace(" ", "_").replace("-", "_")
            if any(keyword in normalized for keyword in template_keywords):
                template_hits += 1
            if any(keyword in normalized for keyword in log_keywords):
                log_hits += 1
            if any(keyword in normalized for keyword in analytics_keywords):
                analytics_hits += 1

        template_keyword_ratio = template_hits / max(column_count, 1)
        log_keyword_ratio = log_hits / max(column_count, 1)
        analytics_keyword_ratio = analytics_hits / max(column_count, 1)

        sample_values = []
        for col in categorical_cols[:10]:
            sample_values.extend(df[col].dropna().astype(str).head(200).tolist())
        sample_values = sample_values[:1000]
        if sample_values:
            log_matches = 0
            http_verb_matches = 0
            status_code_matches = 0
            error_matches = 0
            for value in sample_values:
                if any(pattern.search(value) for pattern in log_value_patterns):
                    log_matches += 1
                if re.match(r"^(GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)\b", value, re.IGNORECASE):
                    http_verb_matches += 1
                if re.search(r"\b[45]\d{2}\b", value):
                    status_code_matches += 1
                if re.search(r"(exception|error|traceback|stack|failed to|timeout|unavailable|refused)", value, re.IGNORECASE):
                    error_matches += 1
            denom = max(len(sample_values), 1)
            log_value_match_ratio = log_matches / denom
            http_verb_ratio = http_verb_matches / denom
            status_code_ratio = status_code_matches / denom
            error_value_ratio = error_matches / denom
        else:
            log_value_match_ratio = 0.0
            http_verb_ratio = 0.0
            status_code_ratio = 0.0
            error_value_ratio = 0.0

        return {
            "log_rows": np.log1p(row_count),
            "log_cols": np.log1p(column_count),
            "numeric_ratio": numeric_ratio,
            "categorical_ratio": categorical_ratio,
            "datetime_ratio": datetime_ratio,
            "avg_unique_ratio": avg_unique_ratio,
            "median_unique_ratio": median_unique_ratio,
            "high_unique_ratio_pct": high_unique_ratio_pct,
            "low_unique_ratio_pct": low_unique_ratio_pct,
            "missing_cell_ratio": missing_cell_ratio,
            "duplicate_ratio": duplicate_ratio,
            "formula_ratio": formula_ratio,
            "template_keyword_ratio": template_keyword_ratio,
            "log_keyword_ratio": log_keyword_ratio,
            "analytics_keyword_ratio": analytics_keyword_ratio,
            "log_value_match_ratio": log_value_match_ratio,
            "http_verb_ratio": http_verb_ratio,
            "status_code_ratio": status_code_ratio,
            "error_value_ratio": error_value_ratio,
        }

    @staticmethod
    def to_vector(features: dict[str, float]) -> np.ndarray:
        return np.array([float(features.get(name, 0.0)) for name in FEATURE_NAMES], dtype=float)
