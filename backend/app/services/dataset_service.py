import hashlib
import re
import pandas as pd
import numpy as np
from typing import Any

from app.core.config import settings
from app.services.dataset_type_classifier import DatasetTypeClassifier


class DatasetService:
    """Handles dataset file operations and metadata extraction."""

    @staticmethod
    def read_dataframe(file_path: str, file_type: str) -> pd.DataFrame:
        """Read a dataset file into a pandas DataFrame."""
        readers = {
            "csv": pd.read_csv,
            "parquet": pd.read_parquet,
            "json": pd.read_json,
            "xlsx": pd.read_excel,
            "xls": pd.read_excel,
        }
        reader = readers.get(file_type)
        if not reader:
            raise ValueError(f"Unsupported file type: {file_type}")
        return reader(file_path)

    @staticmethod
    def compute_checksum(file_path: str) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def get_column_info(df: pd.DataFrame) -> dict:
        """Extract column names, types, and basic stats."""
        return {
            "column_names": df.columns.tolist(),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        }

    @staticmethod
    def profile_dataset(df: pd.DataFrame) -> dict:
        """Generate enhanced profiling summary for a dataset."""
        profile = {
            "shape": list(df.shape),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "unique_counts": df.nunique().to_dict(),
            "duplicated_rows": int(df.duplicated().sum()),
            "sample_data": df.head(10).replace({float('nan'): None}).to_dict(orient="records")
        }

        # Numeric columns stats & Distributions
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            desc = df[numeric_cols].describe().to_dict()
            profile["numeric_stats"] = {
                k: {sk: round(sv, 4) if isinstance(sv, float) else sv for sk, sv in v.items()}
                for k, v in desc.items()
            }
            # Skewness
            profile["skewness"] = df[numeric_cols].skew().round(4).to_dict()
            
            # Histograms for numeric features
            profile["histograms"] = {}
            for col in numeric_cols:
                # Remove nans for histogram calculation
                clean_col = df[col].dropna()
                if not clean_col.empty:
                    counts, bins = np.histogram(clean_col, bins=10)
                    profile["histograms"][col] = [
                        {"bin": f"{bins[i]:.2f}-{bins[i+1]:.2f}", "count": int(counts[i])}
                        for i in range(len(counts))
                    ]

            # Correlation Matrix (Pearson)
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr().round(4)
                profile["correlations"] = corr_matrix.to_dict()

        # Categorical columns stats
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if cat_cols:
            profile["categorical_stats"] = {}
            for col in cat_cols:
                vc = df[col].value_counts()
                profile["categorical_stats"][col] = {
                    "unique": int(df[col].nunique()),
                    "top_values": [
                        {"label": str(k), "count": int(v)} 
                        for k, v in vc.head(10).to_dict().items()
                    ],
                    "mode": str(df[col].mode().iloc[0]) if not df[col].mode().empty else None,
                }

        return profile

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(numeric_value) or np.isinf(numeric_value):
            return None
        return numeric_value

    @staticmethod
    def _safe_label(value: str) -> str:
        return value.replace("_", " ").strip().title()

    @staticmethod
    def _normalize_column_name(column_name: str) -> str:
        return (
            column_name.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

    @staticmethod
    def _target_inference_score(series: pd.Series) -> float:
        non_null = series.dropna()
        if non_null.empty:
            return -10.0

        score = 0.0
        unique_count = int(non_null.nunique())
        unique_ratio = unique_count / max(len(non_null), 1)

        if pd.api.types.is_bool_dtype(series):
            score += 12
        elif pd.api.types.is_numeric_dtype(series):
            if unique_count <= 20:
                score += 8
            else:
                score += 3
        else:
            if 2 <= unique_count <= 20:
                score += 10
            elif unique_count <= 50:
                score += 5

        if 0.001 <= unique_ratio <= 0.4:
            score += 8
        elif unique_ratio > 0.9:
            score -= 20

        missing_ratio = float(series.isnull().mean())
        if missing_ratio > 0.3:
            score -= 8

        return score

    @staticmethod
    def recommend_target_column(df: pd.DataFrame) -> str | None:
        canonical_exact = {
            "target": 140,
            "label": 135,
            "labels": 130,
            "class": 128,
            "response": 126,
            "outcome": 126,
            "churn": 150,
            "default": 130,
            "fraud": 130,
            "survived": 130,
            "sale_price": 132,
            "price": 118,
            "y": 110,
        }
        canonical_tokens = {
            "target": 80,
            "label": 76,
            "class": 72,
            "response": 72,
            "outcome": 72,
            "churn": 95,
            "default": 72,
            "fraud": 72,
            "survived": 72,
            "price": 58,
            "status": 48,
        }
        negative_tokens = {
            "id": -80,
            "uuid": -80,
            "guid": -80,
            "date": -24,
            "time": -24,
            "timestamp": -24,
            "month": -14,
            "year": -14,
            "zip": -12,
            "code": -10,
        }

        best_column: str | None = None
        best_score = float("-inf")

        for column in df.columns:
            normalized = DatasetService._normalize_column_name(str(column))
            tokens = [token for token in normalized.split("_") if token]
            if not tokens:
                continue

            score = DatasetService._target_inference_score(df[column])

            if normalized in canonical_exact:
                score += canonical_exact[normalized]

            for token in tokens:
                score += canonical_tokens.get(token, 0)
                score += negative_tokens.get(token, 0)

            if normalized.startswith(("is_", "has_", "did_", "will_")):
                score += 14

            if tokens and tokens[-1] in canonical_tokens:
                score += 12

            if normalized.endswith(("_target", "_label", "_class", "_outcome", "_response")):
                score += 22

            if score > best_score:
                best_score = score
                best_column = str(column)

        if best_score < 45:
            return None
        return best_column

    @staticmethod
    def _infer_feature_role(series: pd.Series) -> tuple[str, float, str]:
        name = series.name.lower()
        non_null = series.dropna()
        unique_ratio = float(non_null.nunique() / max(len(non_null), 1))

        if any(token in name for token in ["id", "uuid", "guid", "key"]) and unique_ratio > 0.9:
            return ("Identifier", 0.97, "Column name and cardinality suggest row-level identifiers.")

        if np.issubdtype(series.dtype, np.datetime64) or any(token in name for token in ["date", "time", "timestamp"]):
            return ("Timestamp", 0.92, "Column appears to represent time or event ordering.")

        if series.dtype == bool:
            return ("Boolean Signal", 0.95, "Binary feature that can be directly modeled.")

        if pd.api.types.is_numeric_dtype(series):
            return ("Numeric Measure", 0.9, "Continuous or discrete numeric feature with measurable scale.")

        cardinality = int(non_null.nunique())
        if cardinality <= 20:
            return ("Categorical Feature", 0.88, "Low-cardinality feature suitable for grouping or encoding.")

        return ("Text / High Cardinality", 0.74, "High-cardinality string feature that may require special handling.")

    @staticmethod
    def _build_segments(
        df: pd.DataFrame,
        recommended_target: str | None,
        problem_type: str,
        categorical_cols: list[str],
    ) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        baseline_signal: float | None = None

        if recommended_target:
            target_series = df[recommended_target]
            if problem_type == "classification":
                encoded_target = pd.Series(pd.factorize(target_series.fillna("Missing"))[0], index=df.index)
                baseline_signal = float(encoded_target.mean()) if len(encoded_target) else None
            else:
                numeric_target = pd.to_numeric(target_series, errors="coerce").dropna()
                baseline_signal = float(numeric_target.mean()) if not numeric_target.empty else None

        candidate_features = [
            col for col in categorical_cols
            if col != recommended_target and df[col].dropna().nunique() >= 2 and df[col].dropna().nunique() <= 12
        ]

        for feature in candidate_features[:6]:
            value_counts = df[feature].fillna("Missing").astype(str).value_counts()
            for cohort, count in value_counts.head(3).items():
                sample_size = int(count)
                share_of_rows = sample_size / max(len(df), 1)
                if share_of_rows < 0.05:
                    continue

                cohort_mask = df[feature].fillna("Missing").astype(str) == str(cohort)
                target_signal: float | None = None
                comparison = "baseline unavailable"
                insight = f"{DatasetService._safe_label(feature)} = {cohort} covers {share_of_rows * 100:.1f}% of rows."

                if recommended_target and baseline_signal is not None:
                    target_slice = df.loc[cohort_mask, recommended_target]
                    if problem_type == "classification":
                        cohort_encoded = pd.Series(pd.factorize(target_slice.fillna("Missing"))[0], index=target_slice.index)
                        if len(cohort_encoded) > 0:
                            target_signal = float(cohort_encoded.mean())
                    else:
                        cohort_numeric = pd.to_numeric(target_slice, errors="coerce").dropna()
                        if not cohort_numeric.empty:
                            target_signal = float(cohort_numeric.mean())

                    if target_signal is not None:
                        delta = target_signal - baseline_signal
                        comparison = f"{delta:+.3f} vs dataset baseline"
                        insight = (
                            f"{DatasetService._safe_label(feature)} = {cohort} spans {share_of_rows * 100:.1f}% of rows "
                            f"with a target signal {delta:+.3f} away from the dataset baseline."
                        )

                segments.append(
                    {
                        "title": f"{DatasetService._safe_label(feature)}: {cohort}",
                        "feature": feature,
                        "cohort": str(cohort),
                        "sample_size": sample_size,
                        "share_of_rows": round(share_of_rows, 4),
                        "target_signal": round(target_signal, 4) if target_signal is not None else None,
                        "comparison": comparison,
                        "insight": insight,
                    }
                )

        segments.sort(key=lambda item: item["share_of_rows"], reverse=True)
        return segments[:8]

    @staticmethod
    def build_dataset_report(df: pd.DataFrame, profile: dict[str, Any]) -> dict[str, Any]:
        row_count = len(df)
        column_count = len(df.columns)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

        missing_percentage = profile.get("missing_percentage", {})
        duplicated_rows = int(profile.get("duplicated_rows", 0) or 0)
        total_missing_cells = int(df.isnull().sum().sum())
        total_cells = max(row_count * max(column_count, 1), 1)
        missing_cell_ratio = total_missing_cells / total_cells
        duplicate_ratio = duplicated_rows / max(row_count, 1)

        constant_features = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
        id_like_features = []
        for col in df.columns:
            series = df[col]
            if series.dropna().empty:
                continue
            unique_ratio = series.dropna().nunique() / max(len(series.dropna()), 1)
            if unique_ratio >= 0.95 and (series.dropna().dtype == object or series.dropna().dtype == "string"):
                id_like_features.append(col)
        high_missing_features = [col for col, pct in missing_percentage.items() if float(pct or 0) >= 25]
        high_cardinality_features = [
            col for col in categorical_cols
            if df[col].dropna().nunique() > max(20, row_count * 0.2)
        ]
        skewed_features = [
            col for col, skew in (profile.get("skewness") or {}).items()
            if DatasetService._safe_float(skew) is not None and abs(float(skew)) >= 1.0
        ]

        identifier_like_features = []
        feature_roles = []
        for col in df.columns:
            role, confidence, rationale = DatasetService._infer_feature_role(df[col])
            feature_roles.append(
                {
                    "name": col,
                    "label": DatasetService._safe_label(col),
                    "role": role,
                    "confidence": round(confidence, 2),
                    "rationale": rationale,
                }
            )
            if role == "Identifier":
                identifier_like_features.append(col)

        correlated_pairs = []
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr().abs()
            seen_pairs: set[tuple[str, str]] = set()
            for left in numeric_cols:
                for right in numeric_cols:
                    if left == right:
                        continue
                    pair = tuple(sorted((left, right)))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    corr_value = corr.loc[left, right]
                    if DatasetService._safe_float(corr_value) is not None and float(corr_value) >= 0.85:
                        correlated_pairs.append(
                            {
                                "left": pair[0],
                                "right": pair[1],
                                "strength": round(float(corr_value), 4),
                            }
                        )
            correlated_pairs.sort(key=lambda item: item["strength"], reverse=True)

        # Dataset suitability checks (avoid classifying formula/template sheets as ML data)
        formula_like_cols = []
        for col in df.columns:
            series = df[col].dropna().astype(str)
            if series.empty:
                continue
            formula_ratio = (series.str.startswith("=")).mean()
            if formula_ratio >= 0.4:
                formula_like_cols.append(col)

        too_small = row_count < 30 or column_count < 3
        template_keywords = ["formula", "comment", "note", "description", "label", "cell", "sheet", "calc"]
        template_hit = any(any(k in str(col).lower() for k in template_keywords) for col in df.columns)
        log_keywords = [
            "method",
            "endpoint",
            "path",
            "route",
            "status",
            "code",
            "error",
            "exception",
            "trace",
            "stack",
            "message",
            "occurrence",
            "occurrences",
            "latency",
            "duration",
            "ms",
            "timestamp",
            "time",
            "level",
            "severity",
            "service",
            "request",
            "response",
            "payload",
            "user_agent",
            "ip",
            "host",
        ]
        log_name_hits = []
        for col in df.columns:
            normalized = DatasetService._normalize_column_name(str(col))
            if any(keyword in normalized for keyword in log_keywords):
                log_name_hits.append(col)

        log_value_patterns = [
            re.compile(r"^(GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)\b", re.IGNORECASE),
            re.compile(r"\b[45]\d{2}\b"),
            re.compile(r"(exception|error|traceback|stack|failed to|timeout|unavailable|refused)", re.IGNORECASE),
        ]
        log_value_hits = []
        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                continue
            if series.dtype == object or series.dtype.name == "string":
                sample = series.astype(str).head(200)
                if sample.empty:
                    continue
                matches = 0
                for value in sample:
                    if any(pattern.search(value) for pattern in log_value_patterns):
                        matches += 1
                if matches / max(len(sample), 1) >= 0.2:
                    log_value_hits.append(col)

        log_like = (
            len(log_name_hits) >= 2
            or (len(log_name_hits) >= 1 and len(log_value_hits) >= 1)
            or len(log_value_hits) >= 2
        )
        analytics_keywords = [
            "metric",
            "measure",
            "kpi",
            "score",
            "value",
            "rate",
            "ratio",
            "percent",
            "percentage",
            "count",
            "total",
            "sum",
            "avg",
            "mean",
            "median",
            "min",
            "max",
            "p50",
            "p90",
            "p95",
            "p99",
            "trend",
            "window",
            "date",
            "day",
            "week",
            "month",
            "year",
        ]
        analytics_name_hits = []
        for col in df.columns:
            normalized = DatasetService._normalize_column_name(str(col))
            if any(keyword in normalized for keyword in analytics_keywords):
                analytics_name_hits.append(col)

        numeric_ratio = len(numeric_cols) / max(column_count, 1)
        categorical_ratio = len(categorical_cols) / max(column_count, 1)
        high_unique_ratio_cols = []
        low_unique_ratio_cols = []
        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                continue
            unique_ratio = series.nunique() / max(len(series), 1)
            if unique_ratio >= 0.95:
                high_unique_ratio_cols.append(col)
            if unique_ratio <= 0.05:
                low_unique_ratio_cols.append(col)

        metrics_report_like = (
            len(analytics_name_hits) >= 2
            and numeric_ratio >= 0.5
            and row_count <= 200
            and len(low_unique_ratio_cols) >= max(1, int(column_count * 0.3))
        )
        reference_table_like = (
            row_count <= 1000
            and column_count <= 6
            and numeric_ratio < 0.5
            and categorical_ratio >= 0.5
            and len(high_unique_ratio_cols) == 0
        )
        analytics_export_like = (
            len(analytics_name_hits) >= 2
            and numeric_ratio >= 0.4
            and row_count <= 2000
        )
        dataset_type_signal = []
        if template_hit or formula_like_cols:
            dataset_type_signal.append("template_sheet")
        if log_like:
            dataset_type_signal.append("logs_telemetry")
        if metrics_report_like:
            dataset_type_signal.append("metrics_report")
        if reference_table_like:
            dataset_type_signal.append("reference_table")
        if analytics_export_like:
            dataset_type_signal.append("analytics_export")
        non_dataset_flags = []
        if too_small:
            non_dataset_flags.append("Dataset is very small for reliable training.")
        if formula_like_cols:
            non_dataset_flags.append("Detected spreadsheet-style formula entries.")
        if template_hit:
            non_dataset_flags.append("Column names suggest metadata or spreadsheet template fields.")
        if log_like:
            non_dataset_flags.append("Detected log/telemetry style fields rather than modeling features.")
        if metrics_report_like:
            non_dataset_flags.append("Appears to be a KPI or metrics report rather than row-level observations.")
        if reference_table_like:
            non_dataset_flags.append("Looks like a reference/lookup table rather than a training dataset.")
        if analytics_export_like and not log_like:
            non_dataset_flags.append("Appears to be an analytics export with aggregated metrics.")

        if "template_sheet" in dataset_type_signal:
            dataset_type = "template_sheet"
        elif "logs_telemetry" in dataset_type_signal:
            dataset_type = "logs_telemetry"
        elif "metrics_report" in dataset_type_signal:
            dataset_type = "metrics_report"
        elif "analytics_export" in dataset_type_signal:
            dataset_type = "analytics_export"
        elif "reference_table" in dataset_type_signal:
            dataset_type = "reference_table"
        else:
            dataset_type = "training_dataset"

        dataset_type_confidence = 0.56
        dataset_type_signals = []
        if dataset_type == "template_sheet":
            dataset_type_confidence = 0.92 if formula_like_cols else 0.82
            dataset_type_signals = [
                "formula_style_cells" if formula_like_cols else "template_keywords",
                "small_dataset" if too_small else "metadata_columns",
            ]
        elif dataset_type == "logs_telemetry":
            dataset_type_confidence = 0.88 if len(log_value_hits) >= 1 else 0.78
            dataset_type_signals = ["log_columns", "error_pattern_values" if log_value_hits else "telemetry_fields"]
        elif dataset_type == "metrics_report":
            dataset_type_confidence = 0.78
            dataset_type_signals = ["metrics_keywords", "small_row_count", "mostly_aggregates"]
        elif dataset_type == "analytics_export":
            dataset_type_confidence = 0.72
            dataset_type_signals = ["analytics_keywords", "aggregate_metrics"]
        elif dataset_type == "reference_table":
            dataset_type_confidence = 0.7
            dataset_type_signals = ["lookup_shape", "categorical_majority"]
        else:
            dataset_type_confidence = 0.64
            dataset_type_signals = ["row_level_shape", "mixed_feature_types"]

        classifier = DatasetTypeClassifier.load(settings.DATASET_TYPE_MODEL_PATH)
        if classifier:
            features = DatasetTypeClassifier.extract_features(
                df,
                template_keywords=template_keywords,
                log_keywords=log_keywords,
                analytics_keywords=analytics_keywords,
                log_value_patterns=log_value_patterns,
            )
            prediction = classifier.predict(DatasetTypeClassifier.to_vector(features))
            if prediction.confidence >= 0.55:
                dataset_type = prediction.dataset_type
                dataset_type_confidence = prediction.confidence
                dataset_type_signals = prediction.signals

        if template_hit or formula_like_cols:
            dataset_type = "template_sheet"
            dataset_type_confidence = max(dataset_type_confidence, 0.9)
            dataset_type_signals = ["formula_style_cells" if formula_like_cols else "template_keywords"]
        elif log_like and dataset_type != "logs_telemetry" and dataset_type_confidence < 0.75:
            dataset_type = "logs_telemetry"
            dataset_type_confidence = max(dataset_type_confidence, 0.8)
            dataset_type_signals = ["log_columns", "error_pattern_values" if log_value_hits else "telemetry_fields"]

        override_type = profile.get("dataset_type_override")
        allowed_types = {
            "training_dataset",
            "logs_telemetry",
            "template_sheet",
            "metrics_report",
            "analytics_export",
            "reference_table",
        }
        if override_type in allowed_types:
            dataset_type = override_type
            dataset_type_confidence = 0.99
            dataset_type_signals = ["user_override"]
            if override_type != "training_dataset" and not any("User override" in flag for flag in non_dataset_flags):
                non_dataset_flags.append("User override applied to dataset type.")

        if dataset_type != "training_dataset" and not any("Classifier suggests" in flag for flag in non_dataset_flags):
            non_dataset_flags.append(f"Classifier suggests {dataset_type.replace('_', ' ')} data.")

        recommended_target = None
        if dataset_type == "training_dataset":
            recommended_target = DatasetService.recommend_target_column(df)

        problem_type = DatasetService.detect_problem_type(df, recommended_target) if recommended_target else "classification"

        target_analysis: dict[str, Any] = {
            "recommended_target": recommended_target,
            "problem_type": problem_type,
            "rationale": (
                f"`{recommended_target}` matched AuroraML target heuristics and fits a {problem_type} workflow."
                if recommended_target
                else (
                    "AuroraML suspects this file is not a modeling dataset, so target inference is disabled."
                    if dataset_type != "training_dataset"
                    else "No obvious target column was detected. AuroraML will need user confirmation before training."
                )
            ),
            "target_health": [],
        }

        if recommended_target:
            target_series = df[recommended_target]
            target_missing = round(float(target_series.isnull().mean() * 100), 2)
            target_analysis["target_health"].append(
                {
                    "label": "Missing Target Values",
                    "value": f"{target_missing:.2f}%",
                    "status": "warning" if target_missing > 0 else "healthy",
                }
            )
            target_analysis["target_health"].append(
                {
                    "label": "Unique Target Values",
                    "value": str(int(target_series.nunique(dropna=True))),
                    "status": "healthy",
                }
            )

            if problem_type == "classification":
                distribution = target_series.fillna("Missing").astype(str).value_counts(normalize=True).head(5)
                target_analysis["distribution"] = [
                    {"label": str(label), "share": round(float(share), 4)}
                    for label, share in distribution.items()
                ]
                dominant_share = float(distribution.iloc[0]) if not distribution.empty else 0.0
                target_analysis["imbalance_ratio"] = round(dominant_share, 4)
            else:
                numeric_target = pd.to_numeric(target_series, errors="coerce")
                target_analysis["distribution"] = [
                    {
                        "label": "Mean",
                        "share": round(float(numeric_target.mean()), 4) if not numeric_target.dropna().empty else 0.0,
                    },
                    {
                        "label": "Std Dev",
                        "share": round(float(numeric_target.std()), 4) if not numeric_target.dropna().empty else 0.0,
                    },
                ]

            target_relationships = []
            if recommended_target in numeric_cols:
                numeric_target = pd.to_numeric(df[recommended_target], errors="coerce")
            else:
                numeric_target = pd.Series(pd.factorize(df[recommended_target].fillna("Missing"))[0], index=df.index)

            for col in numeric_cols:
                if col == recommended_target:
                    continue
                paired = pd.concat([pd.to_numeric(df[col], errors="coerce"), numeric_target], axis=1).dropna()
                if len(paired) < 3:
                    continue
                corr = paired.iloc[:, 0].corr(paired.iloc[:, 1])
                corr_value = DatasetService._safe_float(corr)
                if corr_value is None:
                    continue
                target_relationships.append(
                    {
                        "feature": col,
                        "strength": round(abs(corr_value), 4),
                        "direction": "positive" if corr_value >= 0 else "negative",
                    }
                )
            target_relationships.sort(key=lambda item: item["strength"], reverse=True)
            target_analysis["top_relationships"] = target_relationships[:5]
        else:
            target_analysis["top_relationships"] = []

        segments = DatasetService._build_segments(df, recommended_target, problem_type, categorical_cols)

        findings = []
        recommendations = []

        def add_finding(title: str, detail: str, severity: str, feature: str | None = None):
            findings.append(
                {
                    "title": title,
                    "detail": detail,
                    "severity": severity,
                    "feature": feature,
                }
            )

        if duplicate_ratio > 0:
            add_finding(
                "Duplicate Records Detected",
                f"{duplicated_rows} rows ({duplicate_ratio * 100:.2f}%) are exact duplicates and can distort validation metrics.",
                "high" if duplicate_ratio > 0.05 else "medium",
            )
            recommendations.append("Review duplicate records before training to avoid inflated confidence.")

        for feature in high_missing_features[:5]:
            add_finding(
                "High Missingness",
                f"`{feature}` has {missing_percentage[feature]:.2f}% missing values and may need imputation or exclusion.",
                "high" if float(missing_percentage[feature]) >= 40 else "medium",
                feature,
            )

        for feature in identifier_like_features[:4]:
            add_finding(
                "Identifier-Like Feature",
                f"`{feature}` appears unique to each row and is likely unsuitable as a predictive signal.",
                "medium",
                feature,
            )

        for pair in correlated_pairs[:3]:
            add_finding(
                "Strong Correlation Cluster",
                f"`{pair['left']}` and `{pair['right']}` are highly correlated ({pair['strength']:.2f}), which can reduce interpretability.",
                "medium",
                pair["left"],
            )

        if not findings:
            add_finding(
                "Dataset Looks Modeling-Ready",
                "AuroraML did not detect major structural blockers in the current dataset profile.",
                "low",
            )

        if non_dataset_flags:
            add_finding(
                "Dataset Appears Non-Modeling",
                "This file looks like a spreadsheet template or operational log export rather than a training dataset. "
                "Confirm you uploaded the correct source data.",
                "high",
            )
            recommendations.append("Upload a raw dataset (rows = observations, columns = features) before training.")

        if identifier_like_features:
            recommendations.append("Exclude identifier-like columns from training unless they encode meaningful hierarchy.")
        if skewed_features:
            recommendations.append("Inspect strongly skewed numeric features for winsorization, log scaling, or robust models.")
        if correlated_pairs:
            recommendations.append("Review correlated feature groups and consider pruning redundant proxies.")
        if recommended_target:
            recommendations.append(f"Validate `{recommended_target}` as the true target before launching production training.")
        else:
            recommendations.append("Confirm the target column explicitly so the pipeline can generate target-aware diagnostics.")

        quality_score = 100.0
        quality_score -= min(missing_cell_ratio * 120, 30)
        quality_score -= min(duplicate_ratio * 180, 20)
        quality_score -= min(len(constant_features) * 2.5, 10)
        quality_score -= min(len(identifier_like_features) * 2, 10)
        quality_score -= min(len(high_cardinality_features) * 1.5, 8)
        quality_score -= min(len(id_like_features) * 2, 10)
        if too_small:
            quality_score -= 25
        if formula_like_cols:
            quality_score -= 30
        if template_hit:
            quality_score -= 15
        if log_like:
            quality_score -= 30
        if metrics_report_like:
            quality_score -= 20
        if reference_table_like:
            quality_score -= 12
        if analytics_export_like and not log_like:
            quality_score -= 15
        quality_score = max(0.0, round(quality_score, 1))

        readiness_score = quality_score
        if not recommended_target:
            readiness_score -= 12
        if len(high_missing_features) > 3:
            readiness_score -= 6
        if non_dataset_flags:
            readiness_score -= 20
        if log_like:
            readiness_score -= 8
        if metrics_report_like:
            readiness_score -= 8
        if analytics_export_like and not log_like:
            readiness_score -= 6
        readiness_score = max(0.0, round(readiness_score, 1))

        if dataset_type == "template_sheet":
            quality_score = min(quality_score, 25.0)
            readiness_score = min(readiness_score, 15.0)
        elif dataset_type == "logs_telemetry":
            quality_score = min(quality_score, 45.0)
            readiness_score = min(readiness_score, 20.0)
        elif dataset_type == "metrics_report":
            quality_score = min(quality_score, 55.0)
            readiness_score = min(readiness_score, 25.0)
        elif dataset_type == "analytics_export":
            quality_score = min(quality_score, 65.0)
            readiness_score = min(readiness_score, 35.0)
        elif dataset_type == "reference_table":
            quality_score = min(quality_score, 70.0)
            readiness_score = min(readiness_score, 40.0)

        if readiness_score >= 82:
            readiness_status = "ready"
        elif readiness_score >= 65:
            readiness_status = "review"
        else:
            readiness_status = "needs_attention"

        feature_spotlight = []
        for col in df.columns[: min(column_count, 8)]:
            missing_pct = float(missing_percentage.get(col, 0) or 0)
            unique_values = int(df[col].nunique(dropna=True))
            role_entry = next((item for item in feature_roles if item["name"] == col), None)
            note_parts = [f"{unique_values} unique values"]
            if missing_pct > 0:
                note_parts.append(f"{missing_pct:.2f}% missing")
            if col in skewed_features:
                note_parts.append("skewed distribution")
            feature_spotlight.append(
                {
                    "name": col,
                    "label": DatasetService._safe_label(col),
                    "role": role_entry["role"] if role_entry else "Feature",
                    "quality_score": round(max(0.0, 100.0 - missing_pct - (15 if col in constant_features else 0)), 1),
                    "note": ", ".join(note_parts),
                }
            )

        analyst_brief = [
            f"AuroraML detected {row_count:,} rows across {column_count} columns with a quality score of {quality_score:.1f}/100.",
            (
                f"The dataset appears aligned to a {problem_type} workflow with `{recommended_target}` as the most likely target."
                if recommended_target
                else (
                    "AuroraML flagged this file as non-modeling data, so target confirmation is paused."
                    if dataset_type != "training_dataset"
                    else "No target column is obvious, so target confirmation should be the first user action."
                )
            ),
            (
                "The strongest modeling risks come from missingness, duplicates, identifiers, and correlated numeric clusters."
                if findings
                else "No material structural risks were identified by the deterministic analyzer."
            ),
        ]
        if non_dataset_flags:
            analyst_brief.append("AuroraML suspects this file may be a template or telemetry log, not a training dataset.")

        research_report = [
            {
                "title": "Executive Summary",
                "body": (
                    f"This dataset contains {row_count:,} rows and {column_count} columns. "
                    f"AuroraML scored overall data quality at {quality_score:.1f}/100 and modeling readiness at "
                    f"{readiness_score:.1f}/100, indicating a status of {readiness_status.replace('_', ' ')}."
                ),
            },
            {
                "title": "Target & Modeling Framing",
                "body": (
                    f"The most likely target is `{recommended_target}` and the dataset currently aligns with a "
                    f"{problem_type} workflow. {target_analysis['rationale']}"
                    if recommended_target
                    else (
                        "AuroraML classified this file as non-modeling data, so supervised training should not begin until a true dataset is provided."
                        if dataset_type != "training_dataset"
                        else "AuroraML could not confidently infer a target column, so supervised training should not begin until the target is confirmed."
                    )
                ),
            },
            {
                "title": "Key Risks",
                "body": " ".join(finding["detail"] for finding in findings[:3]),
            },
            {
                "title": "Recommended Preparation Plan",
                "body": " ".join(recommendations[:3]),
            },
            {
                "title": "Cohort Signals",
                "body": (
                    "AuroraML identified notable segments: "
                    + " ".join(segment["insight"] for segment in segments[:3])
                    if segments
                    else "No strong cohort-level categorical segments met the current significance thresholds."
                ),
            },
        ]

        return {
            "overview": {
                "rows": row_count,
                "columns": column_count,
                "numeric_features": len(numeric_cols),
                "categorical_features": len(categorical_cols),
                "datetime_features": len(datetime_cols),
                "quality_score": quality_score,
                "modeling_readiness_score": readiness_score,
                "dataset_type": dataset_type,
                "dataset_type_confidence": round(dataset_type_confidence, 2),
                "dataset_type_signals": dataset_type_signals,
            },
            "quality": {
                "score": quality_score,
                "missing_cell_ratio": round(missing_cell_ratio, 4),
                "duplicate_ratio": round(duplicate_ratio, 4),
                "constant_features": constant_features[:10],
                "id_like_features": id_like_features[:10],
                "high_missing_features": high_missing_features[:10],
                "identifier_like_features": identifier_like_features[:10],
                "high_cardinality_features": high_cardinality_features[:10],
                "skewed_features": skewed_features[:10],
                "formula_like_columns": formula_like_cols[:10],
                "template_flags": non_dataset_flags,
                "non_dataset_flags": non_dataset_flags,
                "dataset_type": dataset_type,
                "dataset_type_confidence": round(dataset_type_confidence, 2),
                "dataset_type_signals": dataset_type_signals,
                "correlated_pairs": correlated_pairs[:10],
            },
            "modeling_readiness": {
                "score": readiness_score,
                "status": readiness_status,
                "summary": (
                    "Dataset is structurally ready for modeling with minor review."
                    if readiness_status == "ready"
                    else "Dataset is trainable but requires review of highlighted data risks."
                    if readiness_status == "review"
                    else "Dataset needs remediation before a production-grade training run."
                ),
            },
            "target_analysis": target_analysis,
            "segments": segments,
            "feature_roles": feature_roles[:20],
            "feature_spotlight": feature_spotlight,
            "findings": findings[:8],
            "recommendations": recommendations[:6],
            "analyst_brief": analyst_brief,
            "research_report": research_report,
        }

    @staticmethod
    def detect_problem_type(df: pd.DataFrame, target_column: str) -> str:
        """Auto-detect whether the problem is classification or regression."""
        target = df[target_column]
        if target.dtype == "object" or target.dtype.name == "category":
            return "classification"
        unique_ratio = target.nunique() / len(target)
        if unique_ratio < 0.05 or target.nunique() <= 20:
            return "classification"
        return "regression"
