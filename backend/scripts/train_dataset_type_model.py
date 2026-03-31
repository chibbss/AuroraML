import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from app.services.dataset_type_classifier import (
    CLASS_NAMES,
    FEATURE_NAMES,
    DatasetTypeClassifier,
)


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def train_softmax_regression(
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    epochs: int = 400,
    lr: float = 0.2,
    reg: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    num_samples, num_features = x.shape
    weights = np.zeros((num_classes, num_features), dtype=float)
    bias = np.zeros(num_classes, dtype=float)

    for _ in range(epochs):
        scores = x @ weights.T + bias
        probs = softmax(scores)
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(num_samples), y] = 1

        grad_w = (probs - y_onehot).T @ x / num_samples + reg * weights
        grad_b = (probs - y_onehot).mean(axis=0)

        weights -= lr * grad_w
        bias -= lr * grad_b

    return weights, bias


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dataset type classifier.")
    parser.add_argument("--labels", required=True, help="CSV with columns: path,label")
    parser.add_argument("--output", required=True, help="Output JSON path for model")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--reg", type=float, default=0.01)
    args = parser.parse_args()

    label_df = pd.read_csv(args.labels)
    if "path" not in label_df.columns or "label" not in label_df.columns:
        raise ValueError("Label CSV must include 'path' and 'label' columns.")

    features = []
    labels = []
    for _, row in label_df.iterrows():
        path = str(row["path"])
        label = str(row["label"])
        if label not in CLASS_NAMES:
            raise ValueError(f"Unknown label: {label}")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext == ".json":
            df = pd.read_json(path)
        elif ext in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        feature_map = DatasetTypeClassifier.extract_features(
            df,
            template_keywords=["formula", "comment", "note", "description", "label", "cell", "sheet", "calc"],
            log_keywords=[
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
            ],
            analytics_keywords=[
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
            ],
            log_value_patterns=[
                re.compile(r"^(GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)\b", re.IGNORECASE),
                re.compile(r"\b[45]\d{2}\b"),
                re.compile(r"(exception|error|traceback|stack|failed to|timeout|unavailable|refused)", re.IGNORECASE),
            ],
        )
        features.append([feature_map.get(name, 0.0) for name in FEATURE_NAMES])
        labels.append(CLASS_NAMES.index(label))

    x = np.array(features, dtype=float)
    y = np.array(labels, dtype=int)

    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds[stds == 0] = 1.0
    x_norm = (x - means) / stds

    weights, bias = train_softmax_regression(
        x_norm,
        y,
        num_classes=len(CLASS_NAMES),
        epochs=args.epochs,
        lr=args.lr,
        reg=args.reg,
    )

    model = {
        "class_names": CLASS_NAMES,
        "feature_names": FEATURE_NAMES,
        "feature_means": means.tolist(),
        "feature_stds": stds.tolist(),
        "weights": weights.tolist(),
        "bias": bias.tolist(),
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=2)


if __name__ == "__main__":
    main()
