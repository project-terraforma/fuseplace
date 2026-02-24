"""Train/apply an ML baseline for attribute-level source selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from scripts.utils.conflation import CORE_ATTRIBUTES, feature_vector, proxy_label
from scripts.utils.io import ANALYSIS_DIR, DEFAULT_DATA_PATH, PROJECT_ROOT, REPORTS_DIR, ensure_dir, load_parquet_duckdb, write_csv, write_json


def _safe_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _load_manual_labels(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    labels: dict[tuple[str, str], str] = {}
    for row in payload:
        row_id = str(row.get("id", ""))
        attr_labels = row.get("labels", {}) or {}
        for attr, label in attr_labels.items():
            normalized = str(label).strip().lower()
            if normalized in {"current", "base", "skip", "tie"}:
                labels[(row_id, attr)] = normalized

    return labels


def _build_training_table(
    df: pd.DataFrame,
    attrs: list[str],
    manual_labels: dict[tuple[str, str], str],
    use_proxy_labels: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rows: list[dict] = []
    labels: list[int] = []
    meta_rows: list[dict] = []

    for _, record in df.iterrows():
        record_id = str(record.get("id", ""))
        confidence = _safe_float(record.get("confidence"))
        base_confidence = _safe_float(record.get("base_confidence"))

        for attr in attrs:
            features = feature_vector(
                attr=attr,
                current_value=record.get(attr),
                base_value=record.get(f"base_{attr}"),
                confidence=confidence,
                base_confidence=base_confidence,
                current_sources=record.get("sources"),
                base_sources=record.get("base_sources"),
            )

            # Add one-hot-ish attribute indicator features for a shared model.
            for name in attrs:
                features[f"attr_{name}"] = 1.0 if name == attr else 0.0

            manual = manual_labels.get((record_id, attr))
            if manual in {"current", "base"}:
                label = manual
                label_source = "manual"
            elif use_proxy_labels:
                label = proxy_label(
                    attr=attr,
                    current_value=record.get(attr),
                    base_value=record.get(f"base_{attr}"),
                    confidence=confidence,
                    base_confidence=base_confidence,
                )
                label_source = "proxy"
            else:
                continue

            if label not in {"current", "base"}:
                continue

            rows.append(features)
            labels.append(1 if label == "current" else 0)
            meta_rows.append({"id": record_id, "attribute": attr, "label_source": label_source})

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="target")
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


def _fit_model(X: pd.DataFrame, y: pd.Series) -> tuple[LogisticRegression, dict]:
    if y.nunique() < 2:
        raise RuntimeError("Need both classes (current/base) in training data for ML model.")

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=["base", "current"], output_dict=True)

    metrics = {
        "accuracy": float(accuracy),
        "support_test": int(len(y_test)),
        "classification_report": report,
    }
    return model, metrics


def _predict_decisions(
    model: LogisticRegression,
    feature_columns: list[str],
    df: pd.DataFrame,
    attrs: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision_rows: list[dict] = []
    selected_rows: list[dict] = []

    for _, record in df.iterrows():
        confidence = _safe_float(record.get("confidence"))
        base_confidence = _safe_float(record.get("base_confidence"))

        selected_record = {
            "id": record.get("id"),
            "base_id": record.get("base_id"),
        }

        for attr in attrs:
            feat = feature_vector(
                attr=attr,
                current_value=record.get(attr),
                base_value=record.get(f"base_{attr}"),
                confidence=confidence,
                base_confidence=base_confidence,
                current_sources=record.get("sources"),
                base_sources=record.get("base_sources"),
            )
            for name in attrs:
                feat[f"attr_{name}"] = 1.0 if name == attr else 0.0

            frame = pd.DataFrame([feat]).reindex(columns=feature_columns, fill_value=0.0)
            pred = int(model.predict(frame)[0])
            prob_current = float(model.predict_proba(frame)[0][1])
            winner = "current" if pred == 1 else "base"

            if winner == "current":
                selected_value = record.get(attr)
            else:
                selected_value = record.get(f"base_{attr}")

            selected_record[f"selected_{attr}"] = selected_value
            selected_record[f"selected_from_{attr}"] = winner

            decision_rows.append(
                {
                    "id": record.get("id"),
                    "base_id": record.get("base_id"),
                    "attribute": attr,
                    "winner": winner,
                    "prob_current": prob_current,
                }
            )

        selected_rows.append(selected_record)

    return pd.DataFrame(decision_rows), pd.DataFrame(selected_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/apply ML baseline for attribute selection")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--golden-labels",
        type=Path,
        default=ANALYSIS_DIR / "golden" / "golden_dataset_template.json",
        help="JSON with labels[attr] = current/base/skip/tie",
    )
    parser.add_argument("--no-proxy", action="store_true", help="Disable weak proxy labels")
    parser.add_argument("--model-out", type=Path, default=PROJECT_ROOT / "models" / "ml_selector.joblib")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR / "conflation")
    args = parser.parse_args()

    df = load_parquet_duckdb(args.input)
    attrs = [a for a in CORE_ATTRIBUTES if a in df.columns and f"base_{a}" in df.columns]

    manual_labels = _load_manual_labels(args.golden_labels)
    X, y, meta = _build_training_table(df, attrs, manual_labels, use_proxy_labels=not args.no_proxy)

    if X.empty:
        raise RuntimeError("No training rows available. Add manual labels or enable proxy labels.")

    model, metrics = _fit_model(X, y)

    ensure_dir(args.model_out.parent)
    artifact_payload = {
        "model": model,
        "feature_columns": list(X.columns),
        "attributes": attrs,
        "metrics": metrics,
        "training_rows": int(len(X)),
        "manual_label_rows": int((meta["label_source"] == "manual").sum()) if not meta.empty else 0,
        "proxy_label_rows": int((meta["label_source"] == "proxy").sum()) if not meta.empty else 0,
    }
    joblib.dump(artifact_payload, args.model_out)

    output_dir = ensure_dir(args.output_dir)
    metrics_path = output_dir / "ml_training_metrics.json"
    metrics_payload = {
        "feature_columns": list(X.columns),
        "attributes": attrs,
        "metrics": metrics,
        "training_rows": int(len(X)),
        "manual_label_rows": int((meta["label_source"] == "manual").sum()) if not meta.empty else 0,
        "proxy_label_rows": int((meta["label_source"] == "proxy").sum()) if not meta.empty else 0,
    }
    write_json(metrics_path, metrics_payload, indent=2)

    decisions_df, selected_df = _predict_decisions(model, list(X.columns), df, attrs)

    decisions_path = output_dir / "ml_attribute_decisions.csv"
    selected_path = output_dir / "ml_selected_records.csv"
    summary_path = output_dir / "ml_summary.csv"

    summary_df = (
        decisions_df.groupby(["attribute", "winner"]).size().reset_index(name="count")
        .sort_values(["attribute", "winner"])
    )

    write_csv(decisions_path, decisions_df)
    write_csv(selected_path, selected_df)
    write_csv(summary_path, summary_df)

    print("ML conflation complete. Wrote:")
    print(f"  model: {args.model_out}")
    print(f"  metrics: {metrics_path}")
    print(f"  {decisions_path}")
    print(f"  {selected_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
