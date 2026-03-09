"""Train and apply a Random Forest classifier for attribute source selection.

Uses golden labels (Yelp-verified + heuristic) to train a single shared model
across all attributes. Evaluates via stratified train/test split.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from scripts.utils.conflation import CORE_ATTRIBUTES, feature_vector, proxy_label
from scripts.utils.io import (
    ANALYSIS_DIR,
    DEFAULT_DATA_PATH,
    PROJECT_ROOT,
    REPORTS_DIR,
    ensure_dir,
    load_parquet_duckdb,
    write_csv,
    write_json,
)


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
    """Build a single training table across all attributes."""
    rows: list[dict] = []
    labels: list[int] = []
    metas: list[dict] = []

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
            metas.append({"id": record_id, "attribute": attr, "label_source": label_source})

    X = pd.DataFrame(rows).fillna(0.0)
    y = pd.Series(labels, name="target")
    meta = pd.DataFrame(metas)
    return X, y, meta


def _predict(
    model: object,
    feature_cols: list[str],
    df: pd.DataFrame,
    attrs: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision_rows: list[dict] = []
    selected_rows: list[dict] = []

    for _, record in df.iterrows():
        confidence = _safe_float(record.get("confidence"))
        base_confidence = _safe_float(record.get("base_confidence"))

        selected_record = {"id": record.get("id"), "base_id": record.get("base_id")}

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

            frame = pd.DataFrame([feat]).reindex(columns=feature_cols, fill_value=0.0)
            pred = int(model.predict(frame)[0])
            winner = "current" if pred == 1 else "base"

            prob_current = float(model.predict_proba(frame)[0][1])

            selected_value = record.get(attr) if winner == "current" else record.get(f"base_{attr}")
            selected_record[f"selected_{attr}"] = selected_value
            selected_record[f"selected_from_{attr}"] = winner

            decision_rows.append({
                "id": record.get("id"),
                "base_id": record.get("base_id"),
                "attribute": attr,
                "winner": winner,
                "prob_current": prob_current,
            })

        selected_rows.append(selected_record)

    return pd.DataFrame(decision_rows), pd.DataFrame(selected_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Random Forest for attribute selection")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--golden-labels", type=Path,
        default=ANALYSIS_DIR / "golden" / "golden_dataset_template.json",
    )
    parser.add_argument("--no-proxy", action="store_true", help="Disable weak proxy labels")
    parser.add_argument("--model-out", type=Path, default=PROJECT_ROOT / "models" / "ml_selector.joblib")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR / "conflation")
    args = parser.parse_args()

    df = load_parquet_duckdb(args.input)
    attrs = [a for a in CORE_ATTRIBUTES if a in df.columns and f"base_{a}" in df.columns]

    manual_labels = _load_manual_labels(args.golden_labels)
    use_proxy = not args.no_proxy

    print(f"\nBuilding training data...")
    X, y, meta = _build_training_table(df, attrs, manual_labels, use_proxy)

    manual_n = int((meta["label_source"] == "manual").sum())
    proxy_n = int((meta["label_source"] == "proxy").sum())

    print(f"  Training samples: {len(X)} (manual={manual_n}, proxy={proxy_n})")
    print(f"  Features: {X.shape[1]}")
    print(f"  Class balance: current={int(y.sum())}, base={int(len(y) - y.sum())}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate on held-out test set
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    report = classification_report(y_test, preds, target_names=["base", "current"], output_dict=True)

    print(f"\n{'='*50}")
    print(f"  HOLDOUT EVALUATION (20% test set)")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print(f"  Base F1:   {report['base']['f1-score']:.4f}")
    print(f"  Current F1:{report['current']['f1-score']:.4f}")
    print(classification_report(y_test, preds, target_names=["base", "current"]))

    # Retrain on full data for final predictions
    model.fit(X, y)

    # Save model
    ensure_dir(args.model_out.parent)
    artifact = {
        "model": model,
        "feature_columns": list(X.columns),
        "attributes": attrs,
    }
    joblib.dump(artifact, args.model_out)

    # Save metrics
    output_dir = ensure_dir(args.output_dir)
    metrics_path = output_dir / "ml_training_metrics.json"
    write_json(metrics_path, {
        "best_model": "random_forest",
        "feature_columns": list(X.columns),
        "attributes": attrs,
        "holdout_metrics": {
            "accuracy": acc,
            "support_test": len(y_test),
            "classification_report": report,
        },
        "training_rows": len(X),
        "manual_label_rows": manual_n,
        "proxy_label_rows": proxy_n,
    }, indent=2)

    # Predict on full dataset
    decisions_df, selected_df = _predict(model, list(X.columns), df, attrs)

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

    print(f"\nML conflation complete. Wrote:")
    print(f"  model:   {args.model_out}")
    print(f"  metrics: {metrics_path}")
    print(f"  {decisions_path}")
    print(f"  {selected_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
