"""Train/apply per-attribute ML models for source selection (hybrid approach).

Compares Logistic Regression, Random Forest, Gradient Boosting, and XGBoost
per attribute. Selects the best model for each attribute by macro-F1 via
stratified cross-validation, then produces final predictions on the full dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from scripts.utils.conflation import CORE_ATTRIBUTES, feature_vector, proxy_label
from scripts.utils.io import ANALYSIS_DIR, DEFAULT_DATA_PATH, PROJECT_ROOT, REPORTS_DIR, ensure_dir, load_parquet_duckdb, write_csv, write_json


def _make_candidates() -> dict[str, object]:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42,
            eval_metric="logloss", use_label_encoder=False,
            scale_pos_weight=1.0, verbosity=0,
        ),
    }


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


def _build_per_attr_tables(
    df: pd.DataFrame,
    attrs: list[str],
    manual_labels: dict[tuple[str, str], str],
    use_proxy_labels: bool,
) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
    """Build separate training tables for each attribute."""
    attr_data: dict[str, tuple[list, list, list]] = {a: ([], [], []) for a in attrs}

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

            rows, labels, metas = attr_data[attr]
            rows.append(features)
            labels.append(1 if label == "current" else 0)
            metas.append({"id": record_id, "attribute": attr, "label_source": label_source})

    result = {}
    for attr in attrs:
        rows, labels, metas = attr_data[attr]
        if rows:
            X = pd.DataFrame(rows).fillna(0.0)
            y = pd.Series(labels, name="target")
            meta = pd.DataFrame(metas)
            result[attr] = (X, y, meta)

    return result


def _select_best_for_attr(
    attr: str, X: pd.DataFrame, y: pd.Series, n_folds: int = 5,
) -> tuple[str, object, dict]:
    cv = StratifiedKFold(n_splits=min(n_folds, y.value_counts().min()), shuffle=True, random_state=42)

    best_name = ""
    best_f1 = -1.0
    best_model = None
    results = {}

    for name, model in _make_candidates().items():
        try:
            cv_preds = cross_val_predict(model, X, y, cv=cv)
        except Exception:
            continue

        macro_f1 = f1_score(y, cv_preds, average="macro")
        acc = accuracy_score(y, cv_preds)
        report = classification_report(y, cv_preds, target_names=["base", "current"], output_dict=True)

        results[name] = {"accuracy": float(acc), "macro_f1": float(macro_f1), "classification_report": report}

        base_f1 = report["base"]["f1-score"]
        cur_f1 = report["current"]["f1-score"]
        print(f"    {name:>22s}  |  macro-F1={macro_f1:.4f}  (base={base_f1:.4f}, current={cur_f1:.4f})")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_name = name
            best_model = model

    if best_model is not None:
        best_model.fit(X, y)

    return best_name, best_model, results


def _predict_per_attr(
    attr_models: dict[str, tuple[object, list[str]]],
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
            if attr not in attr_models:
                continue

            model, feature_cols = attr_models[attr]
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

            prob_current = 0.5
            if hasattr(model, "predict_proba"):
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
    parser = argparse.ArgumentParser(description="Train per-attribute ML models (hybrid)")
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
    attr_tables = _build_per_attr_tables(df, attrs, manual_labels, use_proxy)

    print(f"\n{'='*60}")
    print(f"  PER-ATTRIBUTE MODEL SELECTION (hybrid={'yes' if use_proxy else 'no'})")
    print(f"{'='*60}")

    attr_models: dict[str, tuple[object, list[str]]] = {}
    all_cv_results: dict[str, dict] = {}
    all_holdout: dict[str, dict] = {}
    total_manual = 0
    total_proxy = 0

    for attr in attrs:
        if attr not in attr_tables:
            print(f"\n  [{attr}] SKIPPED — no training data")
            continue

        X, y, meta = attr_tables[attr]
        manual_n = int((meta["label_source"] == "manual").sum())
        proxy_n = int((meta["label_source"] == "proxy").sum())
        total_manual += manual_n
        total_proxy += proxy_n

        if y.nunique() < 2:
            print(f"\n  [{attr}] SKIPPED — only one class present")
            continue

        print(f"\n  [{attr.upper()}] samples={len(X)} (manual={manual_n}, proxy={proxy_n})  features={X.shape[1]}")

        best_name, best_model, cv_results = _select_best_for_attr(attr, X, y)
        all_cv_results[attr] = {"best_model": best_name, "models": cv_results}
        attr_models[attr] = (best_model, list(X.columns))

        # Holdout eval
        if len(X) >= 20:
            stratify = y if y.nunique() > 1 else None
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
            model_copy = clone(best_model)
            model_copy.fit(X_tr, y_tr)
            preds = model_copy.predict(X_te)
            holdout_f1 = f1_score(y_te, preds, average="macro")
            all_holdout[attr] = {
                "model": best_name,
                "holdout_macro_f1": float(holdout_f1),
                "holdout_accuracy": float(accuracy_score(y_te, preds)),
                "test_size": int(len(X_te)),
            }
            print(f"    >>> Best: {best_name} | Holdout F1={holdout_f1:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for attr in attrs:
        if attr in all_holdout:
            h = all_holdout[attr]
            print(f"  {attr:>12s}  |  {h['model']:>22s}  |  Holdout-F1={h['holdout_macro_f1']:.4f}  (n={h['test_size']})")
    print(f"\n  Total training: manual={total_manual}, proxy={total_proxy}")

    # Save artifacts
    ensure_dir(args.model_out.parent)
    artifact = {
        "attr_models": {a: {"model": m, "columns": c} for a, (m, c) in attr_models.items()},
        "attributes": attrs,
        "cv_results": all_cv_results,
        "holdout": all_holdout,
        "total_manual": total_manual,
        "total_proxy": total_proxy,
    }
    joblib.dump(artifact, args.model_out)

    output_dir = ensure_dir(args.output_dir)
    metrics_path = output_dir / "ml_training_metrics.json"
    metrics_payload = {
        "approach": "per_attribute_hybrid",
        "attributes": attrs,
        "cv_results": all_cv_results,
        "holdout": all_holdout,
        "total_training_manual": total_manual,
        "total_training_proxy": total_proxy,
    }
    write_json(metrics_path, metrics_payload, indent=2)

    decisions_df, selected_df = _predict_per_attr(attr_models, df, attrs)

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
    print(f"  model: {args.model_out}")
    print(f"  metrics: {metrics_path}")
    print(f"  {decisions_path}")
    print(f"  {selected_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
