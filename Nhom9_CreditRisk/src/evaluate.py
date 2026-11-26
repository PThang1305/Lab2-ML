"""Probability metrics, OOF threshold selection, and one-pass holdout scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .config import CV_FOLDS, OUTPUTS_DIR, RANDOM_STATE


def compute_classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "ap": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def optimize_oof_threshold(pipeline, X_train, y_train):
    """Choose thresholds using only stratified out-of-fold training probabilities."""
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = cross_val_predict(
        clone(pipeline), X_train, y_train,
        cv=splitter, method="predict_proba", n_jobs=-1,
    )[:, 1]
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    rows = []
    for threshold in thresholds:
        metrics = compute_classification_metrics(y_train, oof_probs, float(threshold))
        rows.append(metrics)
    table = pd.DataFrame(rows)
    max_f1 = table["f1"].max()
    # If several thresholds tie, select the largest threshold for fewer alerts.
    t_opt = float(table.loc[table["f1"] == max_f1, "threshold"].max())
    eligible = table.loc[table["recall"] >= 0.70, "threshold"]
    t_recall70 = float(eligible.max()) if not eligible.empty else float("nan")
    table["selected_for_f1"] = table["threshold"].eq(t_opt)
    table["selected_for_recall70"] = table["threshold"].eq(t_recall70)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUTS_DIR / "threshold_analysis.csv", index=False)
    return table, t_opt, t_recall70, oof_probs


def evaluate_final_test(
    fitted_pipeline, X_test, y_test, t_opt: float,
    model_name: str, scenario: str,
) -> pd.DataFrame:
    """Predict the holdout once, then summarize at the default and locked threshold."""
    probabilities = fitted_pipeline.predict_proba(X_test)[:, 1]
    rows = []
    for label, threshold in (("default_0.50", 0.50), ("oof_t_opt", t_opt)):
        metrics = compute_classification_metrics(y_test, probabilities, threshold)
        rows.append({
            "model": model_name, "scenario": scenario.upper(),
            "evaluation": label, **metrics,
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUTS_DIR / "test_results.csv", index=False)
    return result


def run_scenario_comparison(
    X_train_b, y_train_b, X_test_b, y_test_b,
    X_train_a, y_train_a, X_test_a, y_test_a,
    best_estimator_template,
    precomputed_b=None,
) -> pd.DataFrame:
    """Compare full-feature and pre-decision scenarios using training-only OOF cutoffs."""
    rows = []
    from .preprocessing import build_model_pipeline

    classifier_template = best_estimator_template.named_steps["classifier"]
    for scenario, X_train, y_train, X_test, y_test in (
        ("B", X_train_b, y_train_b, X_test_b, y_test_b),
        ("A", X_train_a, y_train_a, X_test_a, y_test_a),
    ):
        if scenario == "B" and precomputed_b is not None:
            test_prob, t_opt = precomputed_b
        else:
            pipeline = build_model_pipeline(clone(classifier_template), scenario)
            oof = cross_val_predict(
                pipeline, X_train, y_train,
                cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                method="predict_proba", n_jobs=-1,
            )[:, 1]
            thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
            f1s = [f1_score(y_train, oof >= t, zero_division=0) for t in thresholds]
            t_opt = float(thresholds[np.flatnonzero(np.asarray(f1s) == np.max(f1s))[-1]])
            pipeline.fit(X_train, y_train)
            test_prob = pipeline.predict_proba(X_test)[:, 1]
        for label, threshold in (("default_0.50", 0.5), ("oof_t_opt", t_opt)):
            rows.append({
                "scenario": scenario,
                "scenario_name": "Pre-Decision" if scenario == "B" else "Full-Feature (leakage comparison)",
                "oof_t_opt": t_opt,
                "evaluation": label,
                **compute_classification_metrics(y_test, test_prob, threshold),
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUTS_DIR / "scenario_comparison.csv", index=False)
    return result
