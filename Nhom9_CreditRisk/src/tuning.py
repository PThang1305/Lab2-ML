"""Cross-validated baselines and AP-led hyperparameter search."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)

from .config import CV_FOLDS, OUTPUTS_DIR, RANDOM_STATE
from .preprocessing import build_model_pipeline, get_candidate_models

SCORING = {
    "AP": "average_precision",
    "ROC_AUC": "roc_auc",
    "F1": "f1",
    "Precision": "precision",
    "Recall": "recall",
    "Accuracy": "accuracy",
}


def _cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def evaluate_baselines_cv(X_train, y_train, scenario: str = "B") -> pd.DataFrame:
    """Compare unweighted and balanced model families on identical training folds."""
    rows: list[dict] = []
    for balanced in (False, True):
        label = "Balanced" if balanced else "Unweighted"
        for name, estimator in get_candidate_models(balanced=balanced).items():
            pipeline = build_model_pipeline(estimator, scenario)
            scores = cross_validate(
                pipeline, X_train, y_train, cv=_cv(), scoring=SCORING,
                n_jobs=-1, return_train_score=False, error_score="raise",
            )
            row = {"model": name, "weighting": label, "scenario": scenario.upper()}
            for metric in SCORING:
                row[f"{metric}_mean"] = float(scores[f"test_{metric}"].mean())
                row[f"{metric}_std"] = float(scores[f"test_{metric}"].std(ddof=1))
            rows.append(row)
    result = pd.DataFrame(rows).sort_values("AP_mean", ascending=False)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUTS_DIR / "baseline_results.csv", index=False)
    return result


def get_search_spaces() -> dict:
    """Return the prescribed search grids and sampling distributions."""
    return {
        "DecisionTree": {
            "type": "grid",
            "params": {
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": [3, 5, 7, 10, 14, None],
                "classifier__min_samples_split": [2, 10, 30],
                "classifier__min_samples_leaf": [1, 5, 15, 30],
            },
        },
        "RandomForest": {
            "type": "random",
            "n_iter": 25,
            "params": {
                "classifier__n_estimators": [150, 300],
                "classifier__max_depth": [6, 10, 14, 18, None],
                "classifier__min_samples_split": [2, 10, 25],
                "classifier__min_samples_leaf": [1, 5, 15],
                "classifier__max_features": ["sqrt", "log2", 0.5],
            },
        },
        "GradientBoosting": {
            "type": "random",
            "n_iter": 25,
            "params": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__max_depth": [2, 3, 4, 5],
                "classifier__subsample": [0.8, 1.0],
                "classifier__min_samples_leaf": [10, 20, 50],
                "classifier__max_features": ["sqrt", None],
            },
        },
    }


def run_hyperparameter_tuning(
    X_train, y_train, scenario: str = "B", balanced: bool = True
) -> dict:
    """Tune all three families against AP and report CV metrics for each winner."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    candidates = get_candidate_models(balanced=balanced)
    spaces = get_search_spaces()
    best: dict = {}
    summaries: list[dict] = []

    for name, estimator in candidates.items():
        pipeline = build_model_pipeline(estimator, scenario)
        space = spaces[name]
        common = dict(
            estimator=pipeline,
            scoring="average_precision",
            cv=_cv(),
            n_jobs=-1,
            refit=True,
            return_train_score=False,
            error_score="raise",
            pre_dispatch="2*n_jobs",
        )
        if space["type"] == "grid":
            search = GridSearchCV(param_grid=space["params"], **common)
        else:
            search = RandomizedSearchCV(
                param_distributions=space["params"],
                n_iter=space["n_iter"],
                random_state=RANDOM_STATE,
                **common,
            )
        search.fit(X_train, y_train)
        best[name] = search.best_estimator_
        search_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        search_results.to_csv(OUTPUTS_DIR / f"tuning_{ {'DecisionTree':'dt','RandomForest':'rf','GradientBoosting':'gbm'}[name] }.csv", index=False)

        cv_scores = cross_validate(
            search.best_estimator_, X_train, y_train,
            cv=_cv(), scoring=SCORING, n_jobs=-1,
            return_train_score=False, error_score="raise",
        )
        row = {
            "model": name,
            "scenario": scenario.upper(),
            "weighting": "Balanced" if balanced else "Unweighted",
            "best_params": repr(search.best_params_),
            "search_best_AP": float(search.best_score_),
        }
        for metric in SCORING:
            row[f"{metric}_mean"] = float(cv_scores[f"test_{metric}"].mean())
            row[f"{metric}_std"] = float(cv_scores[f"test_{metric}"].std(ddof=1))
        summaries.append(row)

    summary = pd.DataFrame(summaries).sort_values("AP_mean", ascending=False)
    summary.to_csv(OUTPUTS_DIR / "cv_results.csv", index=False)
    return {"best_estimators": best, "cv_results": summary}
