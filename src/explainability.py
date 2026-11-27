"""Tree importance summaries and from-scratch mathematical checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .config import OUTPUTS_DIR, RANDOM_STATE


def extract_feature_importances(fitted_pipeline, X_eval, y_eval):
    """Join built-in transformed-feature importance to raw-column permutation importance."""
    classifier = fitted_pipeline.named_steps["classifier"]
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)
    built_in = np.asarray(classifier.feature_importances_, dtype=float)
    perm = permutation_importance(
        fitted_pipeline, X_eval, y_eval, scoring="average_precision",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    raw_columns = list(X_eval.columns)
    grouped_builtin = {col: 0.0 for col in raw_columns}
    rows_encoded = []
    for name, value in zip(names, built_in, strict=True):
        source = next((col for col in raw_columns if name == col or name.startswith(f"{col}_")), name)
        if source in grouped_builtin:
            grouped_builtin[source] += float(value)
        rows_encoded.append({"encoded_feature": name, "source_feature": source, "built_in_importance": float(value)})
    result = pd.DataFrame([
        {
            "feature": col,
            "built_in_impurity_importance": grouped_builtin[col],
            "permutation_ap_mean": float(perm.importances_mean[i]),
            "permutation_ap_std": float(perm.importances_std[i]),
        }
        for i, col in enumerate(raw_columns)
    ]).sort_values("permutation_ap_mean", ascending=False)
    result.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)
    pd.DataFrame(rows_encoded).to_csv(OUTPUTS_DIR / "feature_importance_encoded.csv", index=False)
    return result, pd.DataFrame(rows_encoded)


def gini_impurity(y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    weights = np.ones(y.size, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    total = weights.sum()
    if total <= 0:
        return 0.0
    _, inverse = np.unique(y, return_inverse=True)
    class_weights = np.bincount(inverse, weights=weights)
    probabilities = class_weights / total
    return float(1.0 - np.square(probabilities).sum())


def find_best_root_split_scratch(X, y, feature_names, sample_weight=None) -> dict:
    """Find the maximum weighted Gini decrease in O(p N log N)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    weights = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    classes, inverse = np.unique(y, return_inverse=True)
    total_weight = weights.sum()
    parent = gini_impurity(y, weights)
    best = {"feature": None, "feature_index": -1, "threshold": np.nan, "information_gain": -np.inf}
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="mergesort")
        sorted_x = X[order, j]
        sorted_w = weights[order]
        sorted_y = inverse[order]
        cumulative = np.zeros((len(classes), len(y)), dtype=float)
        for c in range(len(classes)):
            cumulative[c] = np.cumsum(sorted_w * (sorted_y == c))
        valid = np.flatnonzero(sorted_x[:-1] < sorted_x[1:])
        if valid.size == 0:
            continue
        left_class = cumulative[:, valid]
        left_weight = left_class.sum(axis=0)
        right_class = cumulative[:, -1, None] - left_class
        right_weight = total_weight - left_weight
        left_gini = 1.0 - np.square(left_class / left_weight).sum(axis=0)
        right_gini = 1.0 - np.square(right_class / right_weight).sum(axis=0)
        gains = parent - (left_weight / total_weight) * left_gini - (right_weight / total_weight) * right_gini
        local = int(np.argmax(gains))
        gain = float(gains[local])
        if gain > best["information_gain"] + 1e-15:
            k = int(valid[local])
            best = {
                "feature": str(feature_names[j]),
                "feature_index": int(j),
                "threshold": float(sorted_x[k] / 2.0 + sorted_x[k + 1] / 2.0),
                "information_gain": gain,
                "left_gini": float(left_gini[local]),
                "right_gini": float(right_gini[local]),
                "left_weight": float(left_weight[local]),
                "right_weight": float(right_weight[local]),
            }
    return best


def verify_gini_against_sklearn(X_train_transformed, y_train, feature_names) -> pd.DataFrame:
    records = []
    for label, sample_weight in (
        ("unweighted", None),
        ("balanced", __import__("sklearn.utils.class_weight", fromlist=["compute_sample_weight"]).compute_sample_weight("balanced", y_train)),
    ):
        scratch = find_best_root_split_scratch(X_train_transformed, y_train, feature_names, sample_weight)
        tree = DecisionTreeClassifier(max_depth=1, criterion="gini", random_state=RANDOM_STATE)
        tree.fit(X_train_transformed, y_train, sample_weight=sample_weight)
        tr = tree.tree_
        left, right = tr.children_left[0], tr.children_right[0]
        total = tr.weighted_n_node_samples[0]
        sklearn_gain = float(tr.impurity[0] - (tr.weighted_n_node_samples[left] / total) * tr.impurity[left] - (tr.weighted_n_node_samples[right] / total) * tr.impurity[right])
        diff = abs(float(scratch["information_gain"]) - sklearn_gain)
        records.append({
            "weighting": label,
            "scratch_feature": scratch["feature"],
            "scratch_threshold": scratch["threshold"],
            "scratch_information_gain": scratch["information_gain"],
            "sklearn_feature": str(feature_names[tr.feature[0]]),
            "sklearn_threshold": float(tr.threshold[0]),
            "sklearn_information_gain": sklearn_gain,
            "absolute_gain_difference": diff,
        })
    result = pd.DataFrame(records)
    if result["absolute_gain_difference"].max() >= 1e-6:
        raise AssertionError("Scratch weighted Gini gain does not match scikit-learn within 1e-6.")
    return result


def gradient_boosting_scratch_binary(
    X, y, n_estimators: int = 5, learning_rate: float = 0.1,
    max_depth: int = 3, random_state: int = RANDOM_STATE,
) -> dict:
    """Reproduce sklearn's binary logistic boosting stages with Newton leaf updates."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    p0 = float(y.mean())
    initial = float(np.log(p0 / (1.0 - p0)))
    score = np.full(y.shape[0], initial, dtype=float)
    trees, leaf_values, loss_history = [], [], []
    rng = np.random.RandomState(random_state)
    for _ in range(n_estimators):
        probability = expit(score)
        residual = y - probability
        seed = int(rng.randint(np.iinfo(np.int32).max))
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=seed)
        tree.fit(X, residual)
        leaves = tree.apply(X)
        gamma_map = {}
        for leaf in np.unique(leaves):
            mask = leaves == leaf
            numerator = residual[mask].sum()
            denominator = (probability[mask] * (1.0 - probability[mask])).sum()
            gamma_map[int(leaf)] = float(numerator / denominator)
        update = np.array([gamma_map[int(leaf)] for leaf in leaves])
        score += learning_rate * update
        trees.append(tree)
        leaf_values.append(gamma_map)
        probability_after = np.clip(expit(score), 1e-15, 1 - 1e-15)
        loss_history.append(float(-np.mean(y * np.log(probability_after) + (1-y) * np.log(1-probability_after))))

    reference = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        random_state=random_state,
    ).fit(X, y.astype(int))
    reference_scores = reference.decision_function(X)
    max_abs_diff = float(np.max(np.abs(score - reference_scores)))
    if max_abs_diff >= 1e-12:
        raise AssertionError(f"Scratch boosting mismatch: max absolute difference {max_abs_diff:.3g}")
    return {
        "initial_log_odds": initial,
        "trees": trees,
        "leaf_values": leaf_values,
        "decision_function": score,
        "reference_decision_function": reference_scores,
        "max_abs_diff": max_abs_diff,
        "log_loss_by_stage": loss_history,
        "reference_model": reference,
    }
