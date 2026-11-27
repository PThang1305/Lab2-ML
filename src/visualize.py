"""Create the report's 14 reproducible PNG figures."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import validation_curve
from sklearn.tree import plot_tree

from .config import IMAGES_DIR, RANDOM_STATE
from .preprocessing import build_model_pipeline
from sklearn.tree import DecisionTreeClassifier

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, filename: str) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_all_visuals(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    baseline: pd.DataFrame,
    cv_results: pd.DataFrame,
    threshold_table: pd.DataFrame,
    y_test,
    test_prob,
    t_opt: float,
    importance: pd.DataFrame,
    best_pipeline,
    X_train,
    y_train,
    gini_results: pd.DataFrame,
    scratch_boost: dict,
) -> None:
    """Render all report figures using the actual run data."""
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = raw["loan_status"].value_counts().sort_index()
    ax.bar(["No default (0)", "Default (1)"], counts.values, color=["#4C78A8", "#F58518"])
    ax.set(title="Target distribution in the supplied CSV", ylabel="Applications")
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value:,} ({value / len(raw):.2%})", ha="center", va="bottom")
    _save(fig, "1_eda_target_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    miss = raw.isna().mean().mul(100).sort_values(ascending=False)
    axes[0].bar(miss.index, miss.values, color="#72B7B2")
    axes[0].set(title="Missing values in raw source", ylabel="Rows missing (%)")
    axes[0].tick_params(axis="x", rotation=70)
    axes[1].bar(
        ["Age > 100", "Employment > 60"],
        [(raw.person_age > 100).sum(), (raw.person_emp_length > 60).sum()],
        color="#E45756",
    )
    axes[1].set(title="Domain-rule records in raw source", ylabel="Rows")
    _save(fig, "1_eda_missing_and_outliers.png")

    grade = raw.groupby("loan_grade", observed=True).agg(
        default_rate=("loan_status", "mean"),
        historical_default_share=("cb_person_default_on_file", lambda s: (s == "Y").mean()),
        mean_interest_rate=("loan_int_rate", "mean"),
        n=("loan_status", "size"),
    ).reindex(list("ABCDEFG"))
    fig, ax = plt.subplots(figsize=(9, 5))
    xx = np.arange(len(grade))
    ax.plot(xx, grade.default_rate, marker="o", label="Default rate")
    ax.plot(xx, grade.historical_default_share, marker="s", label="Prior default on file")
    ax.set_xticks(xx, [f"{g} (n={int(n):,})" for g, n in zip(grade.index, grade.n, strict=True)])
    ax.set(ylabel="Share", title="Post-underwriting signals by loan grade", ylim=(0, 1.03))
    ax2 = ax.twinx()
    ax2.plot(xx, grade.mean_interest_rate, color="#E45756", marker="^", label="Mean interest rate")
    ax2.set_ylabel("Mean interest rate (%)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    _save(fig, "1_eda_grade_leakage_analysis.png")

    cols = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "loan_status"]
    corr = raw[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Numeric correlation in raw data")
    _save(fig, "1_eda_correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = baseline["model"] + "\n" + baseline["weighting"]
    ax.barh(labels, baseline.AP_mean, xerr=baseline.AP_std, color="#4C78A8", alpha=0.9)
    ax.set(xlabel="Average Precision (AP)", title="Baseline 5-fold CV (Scenario B)")
    _save(fig, "3_baseline_cv_comparison.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ordered = cv_results.sort_values("AP_mean")
    ax.barh(ordered.model, ordered.AP_mean, xerr=ordered.AP_std, color="#54A24B")
    ax.set(xlabel="Average Precision (AP)", title="Tuned balanced models · 5-fold CV")
    _save(fig, "4_tuned_cv_comparison.png")

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(threshold_table.threshold, threshold_table.f1, label="F1", color="#4C78A8")
    ax1.plot(threshold_table.threshold, threshold_table.recall, label="Recall", color="#F58518")
    ax1.plot(threshold_table.threshold, threshold_table.precision, label="Precision", color="#54A24B")
    ax1.axvline(t_opt, ls="--", color="black", label=f"t_opt = {t_opt:.2f}")
    ax1.axhline(0.70, ls=":", color="#E45756", label="Recall target = 0.70")
    ax1.set(xlabel="Decision threshold", ylabel="Score", title="Threshold selection on training OOF predictions", ylim=(0, 1))
    ax1.legend(ncol=2)
    _save(fig, "5_threshold_optimization.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, test_prob, ax=ax, name="Pre-decision model")
    ax.plot([0, 1], [0, 1], ls="--", color="gray")
    ax.set_title("ROC curve · held-out test")
    _save(fig, "6_roc_curves.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, test_prob, ax=ax, name="Pre-decision model")
    ax.axhline(np.mean(y_test), ls="--", color="gray", label="Test prevalence")
    ax.set_title("Precision–Recall curve · held-out test")
    _save(fig, "6_pr_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, threshold, title in ((axes[0], 0.5, "Threshold 0.50"), (axes[1], t_opt, f"OOF threshold {t_opt:.2f}")):
        ConfusionMatrixDisplay.from_predictions(y_test, test_prob >= threshold, labels=[0, 1], display_labels=["No default", "Default"], cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(title)
    fig.suptitle("Test confusion matrices")
    _save(fig, "6_confusion_matrices.png")

    shallow = build_model_pipeline(DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=RANDOM_STATE), "B")
    shallow.fit(X_train, y_train)
    names = shallow.named_steps["preprocessor"].get_feature_names_out()
    fig, ax = plt.subplots(figsize=(21, 9))
    plot_tree(shallow.named_steps["classifier"], feature_names=names, class_names=["No default", "Default"], filled=True, rounded=True, fontsize=7, ax=ax)
    ax.set_title("Balanced decision tree, depth 3 (Scenario B)")
    _save(fig, "7_dt_tree_structure.png")

    depths = [1, 2, 3, 4, 5, 7, 10]
    curve_pipe = build_model_pipeline(DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE), "B")
    train_scores, valid_scores = validation_curve(
        curve_pipe, X_train, y_train, param_name="classifier__max_depth",
        param_range=depths, cv=3, scoring="average_precision", n_jobs=-1,
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(depths, train_scores.mean(axis=1), yerr=train_scores.std(axis=1), marker="o", label="Train")
    ax.errorbar(depths, valid_scores.mean(axis=1), yerr=valid_scores.std(axis=1), marker="s", label="Validation")
    ax.set(xlabel="Maximum depth", ylabel="Average Precision (AP)", title="Decision tree validation curve")
    ax.legend()
    _save(fig, "7_validation_curve_bias_variance.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    stages = np.arange(1, len(scratch_boost["log_loss_by_stage"]) + 1)
    ax.plot(stages, scratch_boost["log_loss_by_stage"], marker="o", label="From-scratch")
    ax.set(xlabel="Boosting iteration", ylabel="Training binary log loss", title=f"Five-step Newton boosting (max diff {scratch_boost['max_abs_diff']:.1e})")
    ax.legend()
    _save(fig, "7_boosting_from_scratch.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    order = importance.sort_values("permutation_ap_mean", ascending=True)
    ax.barh(order.feature, order.built_in_impurity_importance, alpha=0.65, label="Built-in impurity")
    scaled_perm = order.permutation_ap_mean / max(float(order.permutation_ap_mean.max()), 1e-12)
    ax.scatter(scaled_perm, order.feature, color="#E45756", marker="D", label="Permutation AP (scaled to max=1)")
    ax.set(xlabel="Impurity importance / scaled permutation AP", title="Feature importance methods answer different questions")
    ax.legend()
    _save(fig, "7_feature_importance_comparison.png")
