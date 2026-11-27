"""Data loading, domain cleaning, preprocessing, and model construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from .config import (
    RANDOM_STATE,
    RAW_DATA_PATH,
    SCENARIO_A_BINARY,
    SCENARIO_A_NOMINAL,
    SCENARIO_A_NUMERIC,
    SCENARIO_A_ORDINAL,
    SCENARIO_B_BINARY,
    SCENARIO_B_NOMINAL,
    SCENARIO_B_NUMERIC,
    SCENARIO_FEATURES,
    TARGET_COL,
    TEST_SIZE,
)


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the untouched source CSV."""
    frame = pd.read_csv(path)
    if frame.shape != (32581, 12):
        raise ValueError(f"Unexpected source shape {frame.shape}; expected (32581, 12).")
    return frame


def clean_domain_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates and impossible domain values, preserving missing tenure."""
    cleaned = df.drop_duplicates().copy()
    cleaned = cleaned[
        (cleaned["person_age"] <= 100)
        & ((cleaned["person_emp_length"] <= 60) | cleaned["person_emp_length"].isna())
    ].copy()
    if len(cleaned) != 32409:
        raise AssertionError(f"Expected 32409 cleaned rows, got {len(cleaned)}")
    # The attached specification's 889-row missingness figure does not match this archive.
    # Preserve and report the observed count; do not turn an incorrect figure into a test.
    if cleaned["person_emp_length"].isna().sum() != 887:
        raise AssertionError("Unexpected person_emp_length missing count after domain cleaning.")
    return cleaned.reset_index(drop=True)


def split_train_test(
    df: pd.DataFrame, scenario: str = "B"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the single stratified holdout split for the chosen feature scenario."""
    scenario = scenario.upper()
    if scenario not in SCENARIO_FEATURES:
        raise ValueError("scenario must be 'A' or 'B'")
    X = df.loc[:, SCENARIO_FEATURES[scenario]].copy()
    y = df[TARGET_COL].astype(int).copy()
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def build_preprocessor(scenario: str = "B") -> ColumnTransformer:
    """Build fold-local imputers and encoders. No scaling is needed for trees."""
    scenario = scenario.upper()
    if scenario == "B":
        numeric, nominal, binary, ordinal = (
            SCENARIO_B_NUMERIC, SCENARIO_B_NOMINAL, SCENARIO_B_BINARY, []
        )
    elif scenario == "A":
        numeric, nominal, binary, ordinal = (
            SCENARIO_A_NUMERIC, SCENARIO_A_NOMINAL, SCENARIO_A_BINARY, SCENARIO_A_ORDINAL
        )
    else:
        raise ValueError("scenario must be 'A' or 'B'")

    transformers: list[tuple[str, Pipeline, list[str]]] = [
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
        (
            "nominal",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            nominal,
        ),
        (
            "binary",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ord", OrdinalEncoder(
                    categories=[["N", "Y"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )),
            ]),
            binary,
        ),
    ]
    if ordinal:
        transformers.append((
            "ordinal",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ord", OrdinalEncoder(
                    categories=[["A", "B", "C", "D", "E", "F", "G"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )),
            ]),
            ordinal,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


class BalancedGradientBoostingClassifier(GradientBoostingClassifier):
    """Gradient boosting with fold-local balanced sample weights."""

    def __init__(
        self,
        *,
        loss="log_loss",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        criterion="deprecated",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        init=None,
        random_state=None,
        max_features=None,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=0.0001,
        ccp_alpha=0.0,
        use_balanced_weights=True,
    ):
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            init=init,
            random_state=random_state,
            max_features=max_features,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha,
        )
        self.use_balanced_weights = use_balanced_weights

    def fit(self, X, y, sample_weight=None, monitor=None):
        if self.use_balanced_weights:
            balanced = compute_sample_weight(class_weight="balanced", y=y)
            sample_weight = balanced if sample_weight is None else balanced * np.asarray(sample_weight)
        return super().fit(X, y, sample_weight=sample_weight, monitor=monitor)


def get_candidate_models(balanced: bool = True, random_state: int = RANDOM_STATE) -> dict[str, BaseEstimator]:
    weight = "balanced" if balanced else None
    return {
        "DecisionTree": DecisionTreeClassifier(
            random_state=random_state, class_weight=weight
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, random_state=random_state, class_weight=weight, n_jobs=1
        ),
        "GradientBoosting": BalancedGradientBoostingClassifier(
            random_state=random_state, use_balanced_weights=balanced
        ),
    }


def build_model_pipeline(estimator: BaseEstimator, scenario: str = "B") -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(scenario)),
        ("classifier", estimator),
    ])
