import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from src.preprocessing import (
    BalancedGradientBoostingClassifier,
    build_preprocessor,
    clean_domain_outliers,
    load_raw_data,
    split_train_test,
)


def test_load_shape_and_domain_cleaning_preserves_observed_missing_values():
    raw = load_raw_data()
    assert raw.shape == (32581, 12)
    cleaned = clean_domain_outliers(raw)
    assert len(cleaned) == 32409
    # The supplied archive has 887 after duplicate and domain filters (not 889 as stated in the brief).
    assert cleaned.person_emp_length.isna().sum() == 887


def test_scenario_b_split_has_expected_shape_and_no_leakage_columns():
    cleaned = clean_domain_outliers(load_raw_data())
    X_train, X_test, y_train, y_test = split_train_test(cleaned, "B")
    assert X_train.shape == (25927, 9)
    assert X_test.shape == (6482, 9)
    assert "loan_status" not in X_train
    assert "loan_grade" not in X_train
    assert "loan_int_rate" not in X_train
    assert len(y_train) == 25927 and len(y_test) == 6482


def test_preprocessor_imputes_and_ignores_unseen_categories():
    cleaned = clean_domain_outliers(load_raw_data())
    X_train, X_test, _, _ = split_train_test(cleaned, "B")
    processor = build_preprocessor("B")
    transformed_train = processor.fit_transform(X_train)
    assert not np.isnan(transformed_train).any()
    names = processor.get_feature_names_out().tolist()
    assert any("missingindicator" in name or name.endswith("_missing") for name in names)
    unusual = X_test.iloc[:2].copy()
    unusual.loc[unusual.index[0], "person_home_ownership"] = "UNSEEN_CATEGORY"
    unusual.loc[unusual.index[1], "loan_intent"] = "UNSEEN_CATEGORY"
    transformed = processor.transform(unusual)
    assert transformed.shape[0] == 2
    assert not np.isnan(transformed).any()


def test_balanced_boosting_is_cloneable_and_cross_validates():
    X = np.arange(120, dtype=float).reshape(60, 2)
    y = np.array([0] * 42 + [1] * 18)
    model = BalancedGradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=42)
    assert clone(model).get_params()["use_balanced_weights"] is True
    scores = cross_val_score(model, X, y, cv=3, scoring="average_precision")
    assert len(scores) == 3
    assert np.isfinite(scores).all()
