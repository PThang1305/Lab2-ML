import numpy as np
import pandas as pd

from src.config import SCENARIO_FEATURES
from src.explainability import gradient_boosting_scratch_binary, verify_gini_against_sklearn
from src.predict import predict_dataframe
from src.preprocessing import build_preprocessor, clean_domain_outliers, load_raw_data, split_train_test


def test_scratch_root_gini_matches_sklearn():
    cleaned = clean_domain_outliers(load_raw_data())
    X_train, _, y_train, _ = split_train_test(cleaned, "B")
    sample = np.random.default_rng(42).choice(len(X_train), size=1200, replace=False)
    X_sample = X_train.iloc[sample]
    y_sample = y_train.iloc[sample].to_numpy()
    processor = build_preprocessor("B")
    transformed = processor.fit_transform(X_sample)
    result = verify_gini_against_sklearn(
        transformed, y_sample, processor.get_feature_names_out().tolist()
    )
    assert result.absolute_gain_difference.max() < 1e-6


def test_scratch_boosting_matches_sklearn():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(700, 5))
    y = (X[:, 0] + 0.7 * X[:, 1] > 0.3).astype(int)
    result = gradient_boosting_scratch_binary(X, y, n_estimators=5, max_depth=3, random_state=42)
    assert result["max_abs_diff"] < 1e-12


def test_predict_dataframe_uses_saved_pipeline():
    cleaned = clean_domain_outliers(load_raw_data())
    X_train, _, _, _ = split_train_test(cleaned, "B")
    sample = X_train.iloc[:5].loc[:, SCENARIO_FEATURES["B"]].copy()
    predictions = predict_dataframe(sample)
    assert len(predictions) == 5
    assert predictions.default_probability.between(0, 1).all()
    assert set(predictions.predicted_default.unique()).issubset({0, 1})
    assert set(predictions.risk_level.unique()).issubset({"LOW", "MEDIUM", "HIGH"})
