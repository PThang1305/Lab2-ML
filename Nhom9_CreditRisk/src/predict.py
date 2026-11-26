"""Load the trained pipeline and score new applicant rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .config import MODELS_DIR


def predict_dataframe(df_input: pd.DataFrame, model_path=None, metadata_path=None) -> pd.DataFrame:
    """Return input columns plus probability, locked label, and risk band."""
    model_file = Path(model_path) if model_path else MODELS_DIR / "final_model_pipeline.joblib"
    metadata_file = Path(metadata_path) if metadata_path else MODELS_DIR / "model_metadata.json"
    if not model_file.exists() or not metadata_file.exists():
        raise FileNotFoundError("Run `python -m src.pipeline` to create the model artifact first.")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    required = metadata["input_columns"]
    missing = [column for column in required if column not in df_input.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    model = joblib.load(model_file)
    result = df_input.copy()
    probabilities = model.predict_proba(df_input.loc[:, required])[:, 1]
    threshold = float(metadata["optimal_threshold"])
    result["default_probability"] = probabilities
    result["predicted_default"] = (probabilities >= threshold).astype(int)
    low_max = float(metadata["risk_bands"]["low_max"])
    medium_max = float(metadata["risk_bands"]["medium_max"])
    result["risk_level"] = pd.cut(
        probabilities,
        bins=[-float("inf"), low_max, medium_max, float("inf")],
        labels=["LOW", "MEDIUM", "HIGH"],
        right=False,
    ).astype(str)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score applicant records with the trained credit-risk model.")
    parser.add_argument("--input", required=True, help="CSV containing the scenario's input columns")
    parser.add_argument("--output", required=True, help="Destination CSV for predictions")
    args = parser.parse_args(argv)
    predictions = predict_dataframe(pd.read_csv(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    print(f"Wrote {len(predictions):,} predictions to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
