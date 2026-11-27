"""Shared project paths and feature definitions."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credit_risk_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"
IMAGES_DIR = DOCS_DIR / "images"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TARGET_COL = "loan_status"
MAJORITY_CLASS_BASELINE = 0.7818  # Raw-source prevalence; recomputed on cleaned data in reports.

SCENARIO_B_NUMERIC = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_percent_income", "cb_person_cred_hist_length",
]
SCENARIO_B_NOMINAL = ["person_home_ownership", "loan_intent"]
SCENARIO_B_BINARY = ["cb_person_default_on_file"]

SCENARIO_A_NUMERIC = SCENARIO_B_NUMERIC + ["loan_int_rate"]
SCENARIO_A_NOMINAL = SCENARIO_B_NOMINAL.copy()
SCENARIO_A_BINARY = SCENARIO_B_BINARY.copy()
SCENARIO_A_ORDINAL = ["loan_grade"]

SCENARIO_FEATURES = {
    "B": SCENARIO_B_NUMERIC + SCENARIO_B_NOMINAL + SCENARIO_B_BINARY,
    "A": SCENARIO_A_NUMERIC + SCENARIO_A_NOMINAL + SCENARIO_A_BINARY + SCENARIO_A_ORDINAL,
}


def ensure_directories() -> None:
    """Create only the directories used by the project outputs."""
    for path in (MODELS_DIR, OUTPUTS_DIR, IMAGES_DIR):
        path.mkdir(parents=True, exist_ok=True)
