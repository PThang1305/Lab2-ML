"""End-to-end training, evaluation, reporting, and artifact generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    IMAGES_DIR,
    MAJORITY_CLASS_BASELINE,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    SCENARIO_FEATURES,
    TARGET_COL,
    ensure_directories,
)
from .evaluate import (
    compute_classification_metrics,
    evaluate_final_test,
    optimize_oof_threshold,
    run_scenario_comparison,
)
from .explainability import (
    extract_feature_importances,
    gradient_boosting_scratch_binary,
    verify_gini_against_sklearn,
)
from .preprocessing import clean_domain_outliers, load_raw_data, split_train_test
from .tuning import evaluate_baselines_cv, run_hyperparameter_tuning
from .visualize import create_all_visuals


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    selected = frame.loc[:, columns] if columns else frame
    headers = [str(column) for column in selected.columns]

    def render(value):
        if isinstance(value, (float, np.floating)):
            return "—" if np.isnan(value) else f"{value:.4f}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    rows.extend("| " + " | ".join(render(value) for value in row) + " |" for row in selected.itertuples(index=False, name=None))
    return "\n".join(rows)


def _write_reports(raw, cleaned, X_train, y_train, X_test, y_test, baseline, cv_results,
                   threshold_table, t_opt, t_recall70, test_results, comparison,
                   gini_results, scratch, importance, best_name, dataset_sha):
    missing = int(cleaned.person_emp_length.isna().sum())
    missing_default = float(cleaned.loc[cleaned.person_emp_length.isna(), TARGET_COL].mean())
    present_default = float(cleaned.loc[cleaned.person_emp_length.notna(), TARGET_COL].mean())
    majority = float(max(cleaned[TARGET_COL].mean(), 1 - cleaned[TARGET_COL].mean()))
    grade_table = raw.groupby("loan_grade", observed=True).agg(
        n=(TARGET_COL, "size"), default_rate=(TARGET_COL, "mean"),
        prior_default_share=("cb_person_default_on_file", lambda s: (s == "Y").mean()),
        mean_rate=("loan_int_rate", "mean"),
    ).reset_index()
    grade_table["default_rate"] = grade_table.default_rate.map(lambda x: f"{x:.2%}")
    grade_table["prior_default_share"] = grade_table.prior_default_share.map(lambda x: f"{x:.2%}")
    grade_table["mean_rate"] = grade_table.mean_rate.map(lambda x: f"{x:.2f}%")
    grade_table["n"] = grade_table.n.map(lambda x: f"{x:,}")
    test_show = test_results.copy()
    for col in ("ap", "roc_auc", "f1", "precision", "recall", "accuracy"):
        test_show[col] = test_show[col].map(lambda x: f"{x:.4f}")
    method = f'''# Phương pháp

## Mục tiêu và thời điểm dự đoán

Đây là bài toán phân loại nhị phân dự đoán `loan_status` (1 = vỡ nợ). Kịch bản chính **B / Pre-Decision** chỉ dùng thông tin có sẵn khi nhận hồ sơ. Kịch bản đối chứng **A / Full-Feature** thêm `loan_grade` và `loan_int_rate`; đây là các biến được tạo trong quá trình thẩm định/định giá và có quan hệ rất mạnh với nhãn. Kịch bản A dùng để minh họa rò rỉ nghiệp vụ, không phải phương án triển khai trước quyết định.

## Kiểm toán và làm sạch

Nguồn có {len(raw):,} dòng và {raw.shape[1]} cột; có {int(raw.duplicated().sum()):,} dòng trùng hoàn toàn. Áp dụng đúng quy tắc miền: loại bản trùng, tuổi trên 100, và số năm làm việc trên 60; NaN trong `person_emp_length` được giữ bằng điều kiện rõ ràng. Kết quả là {len(cleaned):,} dòng. Archive thực tế còn **{missing}** giá trị thiếu `person_emp_length` sau lọc (tỷ lệ vỡ nợ {missing_default:.2%}, so với {present_default:.2%} ở nhóm có dữ liệu). Con số 889 trong bản mô tả đi kèm không khớp với CSV được cung cấp; báo cáo này dùng số đo từ dữ liệu.

## Chia dữ liệu và tiền xử lý

Một lần chia train/test stratified 80/20, `random_state=42`, tạo {len(X_train):,} dòng train và {len(X_test):,} dòng test. Mọi bước học tham số—median/mode imputation, one-hot encoding, và mô hình—nằm trong `Pipeline`/`ColumnTransformer`, nên được fit riêng trong từng fold. Không lưu bảng đã impute trước CV. `person_emp_length` có cờ missing do `SimpleImputer(add_indicator=True)`.

Không dùng StandardScaler/MinMaxScaler: cây quyết định phân vùng theo thứ tự và ngưỡng `x_j <= tau`; phép đổi tỷ lệ đơn điệu dương không đổi thứ tự quan sát, nên không thay đổi các phân vùng ứng viên của cây.

## Đánh giá, chọn mô hình và ngưỡng

Dùng `StratifiedKFold(5, shuffle=True, random_state=42)`. Metric chọn/tuning chính là **Average Precision (AP)**, tương ứng `sklearn.metrics.average_precision_score`; ROC-AUC, F1, precision, recall và accuracy được báo cáo bổ sung. Accuracy được đặt cạnh baseline lớp đa số ({majority:.2%} sau làm sạch; {MAJORITY_CLASS_BASELINE:.2%} trên CSV gốc), không dùng để chọn mô hình.

So sánh mô hình thường và `class_weight='balanced'` cho Decision Tree/Random Forest. Với boosting cân bằng, `compute_sample_weight('balanced', y)` được tính ngay trong `fit` trên nhãn của fold hiện tại. Ngưỡng được chọn trên xác suất Out-of-Fold của train: `t_opt={t_opt:.2f}` tối đa F1; `t_recall70={t_recall70:.2f}` là ngưỡng cao nhất giữ recall ít nhất 0.70. Test được dùng ở bước cuối để tính xác suất một lần, sau đó cùng xác suất được tổng hợp ở 0.50 và `t_opt`.

## Giải thích và kiểm chứng toán học

Độ quan trọng impurity tích hợp của mô hình cây được gom về cột nguồn sau one-hot; permutation importance trên AP được tính trên mẫu train giữ lại riêng cho giải thích. Các mức quan trọng là mô tả phụ thuộc dự đoán, không phải quan hệ nhân quả. Đa cộng tuyến/các biến tương quan có thể chia sẻ hoặc làm lệch tầm quan trọng.

Gini và mức giảm impurity ở nút gốc được tính từ đầu bằng sắp xếp và tổng tích lũy, rồi đối chiếu với `DecisionTreeClassifier(max_depth=1)`. Cài đặt boosting nhị phân từ đầu dùng pseudo-residual `y-p` và cập nhật lá Newton `sum(y-p)/sum(p(1-p))`; kết quả được so sánh trực tiếp với 5-stage scikit-learn.

## Tái lập

```bash
python -m pip install -r requirements.txt
python -m src.pipeline
pytest -v
jupyter nbconvert --to notebook --execute --inplace notebooks/Nhom9_CreditRisk_Lab.ipynb
```

Dataset SHA-256: `{dataset_sha}`. Mọi đường dẫn được suy ra từ vị trí repository; các phép biến đổi không tạo `data/processed/`.
'''
    (Path(__file__).resolve().parents[1] / "docs" / "methodology.md").write_text(method, encoding="utf-8")

    cv_show = cv_results[["model", "AP_mean", "AP_std", "ROC_AUC_mean", "ROC_AUC_std", "F1_mean", "F1_std", "Precision_mean", "Recall_mean"]].copy()
    cv_show.columns = ["Model", "AP mean", "AP SD", "ROC-AUC mean", "ROC-AUC SD", "F1 mean", "F1 SD", "Precision", "Recall"]
    for col in cv_show.columns[1:]:
        cv_show[col] = cv_show[col].map(lambda x: f"{x:.4f}")
    comparison_show = comparison[["scenario_name", "evaluation", "ap", "roc_auc", "f1", "precision", "recall", "accuracy", "tn", "fp", "fn", "tp"]].copy()
    test_baseline = pd.DataFrame([{
        "Evaluation": "Majority-class baseline (cleaned data)", "Accuracy": f"{majority:.4f}",
        "AP": "—", "ROC-AUC": "—", "F1": "—", "Precision": "—", "Recall": "—",
    }])
    results = f'''# Kết quả thực nghiệm

## Dữ liệu và lớp đa số

CSV gốc: {len(raw):,} × {raw.shape[1]}; sau xử lý miền: {len(cleaned):,}. Tập train/test: {len(X_train):,}/{len(X_test):,}. Tỷ lệ nhãn 1 sau làm sạch: {cleaned[TARGET_COL].mean():.2%}; baseline accuracy lớp đa số: **{majority:.2%}**. Chỉ số thiếu việc làm đo trên dữ liệu đã làm sạch: {missing} ({missing/len(cleaned):.2%}); tỷ lệ default nhóm thiếu là {missing_default:.2%}, nhóm có giá trị {present_default:.2%}.

## Baseline CV

{_markdown_table(baseline[['model','weighting','AP_mean','AP_std','ROC_AUC_mean','F1_mean','Precision_mean','Recall_mean','Accuracy_mean']].round(4))}

## Mô hình tuned, 5-fold CV trên train

Mô hình được chọn theo AP: **{best_name}**.

{_markdown_table(cv_show)}

## Ngưỡng OOF và đánh giá test cuối cùng

`t_opt={t_opt:.2f}`; `t_recall70={t_recall70:.2f}`. AP/ROC-AUC không phụ thuộc ngưỡng; các metric phân loại thay đổi theo ngưỡng. Accuracy được đọc cùng baseline {majority:.2%}, không dùng làm tiêu chí chọn.

{_markdown_table(test_show[['evaluation','ap','roc_auc','f1','precision','recall','accuracy','tn','fp','fn','tp']])}

## So sánh thời điểm đặc trưng

{_markdown_table(comparison_show.round(4))}

Scenario A có các tín hiệu hậu thẩm định; chênh lệch của nó với B đo mức phụ thuộc mô hình vào các biến này, không phải ước lượng hiệu quả triển khai trước quyết định.

## Tầm quan trọng biến và kiểm chứng từ đầu

{_markdown_table(importance.round(5))}

{_markdown_table(gini_results.round(10))}

Sai số tuyệt đối tối đa của boosting từ đầu so với `GradientBoostingClassifier`: **{scratch['max_abs_diff']:.3e}**.

Các bảng đầy đủ, bao gồm tuning candidates và threshold sweep, nằm trong `outputs/`; 14 hình nằm trong `docs/images/`.
'''
    (Path(__file__).resolve().parents[1] / "docs" / "results.md").write_text(results, encoding="utf-8")
    readme = f'''# Nhom9_CreditRisk

Phân tích và phân loại rủi ro vỡ nợ khoản vay bằng Decision Tree, Random Forest và Gradient Boosting. Mục tiêu là `loan_status` (1 = vỡ nợ).

## Thời điểm dự đoán và rò rỉ

Kịch bản **B / Pre-Decision** là kết quả chính, chỉ dùng 9 đặc trưng sẵn có khi nhận hồ sơ. Kịch bản **A / Full-Feature** thêm `loan_grade` và `loan_int_rate`, hai biến hình thành trong thẩm định/định giá, để đo lượng thông tin hậu quyết định lọt vào mô hình. Không dùng A làm mô hình triển khai trước quyết định.

## Kết quả chạy trên CSV đi kèm

- Sau làm sạch theo quy tắc miền: **{len(cleaned):,}** dòng; train/test: **{len(X_train):,}/{len(X_test):,}**.
- Mô hình chọn theo 5-fold CV Average Precision (AP): **{best_name}**.
- AP test: **{float(test_results.iloc[0].ap):.4f}**; F1 tại ngưỡng OOF `t_opt={t_opt:.2f}`: **{float(test_results.iloc[1].f1):.4f}**.
- Baseline accuracy lớp đa số: **{majority:.2%}** trên dữ liệu đã làm sạch (CSV gốc: {MAJORITY_CLASS_BASELINE:.2%}); accuracy không dùng để chọn mô hình.

CSV archive thực tế có {missing} giá trị thiếu `person_emp_length` sau lọc. Bản mô tả đầu vào ghi 889; số đo trên archive là 887, và mã kiểm tra/report dùng con số quan sát được.

## Cấu trúc

`src/` chứa pipeline, tiền xử lý, tuning, đánh giá, giải thích, trực quan hóa và CLI dự đoán. `notebooks/` có notebook báo cáo thực thi. `data/raw/` giữ dữ liệu nguồn. `outputs/`, `models/`, `docs/` lưu kết quả tái lập. Không tạo `data/processed/`.

## Cài đặt và chạy

```bash
python -m pip install -r requirements.txt
python -m src.pipeline
pytest -v
jupyter nbconvert --to notebook --execute --inplace notebooks/Nhom9_CreditRisk_Lab.ipynb
```

Chạy inference với CSV hồ sơ mới có đủ các cột Scenario B:

```bash
python -m src.predict --input path/to/new_applicants.csv --output path/to/predictions.csv
```

Mở `notebooks/Nhom9_CreditRisk_Lab.ipynb` trong Jupyter để xem EDA, phương pháp, kết quả và kiểm chứng from-scratch. Chi tiết phương pháp ở [docs/methodology.md](docs/methodology.md), số liệu đầy đủ ở [docs/results.md](docs/results.md).
'''
    (Path(__file__).resolve().parents[1] / "README.md").write_text(readme, encoding="utf-8")


def run_pipeline() -> dict:
    ensure_directories()
    raw = load_raw_data()
    cleaned = clean_domain_outliers(raw)
    X_train, X_test, y_train, y_test = split_train_test(cleaned, "B")
    X_train_a, X_test_a, y_train_a, y_test_a = split_train_test(cleaned, "A")
    if len(X_train) != 25927 or len(X_test) != 6482:
        raise AssertionError("Unexpected stratified split size.")
    if list(X_train.columns) != SCENARIO_FEATURES["B"] or not set(X_train.index).isdisjoint(X_test.index):
        raise AssertionError("Feature definition or train/test partition is invalid.")
    if "loan_grade" in X_train or "loan_int_rate" in X_train or TARGET_COL in X_train:
        raise AssertionError("Scenario B contains a target or post-underwriting feature.")

    digest = hashlib.sha256(RAW_DATA_PATH.read_bytes()).hexdigest()
    metadata_file = MODELS_DIR / "model_metadata.json"
    checkpoint_files = [
        OUTPUTS_DIR / name for name in (
            "baseline_results.csv", "tuning_dt.csv", "tuning_rf.csv", "tuning_gbm.csv",
            "cv_results.csv", "threshold_analysis.csv", "test_results.csv",
            "scenario_comparison.csv", "feature_importance.csv",
            "gini_scratch_verification.csv", "scratch_boosting_verification.json",
        )
    ]
    model_file = MODELS_DIR / "final_model_pipeline.joblib"
    if metadata_file.exists() and model_file.exists() and all(path.exists() for path in checkpoint_files):
        saved_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if saved_metadata.get("dataset_sha256") == digest:
            print("[resume] Tuning and scoring outputs found; completing report artifacts", flush=True)
            best_name = saved_metadata["model_name"]
            best_pipeline = joblib.load(model_file)
            baseline = pd.read_csv(OUTPUTS_DIR / "baseline_results.csv")
            cv_results = pd.read_csv(OUTPUTS_DIR / "cv_results.csv")
            threshold_table = pd.read_csv(OUTPUTS_DIR / "threshold_analysis.csv")
            test_results = pd.read_csv(OUTPUTS_DIR / "test_results.csv")
            comparison = pd.read_csv(OUTPUTS_DIR / "scenario_comparison.csv")
            importance = pd.read_csv(OUTPUTS_DIR / "feature_importance.csv")
            gini_results = pd.read_csv(OUTPUTS_DIR / "gini_scratch_verification.csv")
            scratch_saved = json.loads((OUTPUTS_DIR / "scratch_boosting_verification.json").read_text(encoding="utf-8"))
            scratch = {**scratch_saved}
            t_opt = float(saved_metadata["optimal_threshold"])
            t_recall70 = saved_metadata["recall_70_threshold"]
            if t_recall70 is not None:
                t_recall70 = float(t_recall70)
            probability_file = OUTPUTS_DIR / "test_probabilities.npy"
            if probability_file.exists():
                probabilities = np.load(probability_file)
            else:
                probabilities = best_pipeline.predict_proba(X_test)[:, 1]
                np.save(probability_file, probabilities)
            create_all_visuals(
                raw, cleaned, baseline, cv_results, threshold_table, y_test,
                probabilities, t_opt, importance, best_pipeline, X_train, y_train,
                gini_results, scratch,
            )
            _write_reports(
                raw, cleaned, X_train, y_train, X_test, y_test, baseline, cv_results,
                threshold_table, t_opt, t_recall70, test_results, comparison,
                gini_results, scratch, importance, best_name, digest,
            )
            return {
                "best_name": best_name, "best_pipeline": best_pipeline,
                "threshold_table": threshold_table, "t_opt": t_opt,
                "t_recall70": t_recall70, "test_results": test_results,
                "comparison": comparison, "gini_results": gini_results,
                "scratch_boost": scratch, "feature_importance": importance,
            }

    print("[1/8] Baseline cross-validation (Scenario B)", flush=True)
    baseline = evaluate_baselines_cv(X_train, y_train, "B")
    print("[2/8] AP-led hyperparameter tuning (balanced models)", flush=True)
    tuning = run_hyperparameter_tuning(X_train, y_train, "B", balanced=True)
    cv_results = tuning["cv_results"]
    best_name = str(cv_results.iloc[0].model)
    best_pipeline = tuning["best_estimators"][best_name]

    print("[3/8] OOF threshold selection", flush=True)
    threshold_table, t_opt, t_recall70, oof_probs = optimize_oof_threshold(best_pipeline, X_train, y_train)
    print("[4/8] Final test evaluation", flush=True)
    probabilities = best_pipeline.predict_proba(X_test)[:, 1]
    np.save(OUTPUTS_DIR / "test_probabilities.npy", probabilities)
    rows = []
    for label, threshold in (("default_0.50", 0.5), ("oof_t_opt", t_opt)):
        rows.append({"model": best_name, "scenario": "B", "evaluation": label,
                     **compute_classification_metrics(y_test, probabilities, threshold)})
    test_results = pd.DataFrame(rows)
    test_results.to_csv(OUTPUTS_DIR / "test_results.csv", index=False)

    print("[5/8] Pre-decision vs full-feature comparison", flush=True)
    comparison = run_scenario_comparison(
        X_train, y_train, X_test, y_test,
        X_train_a, y_train_a, X_test_a, y_test_a,
        best_pipeline,
        precomputed_b=(probabilities, t_opt),
    )
    print("[6/8] Explainability and mathematical checks", flush=True)
    # A reproducible 5,000-row training-only subset limits compute for permutation importance.
    X_imp, _, y_imp, _ = train_test_split(
        X_train, y_train, train_size=min(5000, len(X_train)),
        stratify=y_train, random_state=RANDOM_STATE,
    )
    importance, encoded_importance = extract_feature_importances(best_pipeline, X_imp, y_imp)
    transformed = best_pipeline.named_steps["preprocessor"].transform(X_train)
    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    gini_results = verify_gini_against_sklearn(transformed, y_train.to_numpy(), feature_names)
    scratch = gradient_boosting_scratch_binary(
        transformed, y_train.to_numpy(), n_estimators=5, learning_rate=0.1,
        max_depth=3, random_state=RANDOM_STATE,
    )
    gini_results.to_csv(OUTPUTS_DIR / "gini_scratch_verification.csv", index=False)
    (OUTPUTS_DIR / "scratch_boosting_verification.json").write_text(
        json.dumps({
            "max_abs_diff": scratch["max_abs_diff"],
            "log_loss_by_stage": scratch["log_loss_by_stage"],
            "initial_log_odds": scratch["initial_log_odds"],
        }, indent=2),
        encoding="utf-8",
    )

    model_file = MODELS_DIR / "final_model_pipeline.joblib"
    metadata_file = MODELS_DIR / "model_metadata.json"
    joblib.dump(best_pipeline, model_file, compress=3)
    digest = hashlib.sha256(RAW_DATA_PATH.read_bytes()).hexdigest()
    metadata = {
        "scenario": "B",
        "scenario_name": "pre_decision",
        "model_name": best_name,
        "input_columns": SCENARIO_FEATURES["B"],
        "target_column": TARGET_COL,
        "positive_class": 1,
        "optimal_threshold": float(t_opt),
        "recall_70_threshold": None if np.isnan(t_recall70) else float(t_recall70),
        "risk_bands": {"low_max": 0.10, "medium_max": 0.30},
        "random_state": RANDOM_STATE,
        "dataset_sha256": digest,
        "test_ap": float(test_results.iloc[0].ap),
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("[7/8] Writing report tables and figures", flush=True)
    create_all_visuals(
        raw, cleaned, baseline, cv_results, threshold_table, y_test,
        probabilities, t_opt, importance, best_pipeline, X_train, y_train,
        gini_results, scratch,
    )
    _write_reports(
        raw, cleaned, X_train, y_train, X_test, y_test, baseline, cv_results,
        threshold_table, t_opt, t_recall70, test_results, comparison,
        gini_results, scratch, importance, best_name, digest,
    )
    print("[8/8] Model and reports saved", flush=True)
    return {
        "best_name": best_name,
        "best_pipeline": best_pipeline,
        "threshold_table": threshold_table,
        "t_opt": t_opt,
        "t_recall70": t_recall70,
        "test_results": test_results,
        "comparison": comparison,
        "gini_results": gini_results,
        "scratch_boost": scratch,
        "feature_importance": importance,
    }


if __name__ == "__main__":
    run_pipeline()
