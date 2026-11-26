# Kết quả thực nghiệm

## Dữ liệu và lớp đa số

CSV gốc: 32,581 × 12; sau xử lý miền: 32,409. Tập train/test: 25,927/6,482. Tỷ lệ nhãn 1 sau làm sạch: 21.87%; baseline accuracy lớp đa số: **78.13%**. Chỉ số thiếu việc làm đo trên dữ liệu đã làm sạch: 887 (2.74%); tỷ lệ default nhóm thiếu là 31.68%, nhóm có giá trị 21.59%.

## Baseline CV

| model | weighting | AP_mean | AP_std | ROC_AUC_mean | F1_mean | Precision_mean | Recall_mean | Accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoosting | Unweighted | 0.7873 | 0.0059 | 0.8790 | 0.6515 | 0.9535 | 0.4951 | 0.8842 |
| GradientBoosting | Balanced | 0.7844 | 0.0063 | 0.8783 | 0.6547 | 0.6080 | 0.7093 | 0.8363 |
| RandomForest | Unweighted | 0.7701 | 0.0038 | 0.8649 | 0.6629 | 0.8862 | 0.5296 | 0.8822 |
| RandomForest | Balanced | 0.7684 | 0.0035 | 0.8651 | 0.6698 | 0.7414 | 0.6109 | 0.8682 |
| DecisionTree | Balanced | 0.4734 | 0.0110 | 0.7596 | 0.6250 | 0.6273 | 0.6229 | 0.8365 |
| DecisionTree | Unweighted | 0.4687 | 0.0123 | 0.7616 | 0.6236 | 0.6113 | 0.6367 | 0.8319 |

## Mô hình tuned, 5-fold CV trên train

Mô hình được chọn theo AP: **GradientBoosting**.

| Model | AP mean | AP SD | ROC-AUC mean | ROC-AUC SD | F1 mean | F1 SD | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoosting | 0.8228 | 0.0085 | 0.9069 | 0.0069 | 0.7032 | 0.0046 | 0.6637 | 0.7478 |
| RandomForest | 0.7889 | 0.0057 | 0.8805 | 0.0033 | 0.6782 | 0.0076 | 0.6673 | 0.6896 |
| DecisionTree | 0.7611 | 0.0075 | 0.8561 | 0.0049 | 0.6293 | 0.0197 | 0.5583 | 0.7226 |

## Ngưỡng OOF và đánh giá test cuối cùng

`t_opt=0.62`; `t_recall70=0.56`. AP/ROC-AUC không phụ thuộc ngưỡng; các metric phân loại thay đổi theo ngưỡng. Accuracy được đọc cùng baseline 78.13%, không dùng làm tiêu chí chọn.

| evaluation | ap | roc_auc | f1 | precision | recall | accuracy | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default_0.50 | 0.8355 | 0.9167 | 0.7136 | 0.6581 | 0.7793 | 0.8632 | 4490 | 574 | 313 | 1105 |
| oof_t_opt | 0.8355 | 0.9167 | 0.7253 | 0.7727 | 0.6834 | 0.8868 | 4779 | 285 | 449 | 969 |

## So sánh thời điểm đặc trưng

| scenario_name | evaluation | ap | roc_auc | f1 | precision | recall | accuracy | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pre-Decision | default_0.50 | 0.8355 | 0.9167 | 0.7136 | 0.6581 | 0.7793 | 0.8632 | 4490 | 574 | 313 | 1105 |
| Pre-Decision | oof_t_opt | 0.8355 | 0.9167 | 0.7253 | 0.7727 | 0.6834 | 0.8868 | 4779 | 285 | 449 | 969 |
| Full-Feature (leakage comparison) | default_0.50 | 0.9152 | 0.9549 | 0.8254 | 0.8292 | 0.8216 | 0.9239 | 4824 | 240 | 253 | 1165 |
| Full-Feature (leakage comparison) | oof_t_opt | 0.9152 | 0.9549 | 0.8443 | 0.9281 | 0.7743 | 0.9375 | 4979 | 85 | 320 | 1098 |

Scenario A có các tín hiệu hậu thẩm định; chênh lệch của nó với B đo mức phụ thuộc mô hình vào các biến này, không phải ước lượng hiệu quả triển khai trước quyết định.

## Tầm quan trọng biến và kiểm chứng từ đầu

| feature | built_in_impurity_importance | permutation_ap_mean | permutation_ap_std |
| --- | --- | --- | --- |
| person_income | 0.3269 | 0.3317 | 0.0062 |
| loan_percent_income | 0.2919 | 0.3200 | 0.0115 |
| person_home_ownership | 0.1065 | 0.2263 | 0.0087 |
| loan_amnt | 0.0483 | 0.1781 | 0.0089 |
| loan_intent | 0.0812 | 0.0975 | 0.0043 |
| cb_person_default_on_file | 0.0718 | 0.0451 | 0.0027 |
| person_age | 0.0287 | 0.0319 | 0.0022 |
| person_emp_length | 0.0290 | 0.0296 | 0.0012 |
| cb_person_cred_hist_length | 0.0148 | 0.0126 | 0.0009 |

| weighting | scratch_feature | scratch_threshold | scratch_information_gain | sklearn_feature | sklearn_threshold | sklearn_information_gain | absolute_gain_difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| unweighted | loan_percent_income | 0.3050 | 0.0629 | loan_percent_income | 0.3050 | 0.0629 | 0.0000 |
| balanced | loan_percent_income | 0.3050 | 0.0838 | loan_percent_income | 0.3050 | 0.0838 | 0.0000 |

Sai số tuyệt đối tối đa của boosting từ đầu so với `GradientBoostingClassifier`: **1.110e-16**.

Các bảng đầy đủ, bao gồm tuning candidates và threshold sweep, nằm trong `outputs/`; 14 hình nằm trong `docs/images/`.
