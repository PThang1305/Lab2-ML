# Phương pháp

## Mục tiêu và thời điểm dự đoán

Đây là bài toán phân loại nhị phân dự đoán `loan_status` (1 = vỡ nợ). Kịch bản chính **B / Pre-Decision** chỉ dùng thông tin có sẵn khi nhận hồ sơ. Kịch bản đối chứng **A / Full-Feature** thêm `loan_grade` và `loan_int_rate`; đây là các biến được tạo trong quá trình thẩm định/định giá và có quan hệ rất mạnh với nhãn. Kịch bản A dùng để minh họa rò rỉ nghiệp vụ, không phải phương án triển khai trước quyết định.

## Kiểm toán và làm sạch

Nguồn có 32,581 dòng và 12 cột; có 165 dòng trùng hoàn toàn. Áp dụng đúng quy tắc miền: loại bản trùng, tuổi trên 100, và số năm làm việc trên 60; NaN trong `person_emp_length` được giữ bằng điều kiện rõ ràng. Kết quả là 32,409 dòng. Archive thực tế còn **887** giá trị thiếu `person_emp_length` sau lọc (tỷ lệ vỡ nợ 31.68%, so với 21.59% ở nhóm có dữ liệu). Con số 889 trong bản mô tả đi kèm không khớp với CSV được cung cấp; báo cáo này dùng số đo từ dữ liệu.

## Chia dữ liệu và tiền xử lý

Một lần chia train/test stratified 80/20, `random_state=42`, tạo 25,927 dòng train và 6,482 dòng test. Mọi bước học tham số—median/mode imputation, one-hot encoding, và mô hình—nằm trong `Pipeline`/`ColumnTransformer`, nên được fit riêng trong từng fold. Không lưu bảng đã impute trước CV. `person_emp_length` có cờ missing do `SimpleImputer(add_indicator=True)`.

Không dùng StandardScaler/MinMaxScaler: cây quyết định phân vùng theo thứ tự và ngưỡng `x_j <= tau`; phép đổi tỷ lệ đơn điệu dương không đổi thứ tự quan sát, nên không thay đổi các phân vùng ứng viên của cây.

## Đánh giá, chọn mô hình và ngưỡng

Dùng `StratifiedKFold(5, shuffle=True, random_state=42)`. Metric chọn/tuning chính là **Average Precision (AP)**, tương ứng `sklearn.metrics.average_precision_score`; ROC-AUC, F1, precision, recall và accuracy được báo cáo bổ sung. Accuracy được đặt cạnh baseline lớp đa số (78.13% sau làm sạch; 78.18% trên CSV gốc), không dùng để chọn mô hình.

So sánh mô hình thường và `class_weight='balanced'` cho Decision Tree/Random Forest. Với boosting cân bằng, `compute_sample_weight('balanced', y)` được tính ngay trong `fit` trên nhãn của fold hiện tại. Ngưỡng được chọn trên xác suất Out-of-Fold của train: `t_opt=0.62` tối đa F1; `t_recall70=0.56` là ngưỡng cao nhất giữ recall ít nhất 0.70. Test được dùng ở bước cuối để tính xác suất một lần, sau đó cùng xác suất được tổng hợp ở 0.50 và `t_opt`.

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

Dataset SHA-256: `ce3c6d2167717bf1627d1c0c81cbccd28323cd4aa7b96d542599366d5ff6aac8`. Mọi đường dẫn được suy ra từ vị trí repository; các phép biến đổi không tạo `data/processed/`.
