# Nhom9_CreditRisk

Phân tích và phân loại rủi ro vỡ nợ khoản vay bằng Decision Tree, Random Forest và Gradient Boosting. Mục tiêu là `loan_status` (1 = vỡ nợ).

## Thời điểm dự đoán và rò rỉ

Kịch bản **B / Pre-Decision** là kết quả chính, chỉ dùng 9 đặc trưng sẵn có khi nhận hồ sơ. Kịch bản **A / Full-Feature** thêm `loan_grade` và `loan_int_rate`, hai biến hình thành trong thẩm định/định giá, để đo lượng thông tin hậu quyết định lọt vào mô hình. Không dùng A làm mô hình triển khai trước quyết định.

## Kết quả chạy trên CSV đi kèm

- Sau làm sạch theo quy tắc miền: **32,409** dòng; train/test: **25,927/6,482**.
- Mô hình chọn theo 5-fold CV Average Precision (AP): **GradientBoosting**.
- AP test: **0.8355**; F1 tại ngưỡng OOF `t_opt=0.62`: **0.7253**.
- Baseline accuracy lớp đa số: **78.13%** trên dữ liệu đã làm sạch (CSV gốc: 78.18%); accuracy không dùng để chọn mô hình.

CSV archive thực tế có 887 giá trị thiếu `person_emp_length` sau lọc. Bản mô tả đầu vào ghi 889; số đo trên archive là 887, và mã kiểm tra/report dùng con số quan sát được.

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
