# FIT4013 — Dự đoán sinh viên có nguy cơ bỏ học

Dự án dự đoán ba trạng thái Dropout, Enrolled, Graduate tại thời điểm nhập học từ 18 trường của bộ [UCI 697](https://doi.org/10.24432/C5MC89), giấy phép CC BY 4.0. Tệp nhỏ `data/raw/data.csv` đã được lưu trong repository, nên không cần mạng để chạy; `python scripts/download_data.py` tải lại tệp từ UCI; dùng `--output` để tránh ghi đè bản đã lưu.

## Cài đặt

Kết quả đã lưu được tạo với Python 3.14.6; theo lượt rà soát trước, các thư viện ghim cũng đã được kiểm thử với Python 3.12.3. Khác biệt cây quyết định trên 3.12.3 chưa được tái lập trong lượt rà soát này; xem thảo luận trong báo cáo. Cài đặt:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-demo.txt
```

## Thí nghiệm

Chia phân tầng 80/20, seed 42; cả bốn tổ hợp dùng cùng 5 fold phân tầng trên train. Mọi bước tiền xử lý được fit trong từng fold. GridSearchCV chọn macro-F1 trung bình:

- K-NN: `n_neighbors={5,11,21}`, `weights={uniform,distance}`, `p={1,2}`.
- Cây: `max_depth={3,8,12}`, `min_samples_leaf={1,10}`, `class_weight={None,balanced}`.
- A: điền median và StandardScaler cho biến số; điền mode và one-hot cho biến phân loại.
- B: như A, thêm gộp mã hiếm qua `OneHotEncoder(min_frequency=0.01)` và hai đặc trưng số chuẩn hóa `age_band`, `admission_grade_difference`.

Chạy đầy đủ theo thứ tự dưới đây từ thư mục gốc. **Các lệnh mặc định ghi đè tệp cùng tên trong `data/experiment/` và ảnh**; lưu bản sao trước khi chạy.

```sh
python -m student_success.train
python scripts/evidence_analysis.py
python scripts/roc_analysis.py
python scripts/review_analysis.py
python scripts/extended_analysis.py
python scripts/compare_criterion.py
python scripts/verify_results.py
python scripts/report_figures.py
python scripts/experiment_figures.py
```

Huấn luyện ghi ảnh vào `results/`, chỉ số, phân chia, dự đoán, CV và log vào `data/experiment/`. Evidence ghi baseline, ablation, bootstrap và phân tích đặc trưng; ROC ghi AUC; review ghi kiểm toán mô tả; extended ghi EDA, lỗi, độ quan trọng và ảnh; criterion ghi so sánh Gini/entropy. Verify đọc kết quả mặc định và kiểm tra tính nhất quán, không ghi tệp. Hai script cuối vẽ ảnh báo cáo.

Chạy an toàn vào thư mục tạm (lệnh verify vẫn kiểm tra kết quả gốc):

```sh
python -m student_success.train --data data/raw/data.csv --output /tmp/fit-images --tables-output /tmp/fit-tables
export FIT_RESULTS=/tmp/fit-tables FIT_IMAGES=/tmp/fit-report-images
python scripts/evidence_analysis.py
python scripts/roc_analysis.py
python scripts/review_analysis.py
python scripts/extended_analysis.py
python scripts/compare_criterion.py
python scripts/verify_results.py
python scripts/report_figures.py
python scripts/experiment_figures.py
unset FIT_RESULTS FIT_IMAGES
```

Nếu chỉ đổi `--output`, bảng mặc định nằm ở `<output>/data`. Các phân tích dùng `FIT_RESULTS` để đọc và ghi bảng, và `FIT_IMAGES` cho ảnh báo cáo; cần tạo bảng bằng train trước. Xem `scripts/tree_sensitivity.py` để phân tích độ nhạy cây mà không thay kết quả gốc.

## Demo và notebook

Chạy `streamlit run app.py`; giao diện nhận một hồ sơ hoặc CSV với 18 cột. Điểm đầu ra chưa được hiệu chỉnh xác suất. Notebook `00`–`08` trong `notebooks/` phân tích, chạy CV hoặc đọc kết quả đã lưu.

## Cấu trúc

- `student_success/data.py`, `preprocess.py`, `train.py`, `evaluate.py`, `labels.py`, `predict_batch.py`: tải và chia dữ liệu, tiền xử lý, huấn luyện, đánh giá, nhãn và dự đoán.
- `scripts/`: phân tích, kiểm tra, tải lại dữ liệu và vẽ hình; `app.py`: ứng dụng Streamlit.
- `data/raw/`: dữ liệu UCI được commit; `data/experiment/`: kết quả thực nghiệm đã báo cáo.
- `results/`: ảnh mặc định do huấn luyện tạo; `report/course-template/images/results/`: ảnh dùng trong báo cáo.
- `report/course-template/`: nguồn LaTeX và PDF; `notebooks/`: notebook; `TEAM_WORK.md`, `AI_USAGE.md`: phân công và khai báo AI.

## Biên dịch báo cáo

Từ `report/course-template`, chạy `XDG_CACHE_HOME=/tmp/fit4013-tectonic-cache tectonic main.tex` (cần Times New Roman hoặc Tinos). Template dựa trên [TrangLe1912/Template_BCTT_DATN](https://github.com/TrangLe1912/Template_BCTT_DATN).
