# Nhập môn học máy (FIT4013) — Dự đoán sinh viên có nguy cơ bỏ học

Dự án dự đoán ba trạng thái Dropout, Enrolled, Graduate tại thời điểm nhập học từ 18 trường của bộ [UCI 697](https://doi.org/10.24432/C5MC89), giấy phép CC BY 4.0. Tệp nhỏ `data/raw/data.csv` đã được lưu trong repository, nên không cần mạng để chạy; `python scripts/download_data.py` tải lại tệp từ UCI; dùng `--output` để tránh ghi đè bản đã lưu.

## Cài đặt

Kết quả đã lưu được tạo với Python 3.14.6. Lần chạy lại trên macOS ARM với Python 3.12.3 và các thư viện ghim cho cùng cấu hình và dự đoán ở cả bốn tổ hợp; xem `data/experiment/reproducibility_python312_macos.json` và phần thảo luận trong báo cáo. Kết quả trên môi trường khác vẫn cần được kiểm tra riêng. Cài đặt:

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

## Thí nghiệm khám phá CategoricalNB

Thí nghiệm riêng kiểm tra biểu diễn từng trường phân loại và rời rạc hóa biến số, không thay đổi bốn tổ hợp A/B × K-NN/Cây. Chạy từ gốc repository với các thư viện ghim:

```sh
python scripts/categorical_nb_experiment.py
python -m unittest discover -s tests -v
python scripts/verify_results.py
```

Script đọc split và fold đã lưu, chọn theo macro-F1 CV trên train; fit lại bốn cấu hình cũ trên đúng fold để tính thêm chỉ số Dropout, kiểm tra từng điểm macro-F1 với tệp gốc. Các tệp mới được ghi vào `data/experiment/categorical_nb/` (grid đầy đủ, fold, so sánh CV, cấu hình, log phiên bản, dự đoán và chỉ số test); ảnh được ghi vào `report/course-template/images/results/categorical_nb_cv_comparison.png` và `CategoricalNB_confusion_matrix.png`. Chạy lại ghi đè các tệp NB này, không ghi đè bốn mô hình cũ. Mã chưa biết được dành mã 0; các bước điền thiếu, mã hóa và tính biên đều fit trong fold. Quantile không subsample, seed 42; cảnh báo biên trùng có thể làm giảm số khoảng thực tế. Kết quả phụ thuộc phiên bản trong `run_log.json`.

Kết quả đã chạy: NB có macro-F1 CV 0,5033 so với 0,4661 của B/K-NN, recall Dropout tương ứng 0,5303 so với 0,5030. Test đã được xem trong phát triển nên chỉ dùng khám phá: NB macro-F1 0,4627, recall Dropout 0,5141, recall Enrolled 0,1635. Các điểm chọn theo CV không chứng minh cải thiện trên tập độc lập.
