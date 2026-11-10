# FIT4013 — Dự đoán sinh viên có nguy cơ bỏ học

Dự đoán tại thời điểm nhập học ba trạng thái cuối thời hạn chuẩn của khóa học: **Dropout**, **Enrolled** và **Graduate**. Đây là bài toán đánh giá học thuật, không phải hệ thống ra quyết định tự động. Hướng dẫn học phần yêu cầu EDA, hai luồng tiền xử lý, ít nhất hai thuật toán, đánh giá và phân tích lỗi.

## Dữ liệu và thời điểm dự đoán

Nguồn: Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021), [Predict Students' Dropout and Academic Success](https://doi.org/10.24432/C5MC89), UCI Machine Learning Repository, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). UCI mô tả dữ liệu nhập học và kết quả hai học kỳ; thời điểm ghi nhận cụ thể của từng trường chưa được xác nhận. Không dùng các chỉ số hai học kỳ, nợ, học phí, học bổng và kinh tế vĩ mô vì có thể chưa biết tại thời điểm dự đoán. Danh sách 18 biến được phép nằm trong `data.py`; cần xác nhận thời điểm thu thập với cơ sở dữ liệu gốc trước triển khai.

Tệp UCI kiểm tra có 4.424 dòng, 37 cột, dấu `;`, không có ô trống, không có dòng trùng hoàn toàn; Dropout 1.421, Enrolled 794, Graduate 2.209. Tên `Daytime/evening attendance` có tab ở cuối trong tệp gốc, được loại khoảng trắng trước khi kiểm định schema. Các thống kê EDA và biểu đồ đặc trưng sau chia chỉ dùng tập train.

## Cài đặt và chạy

Môi trường đã chạy: Python 3.14.6, các phiên bản thư viện ở `requirements.txt`. Cài đặt mới bằng Python 3.14 và pip:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
MPLCONFIGDIR=/tmp/fit4013-matplotlib XDG_CACHE_HOME=/tmp/fit4013-cache python train.py
python3 scripts/verify_results.py
```

Tải dữ liệu vào vị trí khác bằng `python scripts/download_data.py --output /duong/dan/data.csv`, rồi chạy `python train.py --data /duong/dan/data.csv`. `--output` của `train.py` chọn thư mục ảnh; mặc định là `results/`. CSV/JSON mặc định ở `data/experiment/`; nếu đổi `--output`, chúng nằm trong `<output>/data/`. Dùng `--tables-output` để chọn vị trí số liệu riêng. `data/raw/` đã được bỏ qua bởi Git; nếu tải sang nơi khác, tự đảm bảo Git không theo dõi dữ liệu thô. Lệnh tải cần Internet. Một lần chạy ghi đè các tệp kết quả cùng tên.

**Lệnh đã xác minh trong môi trường hiện tại** (Python 3.14.6, môi trường ảo mới cài từ `requirements.txt`; hai biến cache chỉ cần thiết khi thư mục cache người dùng không ghi được):

```sh
MPLCONFIGDIR=/tmp/fit4013-matplotlib XDG_CACHE_HOME=/tmp/fit4013-cache python3 train.py
python3 scripts/verify_results.py
```

Chương trình kiểm tra schema và nhãn, chia phân tầng 80/20 seed 42 và lưu chỉ số dòng train/test. `scripts/verify_results.py` kiểm tra dữ liệu và kết quả tại đường dẫn mặc định, tính lại metric và ma trận từ dự đoán, đồng thời xác nhận các fold không chứa dòng test. Cả bốn thí nghiệm dùng cùng 5 fold phân tầng, seed 42. Mọi imputation, one-hot và scaler được fit trong fold. A dùng StandardScaler; B dùng RobustScaler, còn lại giống nhau. `GridSearchCV` chọn macro-F1 từ K-NN k={3,5,11,21,31} và cây độ sâu={3,5,8,12,None}; lưu cả dropout recall. Tập test chỉ đánh giá sau khi chọn từng cấu hình. Việc so sánh test không chứng minh khác biệt có ý nghĩa thống kê.

## Ứng dụng và notebook

Cài thêm `python -m pip install -r requirements-demo.txt`, rồi chạy `streamlit run demo_app.py` để nhập một hồ sơ hoặc tải CSV dự đoán nhiều hồ sơ và xem kết quả. Tệp CSV cần đủ 18 cột UCI, mã phân loại phải xuất hiện trong tập train; ứng dụng cung cấp tệp mẫu chỉ chứa tên cột. Các điểm đầu ra chưa được hiệu chỉnh xác suất. Mở các notebook `00`–`08` trong `notebooks/` bằng VS Code hoặc Jupyter với kernel của môi trường đã cài. Mỗi notebook chạy độc lập: `04_knn` và `05_cay_quyet_dinh` tự chạy CV trên train; các notebook đánh giá và kiểm tra đọc kết quả từ `data/experiment/`. Chạy `python train.py` trước để tạo lại các kết quả này.

## Cấu trúc và kết quả

- `demo_app.py` và `predict_batch.py`: giao diện Streamlit và kiểm tra/dự đoán CSV theo các bước dữ liệu, EDA, tiền xử lý, mô hình, đánh giá và phân tích lỗi; `notebooks/`: chín notebook theo quy trình, có hai notebook thuật toán riêng.
- `data.py`, `preprocess.py`, `train.py`, `evaluate.py`: kiểm định và chia dữ liệu, hai luồng, huấn luyện và đánh giá.
- `scripts/experiment_figures.py`: vẽ các biểu đồ thí nghiệm từ kết quả đã lưu; `scripts/download_data.py`: tải lại CSV UCI; `requirements.txt`: phiên bản thư viện.
- `data/experiment/audit.json`, `eda_*`: kiểm kê, thống kê, ngoại lệ IQR và biểu đồ. `python scripts/review_analysis.py` bổ sung phân tích mã không rõ/trống, baseline lớp đa số và ví dụ lỗi vào `data/experiment/review_analysis.json`; phân tích này thực hiện sau khi đã xem test, không dùng chọn mô hình.
- `data/experiment/*_cv.csv`, `cv_folds.json`, `selected.json`, `experiment_log.json`: cấu hình và điểm từng fold, phân chia và log.
- `data/experiment/summary.csv`, `metrics.json`, `*_classification_report.csv`, `*_confusion_matrix.csv`, `*_predictions.csv` và ảnh trong `results/`: kết quả test và phân tích lỗi. Chỉ số dòng dự đoán ánh xạ vào tệp CSV gốc sau khi đọc, không phải mã sinh viên.
- `report/course-template/main.tex` và `report/course-template/main.pdf`: báo cáo theo [template giảng viên](https://github.com/TrangLe1912/Template_BCTT_DATN), sử dụng kết quả thực nghiệm; xem hướng dẫn biên dịch bên dưới. `report/report.tex` là bản nháp cũ. `AI_USAGE.md` khai báo sử dụng AI.

Chi tiết phân công thành viên, slide, demo/vấn đáp, repository GitHub được giảng viên chấp thuận và PDF báo cáo theo template là các mục cần nhóm hoàn tất. Không có commit hay push nào được tạo trong phiên này.

Cách tổ chức demo và notebook tham khảo dự án [phân loại spam của buivanchuong9](https://github.com/buivanchuong9/machine_learning-btl); dữ liệu, đặc trưng, mô hình và kết quả trong dự án này thuộc bài toán sinh viên ba lớp.

## Biên dịch báo cáo

Từ thư mục gốc, chạy `python scripts/report_figures.py` để vẽ lại hình sau thí nghiệm. Trong `report/course-template`, chạy `XDG_CACHE_HOME=/tmp/fit4013-tectonic-cache tectonic main.tex` (cần Times New Roman hoặc Tinos). Template dựa trên [TrangLe1912/Template_BCTT_DATN](https://github.com/TrangLe1912/Template_BCTT_DATN), phiên bản `0de06d4fd2af99690d0efe874136afc7d24800b1`.

## Việc nhóm cần hoàn tất

Kiểm tra báo cáo và lớp học; bổ sung minh chứng phân công thực tế trong `TEAM_WORK.md`, chuẩn bị slide và thực hành vấn đáp. Mỗi thành viên cần hiểu pipeline, chỉ số và giới hạn.

## Dàn ý thuyết trình

Bài toán và dữ liệu UCI; EDA và chống rò rỉ; hai luồng tiền xử lý; K-NN và cây quyết định; so sánh CV/test; ma trận nhầm lẫn và lỗi; giới hạn; chạy ứng dụng và trả lời câu hỏi.
