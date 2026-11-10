# Phân công và nhật ký nhóm 9

Nhóm phân công bốn phần việc có phạm vi tương đương. Ngày 09/11/2026, trưởng nhóm xác nhận các phần việc này đã hoàn thành. Xác nhận này do trưởng nhóm cung cấp; thông tin ngày thực hiện, nội dung đóng góp cụ thể và mã commit của từng người vẫn cần bổ sung và đối chiếu. Không mặc định mỗi người đã đóng góp đúng 25%.

## Phân chia file để làm việc trên repository

Đây là phân công dự kiến để các thành viên nhận việc và rà soát; không phải bằng chứng rằng người được giao đã viết các file. Mỗi người cần đọc, chạy, chỉnh sửa phần mình thực sự thực hiện và commit bằng tài khoản của mình; nhật ký bên dưới phải ghi đúng đóng góp thực tế.

### Hoàng Quang Nam — dữ liệu, EDA và điều phối

- Gốc dự án: `.gitignore`, `README.md`, `TEAM_WORK.md`, `AI_USAGE.md`, `requirements.txt`, `requirements-demo.txt`, tệp hướng dẫn học phần PDF.
- Mã và notebook: `student_success/__init__.py`, `student_success/data.py`, `scripts/download_data.py`, `scripts/extended_analysis.py`, `notebooks/00_du_lieu.ipynb`, `notebooks/01_eda.ipynb`.
- Báo cáo: `report/course-template/chapters/chuong1.tex`, `cover.tex`, `settings.tex`, `main.tex`, `.gitignore`, `.latexmkrc`, thư mục `frontmatter/`, thư mục `backmatter/`, `images/logo-dai-nam.png` và bản nháp cũ `report/report.tex`; các đường dẫn ngắn ở mục này nằm trong `report/course-template/` trừ bản nháp.
- Việc cần hiểu: nguồn UCI, chia dữ liệu, phân bố lớp, box plot, tương quan, các trường không dùng. Phối hợp các thành viên để rà soát và biên dịch báo cáo cuối.

### Bùi Huy Bình — tiền xử lý và đặc trưng

- Mã: `student_success/preprocess.py`, `student_success/labels.py`.
- Notebook: `notebooks/02_tien_xu_ly.ipynb`, `notebooks/03_dac_trung.ipynb`.
- Báo cáo: `report/course-template/chapters/chuong2.tex`.
- Việc cần hiểu: pipeline A/B, mã hóa, gộp mã hiếm, nhóm tuổi, hiệu hai điểm, xử lý dữ liệu thiếu và chống rò rỉ trong từng fold.

### Hoàng Nhật Huy — huấn luyện và thí nghiệm

- Mã: `student_success/train.py`, `scripts/compare_criterion.py`, `scripts/evidence_analysis.py`.
- Notebook: `notebooks/04_knn.ipynb`, `notebooks/05_cay_quyet_dinh.ipynb`.
- Báo cáo: `report/course-template/chapters/chuong3.tex`, `report/course-template/chapters/chuong4.tex`.
- Việc cần hiểu: K-NN, cây, lưới tham số, cách chọn macro-F1 bằng CV, baseline, ablation, bootstrap và giới hạn của test đã xem.

### Nguyễn Văn Linh — đánh giá, giao diện và kết quả

- Mã: `student_success/evaluate.py`, `student_success/predict_batch.py`, `app.py`, `scripts/verify_results.py`, `scripts/review_analysis.py`, `scripts/roc_analysis.py`, `scripts/report_figures.py`, `scripts/experiment_figures.py`.
- Notebook: `notebooks/06_danh_gia.ipynb`, `notebooks/07_phan_tich_loi.ipynb`, `notebooks/08_kiem_tra_ket_luan.ipynb`.
- Báo cáo: `report/course-template/chapters/chuong5.tex`, `report/course-template/chapters/chuong6.tex`.
- Đầu ra đã tạo: toàn bộ `data/experiment/`, `data/previous_experiment/`, `results/`, `report/course-template/images/results/` và `report/course-template/main.pdf`. Các đầu ra này cần được kiểm tra/tạo lại sau khi mã của mọi người đã ghép xong.
- Việc cần hiểu: ma trận nhầm lẫn, precision/recall/F1, ROC-AUC, phân tích lỗi và thao tác giao diện.

### Thứ tự ghép và những file không dán văn bản

1. Nam đưa `.gitignore`, dependencies, dữ liệu và khung báo cáo lên trước; Bình bổ sung tiền xử lý; Huy bổ sung huấn luyện; Linh bổ sung đánh giá và giao diện. Các thành viên có thể làm trên nhánh riêng, sau đó ghép và kiểm tra lại.
2. File `.png`, `.pdf` là file nhị phân: chép nguyên file hoặc chạy lệnh để tạo lại, không dán nội dung bằng trình soạn thảo. `main.pdf` chỉ biên dịch sau khi đã có đầy đủ mã nguồn và hình.
3. `data/raw/data.csv` bị `.gitignore` bỏ qua; chạy `python scripts/download_data.py` để tải về, không coi file rỗng trong bản khung là dữ liệu dùng được. Không đưa `.venv/`, cache hoặc `.DS_Store` lên Git.
4. Sau khi ghép đủ mã: chạy các lệnh trong `README.md`, rồi chạy `python scripts/verify_results.py`. Các file kết quả trong `data/experiment/` cần khớp dữ liệu và mô hình của cùng một lần chạy.
5. Git không theo dõi thư mục rỗng. Các file rỗng chỉ là khung; thành viên cần điền/cập nhật phần thực sự phụ trách và dùng tài khoản Git của chính mình khi commit.

## 1. Hoàng Quang Nam — dữ liệu, EDA và điều phối

**Trạng thái:** Hoàn thành theo xác nhận của trưởng nhóm; minh chứng commit cần bổ sung.

- Kiểm tra nguồn và giấy phép UCI, cấu trúc CSV, phân phối lớp, giá trị thiếu theo mã và các cột có thể rò rỉ thông tin.
- Rà soát `student_success/data.py`, các biểu đồ EDA và chương 1 báo cáo; sửa hoặc bổ sung nếu phát hiện sai sót.
- Điều phối tiến độ, tổng hợp nhật ký và đối chiếu minh chứng đóng góp của nhóm.
- **Minh chứng cần bổ sung:** kiểm tra được dữ liệu và chương 1, ghi kết quả kiểm tra và liên kết commit vào nhật ký; giải thích được định nghĩa bài toán và thời điểm dự đoán.

## 2. Bùi Huy Bình — tiền xử lý và đặc trưng

**Trạng thái:** Hoàn thành theo xác nhận của trưởng nhóm; minh chứng commit cần bổ sung.

- Rà soát danh sách đặc trưng tại thời điểm nhập học, mã phân loại, xử lý thiếu và mã hóa one-hot.
- Kiểm tra `student_success/preprocess.py`, cách gộp mã hiếm, tạo đặc trưng và StandardScaler, và cách fit các bước trong fold huấn luyện.
- Rà soát, sửa hoặc bổ sung chương 2 báo cáo và phần giải thích chống rò rỉ.
- **Minh chứng cần bổ sung:** có kết quả kiểm tra và commit thực tế trong nhật ký; giải thích được hai luồng, một ví dụ mã hóa và lý do loại các cột học kỳ.

## 3. Hoàng Nhật Huy — huấn luyện và thí nghiệm

**Trạng thái:** Hoàn thành theo xác nhận của trưởng nhóm; minh chứng commit cần bổ sung.

- Rà soát `student_success/train.py`, K-NN, cây quyết định, lưới siêu tham số và cách chọn theo macro-F1.
- Chạy lại thí nghiệm theo README; đối chiếu các fold, tham số được chọn và log với kết quả lưu.
- Rà soát, sửa hoặc bổ sung phần thiết lập và so sánh thí nghiệm trong chương 4.
- **Minh chứng cần bổ sung:** ghi lệnh chạy, kết quả đối chiếu và commit thực tế vào nhật ký; giải thích được vì sao chọn từng cấu hình.

## 4. Nguyễn Văn Linh — đánh giá và phân tích lỗi

**Trạng thái:** Hoàn thành theo xác nhận của trưởng nhóm; minh chứng commit cần bổ sung.

- Rà soát `student_success/evaluate.py`, bảng metric, dự đoán và ma trận nhầm lẫn của bốn tổ hợp.
- Chạy `python3 scripts/verify_results.py`; đối chiếu recall Dropout, các trường hợp bỏ sót và Enrolled bị nhầm.
- Rà soát, sửa hoặc bổ sung phần phân tích lỗi, hạn chế và kết luận trong chương 5–6.
- **Minh chứng cần bổ sung:** ghi kết quả xác minh và commit thực tế vào nhật ký; giải thích được precision, recall, macro-F1 và ít nhất một ma trận nhầm lẫn.

## Việc chung

Mỗi người rà soát nội dung phần phụ trách, tham gia hoàn thiện slide và tập vấn đáp; tất cả cần hiểu toàn bộ pipeline. Sau khi có nhật ký thực tế, nhóm đối chiếu khối lượng và điều chỉnh nếu một phần nặng hơn. Số commit không phải thước đo duy nhất của đóng góp.

## Nhật ký công việc thực tế

| Ngày | Người thực hiện | Công việc và kết quả | Cách kiểm tra | Mã hoặc liên kết commit | Người rà soát |
|---|---|---|---|---|---|
| Chưa cung cấp | Hoàng Quang Nam | Đã tạo commit README, theo thông tin trưởng nhóm | Chưa đối chiếu lịch sử Git | Cần bổ sung | Cần bổ sung |
| 09/11/2026 | Trưởng nhóm xác nhận | Bốn phần việc đã hoàn thành theo phân công | Chưa đối chiếu minh chứng từng người | Cần bổ sung | Cần bổ sung |

Theo thông tin trưởng nhóm, các thành viên còn lại chưa có commit. Trưởng nhóm đã xác nhận bốn phần việc hoàn thành ngày 09/11/2026; đây là ngày xác nhận, không suy ra ngày thực hiện. Bổ sung từng dòng ghi đúng phần đã làm, kết quả, ngày thực hiện và mã commit sau khi đối chiếu. OpenAI Codex đã hỗ trợ xây dựng mã, chạy thí nghiệm và soạn báo cáo như nêu trong `AI_USAGE.md`; không gán toàn bộ phần đó cho một cá nhân nếu chưa có bằng chứng về công việc người đó thực hiện.

## Xác nhận với giảng viên

Đường liên kết nhúng tại Đề tài 8 trong PDF chỉ tới UCI 697. Phạm vi dự đoán tại thời điểm nhập học và việc loại các biến học kỳ cần nhóm trao đổi với giảng viên để xác nhận có phù hợp yêu cầu đề tài; chưa có xác nhận được cung cấp.

Nội dung nhóm có thể gửi: “Thưa cô, nhóm 9 dùng bộ UCI 697 được liên kết trong Đề tài 8 và dự kiến dự đoán ba trạng thái tại thời điểm nhập học, nên chỉ dùng thông tin sẵn có lúc nhập học và loại các chỉ số học kỳ. Cô cho nhóm hỏi phạm vi này có phù hợp với đề tài và được cô đồng ý không ạ?”
