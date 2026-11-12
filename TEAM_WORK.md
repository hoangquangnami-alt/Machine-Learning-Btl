# Phân công và nhật ký nhóm

Bảng phân công dưới đây phân chia trách nhiệm rà soát và trình bày cho bốn thành viên. Đây là phân công hiện tại, không phải xác nhận ai đã thực hiện công việc trong quá khứ; mỗi thành viên cần kiểm tra nội dung được giao và thống nhất tiến độ với nhóm. Lịch sử Git phía dưới chỉ ghi các thay đổi trong repository.

## Phân công công việc

| Thành viên | Phần phụ trách | Tài liệu và mã cần rà soát |
|---|---|---|
| Hoàng Quang Nam | Dữ liệu và phạm vi bài toán: kiểm tra nguồn, 18 trường tại thời điểm nhập học, chia dữ liệu và phân tích mô tả; trình bày dữ liệu và giới hạn. | `student_success/data.py`, `data/raw/`, báo cáo chương 1–2 |
| Bùi Huy Bình | Tiền xử lý và đặc trưng: kiểm tra luồng A/B, chống rò rỉ, xử lý dữ liệu thiếu và đặc trưng dẫn xuất; trình bày lựa chọn tiền xử lý. | `student_success/preprocess.py`, notebook tiền xử lý, báo cáo chương 2 |
| Hoàng Nhật Huy | Huấn luyện và thí nghiệm: kiểm tra K-NN, Decision Tree, lưới tham số và thí nghiệm CategoricalNB; trình bày quy trình chọn mô hình. | `student_success/train.py`, `student_success/categorical_nb.py`, `scripts/categorical_nb_experiment.py`, báo cáo chương 3–4 |
| Nguyễn Văn Linh | Đánh giá và ứng dụng: kiểm tra chỉ số, dự đoán, sai lệch, tái lập kết quả và demo; trình bày kết luận và giới hạn triển khai. | `student_success/evaluate.py`, `scripts/verify_results.py`, `app.py`, báo cáo chương 4–6 |

Mỗi thành viên chịu trách nhiệm đọc lại phần được giao, kiểm tra bằng chứng tương ứng và chuẩn bị giải thích; thời gian thực hiện và đóng góp thực tế do từng người tự xác nhận với nhóm.

## Nhật ký Git

Area ghi nhóm đường dẫn xuất hiện trong commit, không suy diễn trách nhiệm.

| Ngày | Tác giả | Commit + thông điệp | Area |
|---|---|---|---|
| 2026-11-10 | hoangquangnami-alt | `9186a02` vá lỗi lần cuối | `.gitattributes`, `.gitignore`, `AI_USAGE.md`, `README.md`, `TEAM_WORK.md`, `app.py`, `data/`, `notebooks/`, `report/`, `requirements.txt`, `scripts/`, `student_success/` |
| 2026-11-10 | hoangquangnami-alt | `65539a6` Add team roles and task log for project | `TEAM_WORK.md` |
| 2026-11-10 | hoangquangnami-alt | `63fa0d6` Create README.md for project documentation | `README.md` |
| 2026-11-10 | hoangquangnami-alt | `7307b1f` Add AI usage documentation | `AI_USAGE.md` |
| 2026-11-10 | hoangquangnami-alt | `1a5c74a` Add data loading and validation functions | `student_success/` |
| 2026-11-10 | hoangquangnami-alt | `e7c99dc` Add initial docstring for student success pipeline | `student_success/` |
| 2026-11-10 | hoangquangnami-alt | `eef2a41` thêm báo cáo latex theo template của cô | Không có thay đổi tệp trong commit |
| 2026-11-10 | hoangquangnami-alt | `ae6c936` thêm báo cáo latex theo template của cô | `report/` |
| 2026-11-10 | Nhật Huy | `b37552e` Delete report/course-template directory | `report/` |
| 2026-11-10 | huybinh7826 | `18746c5` Delete report/course-template/chapters directory | `report/` |
| 2026-11-10 | hoangquangnami-alt | `1446b03` Delete report/course-template/backmatter directory | `report/` |
| 2026-11-10 | hoangquangnami-alt | `dd4be2a` thêm các notebook | Không có thay đổi tệp trong commit |
| 2026-11-10 | hoangquangnami-alt | `ad5b8bb` thêm các notebook | `notebooks/` |
| 2026-11-10 | Nhật Huy | `eb5d279` Delete notebooks/08_kiem_tra_ket_luan.ipynb | `notebooks/` |
| 2026-11-10 | Nhật Huy | `02f8f55` Delete notebooks/07_phan_tich_loi.ipynb | `notebooks/` |
| 2026-11-10 | Nhật Huy | `ab273ef` Delete notebooks/06_danh_gia.ipynb | `notebooks/` |
| 2026-11-10 | huybinh7826 | `d82fadd` Delete notebooks/05_cay_quyet_dinh.ipynb | `notebooks/` |
| 2026-11-10 | huybinh7826 | `b34ea16` Delete notebooks/04_knn.ipynb | `notebooks/` |
| 2026-11-10 | huybinh7826 | `4a437dc` Delete notebooks/03_dac_trung.ipynb | `notebooks/` |
| 2026-11-10 | hoangquangnami-alt | `3c8220e` Delete notebooks/02_tien_xu_ly.ipynb | `notebooks/` |
| 2026-11-10 | hoangquangnami-alt | `40955cb` Delete notebooks/01_eda.ipynb | `notebooks/` |
| 2026-11-10 | hoangquangnami-alt | `a7d145d` Delete notebooks/00_du_lieu.ipynb | `notebooks/` |
| 2026-11-10 | hoangquangnami-alt | `f6a718d` add data | `data/` |
| 2026-11-10 | hoangquangnami-alt | `76da313` Delete data directory | `data/` |
| 2026-11-10 | hoangquangnami-alt | `4d88d56` Add KNN classification report CSV file | `data/` |
| 2026-11-10 | Nhật Huy | `72cf9f7` Add files via upload | `requirements-demo.txt` |
| 2026-11-10 | Nhật Huy | `2163af5` Add initial package dependencies to requirements.txt | `requirements.txt` |
| 2026-11-10 | Nhật Huy | `5f5c74a` add results | `results/` |
| 2026-11-10 | Nhật Huy | `195dda2` Delete results directory | `results/` |
| 2026-11-10 | Nhật Huy | `56833e6` Add training script with EDA and model evaluation | `student_success/` |
| 2026-11-10 | Nguyen Van Linh | `d8dafbc` Add files via upload | `notebooks/` |
| 2026-11-10 | Nguyen Van Linh | `8637709` Delete notebooks/08_kiem_tra_ket_luan.ipynb | `notebooks/` |
| 2026-11-10 | Nguyen Van Linh | `c857b18` Delete notebooks/06_danh_gia.ipynb | `notebooks/` |
| 2026-11-10 | Nguyen Van Linh | `e034f2c` Add verification script for model results and metrics | `scripts/` |
| 2026-11-10 | Nguyen Van Linh | `45bc303` Update roc_analysis.py | `scripts/` |
| 2026-11-10 | Nguyen Van Linh | `dd37abf` Add review analysis script for baseline evaluation | `scripts/` |
| 2026-11-10 | Nguyen Van Linh | `a8d7ae1` Update report_figures.py | `scripts/` |
| 2026-11-10 | Nguyen Van Linh | `5218583` Add extended analysis script for dropout and errors | `scripts/` |
| 2026-11-10 | Nguyen Van Linh | `40f618a` Add script for generating experiment figures | `scripts/` |
| 2026-11-10 | Nguyen Van Linh | `6db8e5d` Add evidence analysis script for model evaluation | `scripts/` |
| 2026-11-10 | Nguyen Van Linh | `b22ae59` Add script to download UCI dataset | `scripts/` |
| 2026-11-10 | Nguyen Van Linh | `d726d9c` Update compare_criterion.py | `scripts/` |
| 2026-11-10 | Nguyen Van Linh | `07bc47e` Update app.py | `app.py` |
| 2026-11-10 | Nguyen Van Linh | `79019c8` Add input validation and prediction functions | `student_success/` |
| 2026-11-10 | Nguyen Van Linh | `66d3db1` Implement model evaluation function with metrics saving | `student_success/` |
| 2026-11-10 | huybinh7826 | `62e65d0` Add files via upload | `notebooks/` |
| 2026-11-10 | huybinh7826 | `dedebaf` Delete notebooks/02_tien_xu_ly.ipynb | `notebooks/` |
| 2026-11-10 | huybinh7826 | `cd6cd16` Add labels for student classification data | `student_success/` |
| 2026-11-10 | huybinh7826 | `56a9aab` Add preprocessing for baseline and engineered feature flows | `student_success/` |
| 2026-11-10 | Nhật Huy | `6ce4720` update Huấn luyện, thí nghiệm | `TEAM_WORK.md`, `scripts/`, `student_success/` |
| 2026-11-10 | hoangquangnami-alt | `7f55a6f` update version 1 | `AI_USAGE.md`, `README.md`, `app.py`, `data/`, `notebooks/`, `report/`, `requirements.txt`, `results/` |
| 2026-11-09 | hoangquangnami-alt | `e2ce676` Create README.md | `README.md` |

## Xác nhận với giảng viên

TODO(human): Chưa nhận xác nhận phạm vi dự đoán tại thời điểm nhập học.
