"""Ứng dụng phân tích và dự đoán bằng bốn mô hình đã huấn luyện."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from student_success.data import CATEGORICAL, NUMERIC, LABELS, SEED, split
from student_success.preprocess import build_preprocessor
from student_success.predict_batch import COLUMNS, predict_batch
from student_success.labels import label_for

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/raw/data.csv'
OUT = ROOT / 'data/experiment'
IMAGES = ROOT / 'results'
MODEL_LABELS = {'A_knn': 'Luồng A · K-NN', 'A_tree': 'Luồng A · Cây quyết định',
                'B_knn': 'Luồng B · K-NN', 'B_tree': 'Luồng B · Cây quyết định'}
FIELD_NAMES = {'Application order':'Thứ tự nguyện vọng', 'Previous qualification (grade)':'Điểm văn bằng trước', 'Admission grade':'Điểm tuyển sinh', 'Age at enrollment':'Tuổi lúc nhập học', 'Marital status':'Tình trạng hôn nhân', 'Application mode':'Hình thức đăng ký',
    'Course':'Mã ngành', 'Daytime/evening attendance':'Ca học (mã UCI)',
    'Previous qualification':'Văn bằng trước (mã UCI)', 'Nacionality':'Quốc tịch (mã UCI)',
    "Mother's qualification":'Trình độ của mẹ (mã UCI)', "Father's qualification":'Trình độ của cha (mã UCI)',
    "Mother's occupation":'Nghề của mẹ (mã UCI)', "Father's occupation":'Nghề của cha (mã UCI)',
    'Displaced':'Chuyển nơi ở (mã UCI)', 'Educational special needs':'Nhu cầu giáo dục đặc biệt (mã UCI)',
    'Gender':'Giới tính (mã UCI)', 'International':'Sinh viên quốc tế (mã UCI)'}


def read_source():
    if not DATA.exists():
        raise FileNotFoundError('Chưa có data/raw/data.csv. Chạy python scripts/download_data.py.')
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    audit = json.loads((OUT/'audit.json').read_text())
    if digest != audit['sha256']:
        raise ValueError('Dữ liệu không khớp kết quả đã lưu. Hãy chạy lại toàn bộ python -m student_success.train trước khi sử dụng.')
    df = pd.read_csv(DATA, sep=';'); df.columns = df.columns.str.strip()
    train, test = split(df)
    if test.tolist() != pd.read_csv(OUT/'test_indices.csv').row_index.tolist():
        raise ValueError('Phép chia test không khớp kết quả đã lưu.')
    return df, train, test


@st.cache_resource(show_spinner='Đang chuẩn bị mô hình từ tập huấn luyện...')
def train_model(name, data_hash, selected_hash):
    df, train, test = read_source()
    selected = json.loads((OUT/'selected.json').read_text())[name]['params']
    estimator = KNeighborsClassifier() if name.endswith('knn') else DecisionTreeClassifier(random_state=SEED)
    estimator.set_params(**{key.removeprefix('model__'): value for key, value in selected.items()})
    model = Pipeline([('preprocess', build_preprocessor(name[0])), ('model', estimator)])
    model.fit(df.loc[train, COLUMNS], df.loc[train, 'Target'])
    saved = pd.read_csv(OUT/f'{name}_predictions.csv')
    if saved.row_index.tolist() != test.tolist() or not np.array_equal(model.predict(df.loc[test, COLUMNS]), saved.predicted.to_numpy()):
        raise ValueError('Dự đoán kiểm tra không khớp kết quả đã lưu; kiểm tra phiên bản thư viện và chạy lại python -m student_success.train.')
    return model


def main():
    st.set_page_config(page_title='FIT4013 | Dự đoán sinh viên', page_icon='🎓', layout='wide')
    st.title('🎓 Dự đoán trạng thái sinh viên')
    st.caption('Dự đoán tại thời điểm nhập học · Ba lớp UCI · FIT4013')
    st.info('Mô hình được huấn luyện trên dữ liệu UCI của một cơ sở; chưa được kiểm định cho cơ sở khác. Không dùng dự đoán để tự động quyết định hỗ trợ sinh viên.')
    try:
        df, train, _ = read_source()
        selected_path = OUT/'selected.json'
        selected_hash = hashlib.sha256(selected_path.read_bytes()).hexdigest()
        data_hash = hashlib.sha256(DATA.read_bytes()).hexdigest()
        train_data = df.loc[train]
    except (FileNotFoundError, KeyError, ValueError) as exc:
        st.error(str(exc)); st.stop()
    predict_tab, results_tab, data_tab = st.tabs(['Dự đoán', 'Kết quả mô hình', 'Dữ liệu & cách làm'])
    with predict_tab:
        st.subheader('1. Chọn mô hình dự đoán')
        name = st.selectbox('Mô hình', list(MODEL_LABELS), index=2, format_func=MODEL_LABELS.get)
        st.caption('Chưa biết chọn gì? K-NN B là mô hình được chọn sẵn theo macro-F1 kiểm định chéo. Xem thẻ Kết quả mô hình để so sánh.')
        st.subheader('2. Nhập thông tin sinh viên')
        st.caption('Chọn đầy đủ thông tin theo hồ sơ. Các lựa chọn lấy theo bảng mã của bộ dữ liệu UCI.')
        st.markdown('[Xem giải thích các mã trong dữ liệu UCI](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success). Một số lựa chọn chưa có tên sẽ hiển thị mã gốc.')
        with st.form('student_form'):
            left, right = st.columns(2)
            values = {}
            with left:
                st.markdown('**Thông tin số**')
                values['Application order'] = st.number_input('Thứ tự nguyện vọng (0 là đầu tiên, 9 là cuối)', min_value=0, max_value=9, value=None, placeholder='Nhập từ 0 đến 9')
                for col, label in [('Previous qualification (grade)','Điểm văn bằng trước (0–200)'),('Admission grade','Điểm tuyển sinh (0–200)')]:
                    values[col] = st.number_input(label, min_value=0.0, max_value=200.0, value=None, step=0.1, placeholder='Nhập điểm')
                values['Age at enrollment'] = st.number_input('Tuổi lúc nhập học', min_value=15, max_value=100, value=None, placeholder='Nhập tuổi')
                st.markdown('**Hồ sơ đăng ký**')
                for col in CATEGORICAL[:6]:
                    options = sorted(int(v) for v in train_data[col].dropna().unique())
                    values[col] = st.selectbox(FIELD_NAMES[col].replace(' (mã UCI)', ''), options, index=None, placeholder='Chọn một lựa chọn', format_func=lambda code, field=col: label_for(field, code), key=col)
            with right:
                st.markdown('**Thông tin gia đình và cá nhân**')
                for col in CATEGORICAL[6:]:
                    options = sorted(int(v) for v in train_data[col].dropna().unique())
                    values[col] = st.selectbox(FIELD_NAMES[col].replace(' (mã UCI)', ''), options, index=None, placeholder='Chọn một lựa chọn', format_func=lambda code, field=col: label_for(field, code), key=col)
            submitted = st.form_submit_button('Xem dự đoán', type='primary', use_container_width=True)
        if submitted:
            try:
                missing = [FIELD_NAMES.get(col, col).replace(' (mã UCI)', '') for col in COLUMNS if values.get(col) is None]
                if missing:
                    raise ValueError('Vui lòng nhập đầy đủ thông tin: ' + ', '.join(missing))
                model = train_model(name, data_hash, selected_hash)
                sample = pd.DataFrame([values], columns=COLUMNS)
                prediction = model.predict(sample)[0]
                scores = model.predict_proba(sample)[0]
                st.subheader('3. Kết quả dự đoán')
                st.success(f"Kết quả dự đoán: {dict(Dropout='Bỏ học', Enrolled='Còn theo học', Graduate='Tốt nghiệp')[prediction]}")
                st.caption('Các điểm dưới đây là đầu ra của mô hình, chưa được hiệu chỉnh xác suất. Không xem đây là xác suất chính xác hay chẩn đoán cá nhân.')
                score_map = dict(zip(model.classes_, scores))
                for label in LABELS:
                    st.write(f"{dict(Dropout='Bỏ học', Enrolled='Còn theo học', Graduate='Tốt nghiệp')[label]}: {score_map[label]:.1%}")
                    st.progress(float(score_map[label]))
            except (FileNotFoundError, KeyError, ValueError) as exc:
                st.error(str(exc))
        st.divider()
        st.subheader('Dự đoán nhiều hồ sơ từ CSV')
        st.caption('Tệp UTF-8 dùng dấu phẩy hoặc chấm phẩy, có đủ 18 cột theo tên UCI; có thể thêm cột mã hồ sơ. Mỗi lần tối đa 10.000 dòng. Dữ liệu tải lên được xử lý trong phiên ứng dụng.')
        blank = pd.DataFrame(columns=['student_id'] + COLUMNS).to_csv(index=False).encode('utf-8-sig')
        st.download_button('Tải tệp mẫu cột', blank, file_name='mau_ho_so.csv', mime='text/csv')
        delimiter = st.radio('Dấu phân cách CSV', [',', ';'], horizontal=True, format_func=lambda x: 'Dấu phẩy (,)' if x == ',' else 'Chấm phẩy (;)')
        uploaded = st.file_uploader('Tải tệp hồ sơ CSV', type=['csv'])
        if uploaded is not None:
            if uploaded.size > 10 * 1024 * 1024:
                st.error('Tệp vượt quá 10 MB.')
            else:
                try:
                    frame = pd.read_csv(uploaded, sep=delimiter, encoding='utf-8-sig')
                    st.write(f'Đã đọc {len(frame)} hồ sơ.')
                    if st.button('Dự đoán tệp', type='primary'):
                        model = train_model(name, data_hash, selected_hash)
                        result = predict_batch(frame, train_data, model)
                        st.dataframe(result, hide_index=True, use_container_width=True)
                        st.download_button('Tải kết quả CSV', result.to_csv(index=False).encode('utf-8-sig'), file_name='ket_qua_du_doan.csv', mime='text/csv')
                except (ValueError, FileNotFoundError, KeyError, UnicodeDecodeError, pd.errors.ParserError) as exc:
                    st.error(str(exc))
    with data_tab:
        st.subheader('Nguồn dữ liệu và phạm vi dự đoán')
        st.markdown('Dữ liệu UCI **Predict Students’ Dropout and Academic Success** gồm ba lớp: bỏ học, còn học và tốt nghiệp. Bài toán dự đoán tại thời điểm nhập học; thời điểm thu thập một số trường cần xác minh trước khi áp dụng thực tế.')
        st.markdown('[Nguồn dữ liệu và bảng mã UCI](https://doi.org/10.24432/C5MC89)')
        audit = json.loads((OUT / 'audit.json').read_text())
        a, b, c, d = st.columns(4)
        a.metric('Số dòng', f'{len(df):,}')
        b.metric('Số cột gốc', len(df.columns))
        c.metric('Ô thiếu', sum(audit['missing'].values()))
        d.metric('Dòng trùng hoàn toàn', audit['duplicate_rows'])
        st.write('Chia phân tầng: 3.539 mẫu train và 885 mẫu test, seed 42. Dòng test không được dùng để fit tiền xử lý hoặc chọn siêu tham số.')
        st.dataframe(df.head(), hide_index=True, use_container_width=True)
    with data_tab:
        st.subheader('Khám phá dữ liệu trên tập train')
        st.caption('Các biểu đồ đặc trưng chỉ dùng train; mỗi lớp được thể hiện riêng.')
        st.image(str(ROOT/'report/course-template/images/results/class_distribution.png'), caption='Phân phối ba lớp trên tập train.')
        st.image(str(ROOT/'report/course-template/images/results/histograms.png'), caption='Phân phối bốn đặc trưng số trên tập train.')
        st.image(str(ROOT/'report/course-template/images/results/courses.png'), caption='Số mẫu theo mã ngành trên train.')
        st.image(str(ROOT/'report/course-template/images/results/dropout_by_group.png'), caption='Tỷ lệ Dropout theo nhóm trên tập train; số mẫu được ghi cùng từng nhóm.')
        st.dataframe(pd.read_csv(OUT/'eda_numeric_summary.csv', index_col=0), use_container_width=True)
    with data_tab:
        st.subheader('Tiền xử lý và đặc trưng')
        st.markdown('''Hai luồng dùng cùng **18 đặc trưng**: 14 phân loại và 4 đặc trưng số. Các trường học kỳ và các trường có thể chưa có khi nhập học được loại khỏi đầu vào.

- Đặc trưng số: điền thiếu bằng trung vị và chuẩn hóa bằng StandardScaler trong cả hai luồng.
- Đặc trưng phân loại: điền thiếu bằng mã phổ biến và one-hot; B gộp mã xuất hiện dưới 1% trong dữ liệu fit.
- B thêm nhóm tuổi và chênh lệch giữa điểm tuyển sinh và điểm văn bằng trước.
- Mỗi bước được fit trong từng fold của train. Vì B thay nhiều bước, không thể quy riêng chênh lệch cho một thay đổi.''')
        st.write('Đặc trưng số:', ', '.join(NUMERIC))
        st.write('Đặc trưng phân loại:', ', '.join(CATEGORICAL))
    with results_tab:
        st.subheader('Bốn tổ hợp và lựa chọn trên train')
        st.write('Mỗi luồng kết hợp K-NN và cây quyết định. Dùng cùng 5 fold phân tầng, chọn tham số theo macro-F1 trung bình; recall Dropout được lưu để tham khảo.')
        selection = json.loads((OUT/'selected.json').read_text())
        rows = [{'Tổ hợp': MODEL_LABELS[name], 'Tham số': str(item['params']), 'Macro-F1 CV': item['cv_macro_f1'], 'Recall Dropout CV': item['cv_dropout_recall']} for name,item in selection.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.image(str(ROOT/'report/course-template/images/results/cv_curves.png'), caption='Mỗi điểm là macro-F1 CV cao nhất theo k hoặc độ sâu; các tham số khác được tối ưu trong lưới.')
    with results_tab:
        st.subheader('Phân tích lỗi trên test')
        chosen = st.selectbox('Tổ hợp cần xem', list(MODEL_LABELS), format_func=MODEL_LABELS.get, key='error_model')
        st.image(str(IMAGES/f'{chosen}_confusion_matrix.png'), caption='Hàng: nhãn thật · Cột: nhãn dự đoán')
        pred = pd.read_csv(OUT/f'{chosen}_predictions.csv')
        errors = pred[pred.actual != pred.predicted]
        pairs = errors.groupby(['actual','predicted']).size().rename('Số mẫu').reset_index()
        st.dataframe(pairs, hide_index=True, use_container_width=True)
        actual = st.selectbox('Lớp thật', LABELS, key='error_actual')
        examples = errors[errors.actual == actual].head(10).rename(columns={'row_index':'Chỉ số dòng', 'actual':'Thực tế', 'predicted':'Dự đoán'})
        st.dataframe(examples, hide_index=True, use_container_width=True)
        st.caption('Chỉ số dòng dùng để tra kết quả thí nghiệm, không phải mã sinh viên; không suy đoán nguyên nhân cá nhân từ dự đoán.')
    with results_tab:
        st.subheader('So sánh F1 giữa các mô hình')
        st.image(str(ROOT/'report/course-template/images/results/f1_algorithms.png'), caption='Mô hình được khoanh tròn có macro-F1 CV cao nhất; đường nối chỉ giúp so sánh các tổ hợp và lớp rời rạc.')
        st.subheader('Kết quả trên cùng 885 mẫu test')
        st.caption('Tập test đã được xem trong các lần phát triển; cần tập test độc lập mới để đánh giá cuối cùng.')
        metrics = pd.read_csv(OUT/'summary.csv').copy()
        metrics['model'] = metrics['model'].map(MODEL_LABELS)
        metrics = metrics.rename(columns={'model':'Tổ hợp','accuracy':'Accuracy','macro_precision':'Precision macro',
                    'macro_recall':'Recall macro','macro_f1':'F1 macro','dropout_recall':'Recall Dropout'})
        st.dataframe(metrics, hide_index=True, use_container_width=True)
        st.caption('Các điểm là số từ 0 đến 1; ví dụ 0,462 tương ứng 46,2%.')
        st.image(str(ROOT/'report/course-template/images/results/cv_curves.png'), caption='Chọn siêu tham số bằng CV; không phải điểm test.')
        st.image(str(ROOT/'report/course-template/images/results/cv_test_comparison.png'), caption='So sánh CV và test của từng cấu hình đã chọn.')
        st.image(str(ROOT/'report/course-template/images/results/comparison.png'), caption='Macro-F1 và recall Dropout trên test.')
    with data_tab:
        st.subheader('Kết luận và hướng dẫn')
        st.markdown('''**Ba lớp:** Dropout = bỏ học; Enrolled = còn theo học; Graduate = tốt nghiệp.

**Chọn A/B:** Hai luồng dùng cùng dữ liệu đầu vào; A dùng one-hot thông thường; B gộp mã hiếm và thêm nhóm tuổi cùng chênh lệch điểm.

**Các mã UCI:** Dữ liệu dùng mã số cho ngành, hình thức đăng ký, trình độ và nghề nghiệp; xem tài liệu UCI trước khi nhập. Cần chọn đủ thông tin trong biểu mẫu trước khi dự đoán.

**Giới hạn:** Tập dữ liệu đến từ một cơ sở giáo dục; một số trường chưa có thời điểm ghi nhận rõ. Tập test đã được xem trong các lần phát triển; kết quả chưa đủ tin cậy để áp dụng thực tế.''')


if __name__ == '__main__':
    main()
