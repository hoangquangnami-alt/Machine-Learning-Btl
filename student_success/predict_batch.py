"""Kiểm tra dữ liệu đầu vào trước khi dự đoán theo schema UCI."""
import numpy as np
import pandas as pd
from .data import CATEGORICAL, NUMERIC

COLUMNS = CATEGORICAL + NUMERIC
SCORES = ['prediction', 'score_Dropout', 'score_Enrolled', 'score_Graduate']
LIMITS = {'Application order': (0, 9), 'Previous qualification (grade)': (0, 200),
          'Admission grade': (0, 200), 'Age at enrollment': (15, 100)}


def validate_batch(frame, training):
    frame = frame.copy()
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError('Tên cột bị trùng sau khi bỏ khoảng trắng.')
    missing = [c for c in COLUMNS if c not in frame]
    if missing:
        raise ValueError('Thiếu cột bắt buộc: ' + ', '.join(missing))
    if frame.empty:
        raise ValueError('Tệp không có hồ sơ.')
    if len(frame) > 10000:
        raise ValueError('Mỗi tệp tối đa 10.000 hồ sơ.')
    if set(SCORES) & set(frame.columns):
        raise ValueError('Tệp chứa cột kết quả có sẵn; hãy đổi tên hoặc xóa các cột prediction, score_* trước khi chạy.')
    data = frame[COLUMNS].copy()
    for col in COLUMNS:
        try:
            data[col] = pd.to_numeric(data[col], errors='raise')
        except (ValueError, TypeError):
            raise ValueError(f'Cột {col} phải chứa số theo mã UCI.') from None
        invalid = data[col].isna() | ~np.isfinite(data[col])
        if invalid.any():
            raise ValueError(f'Cột {col} có giá trị trống hoặc không hữu hạn (dòng CSV {int(np.flatnonzero(invalid)[0])+2}).')
        if col in CATEGORICAL:
            invalid = ~data[col].isin(training[col].dropna().unique())
            if invalid.any():
                raise ValueError(f'Cột {col} có mã chưa xuất hiện trong train tại dòng CSV {int(np.flatnonzero(invalid)[0])+2}; cần kiểm tra bảng mã UCI.')
        else:
            lo, hi = LIMITS[col]
            invalid = ~data[col].between(lo, hi)
            if col in ('Application order', 'Age at enrollment'):
                invalid |= data[col] % 1 != 0
            if invalid.any():
                raise ValueError(f'Cột {col} ngoài phạm vi {lo}–{hi} hoặc không đúng kiểu tại dòng CSV {int(np.flatnonzero(invalid)[0])+2}.')
    return data


def predict_batch(frame, training, model):
    data = validate_batch(frame, training)
    result = frame.copy()
    result['prediction'] = model.predict(data)
    probabilities = model.predict_proba(data)
    for index, label in enumerate(model.classes_):
        result[f'score_{label}'] = probabilities[:, index]
    return result
