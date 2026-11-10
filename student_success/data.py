"""Schema validation, audit and deterministic stratified split."""
from pathlib import Path
import hashlib
import json
import pandas as pd
from sklearn.model_selection import train_test_split

CATEGORICAL = ['Marital status', 'Application mode', 'Course', 'Daytime/evening attendance',
    'Previous qualification', 'Nacionality', "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", 'Displaced', 'Educational special needs',
    'Gender', 'International']
NUMERIC = ['Application order', 'Previous qualification (grade)', 'Admission grade', 'Age at enrollment']
SEMESTER = [f'Curricular units {s} sem ({x})' for s in ('1st', '2nd') for x in
            ('credited', 'enrolled', 'evaluations', 'approved', 'grade', 'without evaluations')]
EXCLUDED = ['Debtor', 'Tuition fees up to date', 'Scholarship holder'] + SEMESTER + ['Unemployment rate', 'Inflation rate', 'GDP']
COLUMNS = CATEGORICAL + NUMERIC + EXCLUDED + ['Target']
LABELS = ['Dropout', 'Enrolled', 'Graduate']
SEED = 42


def load_data(path, output):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Data not found: {path}. Run python3 scripts/download_data.py or use --data.')
    df = pd.read_csv(path, sep=';')
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any() or set(df.columns) != set(COLUMNS):
        raise ValueError(f'Unexpected schema; missing={sorted(set(COLUMNS)-set(df.columns))}, extra={sorted(set(df.columns)-set(COLUMNS))}')
    if set(df.Target.dropna()) != set(LABELS) or df.Target.isna().any():
        raise ValueError(f'Unexpected or missing target labels: {df.Target.value_counts(dropna=False).to_dict()}')
    for c in COLUMNS[:-1]:
        df[c] = pd.to_numeric(df[c], errors='raise')
    audit = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'shape': list(df.shape),
             'columns': list(df.columns), 'dtypes': df.dtypes.astype(str).to_dict(),
             'class_counts': df.Target.value_counts().to_dict(), 'missing': df.isna().sum().to_dict(),
             'duplicate_rows': int(df.duplicated().sum())}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return df


def split(df):
    train, test = train_test_split(df.index, test_size=0.2, random_state=SEED, stratify=df.Target)
    return train, test
