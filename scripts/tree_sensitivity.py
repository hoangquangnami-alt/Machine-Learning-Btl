"""Summarize stored tree CV results and refit selected configurations across seeds."""
import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from student_success.data import CATEGORICAL, NUMERIC
from student_success.preprocess import build_preprocessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'data/experiment')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/experiment')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f'Python {platform.python_version()}, {platform.platform()}, {platform.machine()}, numpy {np.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}')
    selected = json.loads((args.results / 'selected.json').read_text())
    df = pd.read_csv(ROOT / 'data/raw/data.csv', sep=';')
    df.columns = df.columns.str.strip()
    train = pd.read_csv(args.results / 'train_indices.csv').row_index
    test = pd.read_csv(args.results / 'test_indices.csv').row_index
    x, y = df[CATEGORICAL + NUMERIC], df.Target
    top, runs = [], []
    for flow in ('A', 'B'):
        cv = pd.read_csv(args.results / f'{flow}_tree_cv.csv').sort_values('mean_test_macro_f1', ascending=False).head(5).copy()
        cv['flow'] = flow
        cv['gap_to_best'] = cv['mean_test_macro_f1'].iloc[0] - cv['mean_test_macro_f1']
        top.append(cv[['flow', 'params', 'mean_test_macro_f1', 'std_test_macro_f1', 'gap_to_best']])
        params = {k.removeprefix('model__'): v for k, v in selected[f'{flow}_tree']['params'].items()}
        for seed in range(10):
            pipe = Pipeline([('preprocess', build_preprocessor(flow)), ('model', DecisionTreeClassifier(random_state=seed, **params))])
            pipe.fit(x.loc[train], y.loc[train])
            pred = pipe.predict(x.loc[test]); truth = y.loc[test]
            runs.append({'flow': flow, 'random_state': seed, 'accuracy': accuracy_score(truth, pred), 'macro_f1': f1_score(truth, pred, average='macro'), 'dropout_recall': recall_score(truth, pred, labels=['Dropout'], average='macro', zero_division=0), 'enrolled_recall': recall_score(truth, pred, labels=['Enrolled'], average='macro', zero_division=0)})
    pd.concat(top, ignore_index=True).to_csv(args.output / 'tree_sensitivity_top5.csv', index=False)
    frame = pd.DataFrame(runs)
    frame.to_csv(args.output / 'tree_sensitivity_runs.csv', index=False)
    frame.groupby('flow').agg(['mean', 'std']).drop(columns='random_state').to_csv(args.output / 'tree_sensitivity_summary.csv')
    print(frame.groupby('flow').agg(['mean', 'std']).to_string())


if __name__ == '__main__':
    main()
