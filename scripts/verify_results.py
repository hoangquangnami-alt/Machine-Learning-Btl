"""Verify stored splits, folds, predictions and metrics from a default run."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RESULTS = Path('data/experiment')
DATA = Path('data/raw/data.csv')
LABELS = ['Dropout', 'Enrolled', 'Graduate']
NAMES = ('A_knn', 'A_tree', 'B_knn', 'B_tree')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    df = pd.read_csv(DATA, sep=';')
    train_list = pd.read_csv(RESULTS / 'train_indices.csv').row_index.tolist()
    test = pd.read_csv(RESULTS / 'test_indices.csv').row_index.tolist()
    train = set(train_list)
    check(len(train_list) == len(train) == 3539, 'Train rows are missing or duplicated')
    check(len(test) == len(set(test)) == 885, 'Test rows are missing or duplicated')
    check(not train.intersection(test), 'Train/test overlap')
    check(train | set(test) == set(df.index), 'Split does not cover the input dataset')
    folds = json.loads((RESULTS / 'cv_folds.json').read_text())
    check(len(folds) == 5, 'Expected five CV folds')
    seen = []
    for i, fold in enumerate(folds):
        tr_list, val_list = fold['train_rows'], fold['validation_rows']
        tr, val = set(tr_list), set(val_list)
        check(len(tr_list) == len(tr) and len(val_list) == len(val), f'Fold {i} contains duplicate rows')
        check(not tr & val and tr | val == train, f'Fold {i} is not a partition of train')
        check(not (tr | val) & set(test), f'Fold {i} contains held-out rows')
        seen.extend(val_list)
    check(len(seen) == len(train) and set(seen) == train, 'Validation folds do not cover train exactly once')
    combined_metrics = json.loads((RESULTS / 'metrics.json').read_text())
    check(set(combined_metrics) == set(NAMES), 'Unexpected combined metric models')
    summary = pd.read_csv(RESULTS / 'summary.csv').set_index('model')
    check(set(summary.index) == set(NAMES) and summary.index.is_unique, 'Unexpected summary models')
    for name in NAMES:
        pred = pd.read_csv(RESULTS / f'{name}_predictions.csv')
        metrics = json.loads((RESULTS / f'{name}_metrics.json').read_text())
        check(pred.row_index.tolist() == test, f'{name}: different held-out rows or ordering')
        check(pred.actual.tolist() == df.loc[test, 'Target'].tolist(), f'{name}: actual labels do not match data')
        check(set(pred.predicted).issubset(LABELS), f'{name}: unexpected predicted class')
        check(metrics == combined_metrics[name], f'{name}: combined metrics differ')
        check(metrics['labels'] == LABELS, f'{name}: unexpected label ordering')
        cm = confusion_matrix(pred.actual, pred.predicted, labels=LABELS)
        cm_csv = pd.read_csv(RESULTS / f'{name}_confusion_matrix.csv', index_col=0)
        check(cm_csv.index.tolist() == LABELS and cm_csv.columns.tolist() == LABELS, f'{name}: confusion CSV labels')
        check(np.array_equal(cm, metrics['confusion_matrix']) and np.array_equal(cm, cm_csv.to_numpy()), f'{name}: confusion matrix mismatch')
        accuracy = accuracy_score(pred.actual, pred.predicted)
        report = classification_report(pred.actual, pred.predicted, labels=LABELS, output_dict=True, zero_division=0)
        report_csv = pd.read_csv(RESULTS / f'{name}_classification_report.csv', index_col=0)
        check(all(np.isclose(accuracy, v) for v in (metrics['accuracy'], metrics['classification_report']['accuracy'], report_csv.loc['accuracy', 'precision'], summary.loc[name, 'accuracy'])), f'{name}: accuracy mismatch')
        for label in LABELS + ['macro avg', 'weighted avg']:
            for metric in ('precision', 'recall', 'f1-score', 'support'):
                value = report[label][metric]
                check(np.isclose(value, metrics['classification_report'][label][metric]) and np.isclose(value, report_csv.loc[label, metric]), f'{name}: {label} {metric} mismatch')
        for column, metric in [('macro_precision', 'precision'), ('macro_recall', 'recall'), ('macro_f1', 'f1-score')]:
            check(np.isclose(summary.loc[name, column], report['macro avg'][metric]), f'{name}: summary {column} mismatch')
        check(np.isclose(summary.loc[name, 'dropout_recall'], report['Dropout']['recall']), f'{name}: dropout recall mismatch')
        cv = pd.read_csv(RESULTS / f'{name}_cv.csv')
        check(len(cv) >= 5 and all(f'split{i}_test_macro_f1' in cv and f'split{i}_test_dropout_recall' in cv for i in range(5)), f'{name}: incomplete CV results')
    print('PASS: four models, identical 885 test rows, train-only folds, predictions, confusion matrices and metrics')


if __name__ == '__main__':
    main()
