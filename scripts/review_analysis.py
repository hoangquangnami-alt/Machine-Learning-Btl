"""Describe existing split and predictions without fitting or tuning a model."""
import json
import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

def main():
    root = Path(os.environ.get('FIT_RESULTS', 'data/experiment'))
    df = pd.read_csv('data/raw/data.csv', sep=';')
    df.columns = df.columns.str.strip()
    train = pd.read_csv(root / 'train_indices.csv').row_index
    test = pd.read_csv(root / 'test_indices.csv').row_index
    majority = df.loc[train, 'Target'].value_counts().idxmax()
    actual = df.loc[test, 'Target'].to_numpy()
    pred = [majority] * len(test)
    unknown = {"Mother's qualification": 34, "Father's qualification": 34,
               "Mother's occupation": 99, "Father's occupation": 99}
    report = {
        'purpose': 'Post-hoc descriptive baseline and semantic unknown audit; not model selection',
        'majority_class_from_train': majority,
        'baseline_test': {'accuracy': accuracy_score(actual, pred),
                          'macro_f1': f1_score(actual, pred, average='macro', zero_division=0),
                          'dropout_recall': recall_score(actual, pred, labels=['Dropout'], average='macro', zero_division=0)},
        'semantic_unknown_codes': {name: {'code': value, 'count_train': int((df.loc[train, name] == value).sum()),
                                          'count_all': int((df[name] == value).sum())}
                                   for name, value in unknown.items()},
        'examples_A_tree': {},
    }
    predictions = pd.read_csv(root / 'A_tree_predictions.csv')
    for actual_label, predicted_label in [('Dropout', 'Graduate'), ('Dropout', 'Enrolled'), ('Enrolled', 'Dropout')]:
        selection = predictions[(predictions.actual == actual_label) & (predictions.predicted == predicted_label)]
        report['examples_A_tree'][f'{actual_label} -> {predicted_label}'] = selection.row_index.head(3).tolist()
    (root / 'review_analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
