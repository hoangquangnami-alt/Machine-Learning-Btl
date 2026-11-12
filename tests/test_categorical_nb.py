"""Focused checks for the exploratory CategoricalNB pipeline and artifacts."""
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from student_success.categorical_nb import build_pipeline
from student_success.data import CATEGORICAL, NUMERIC, LABELS

ROOT = Path(__file__).resolve().parent.parent

class CategoricalNBChecks(unittest.TestCase):
    def test_unknown_missing_constant_clone_and_cv(self):
        n = 30
        frame = pd.DataFrame({c: np.array([1., 2., 1.] * 10) for c in CATEGORICAL + NUMERIC})
        frame[NUMERIC[0]] = 7.0
        frame.loc[0, CATEGORICAL[1]] = np.nan
        frame.loc[1, NUMERIC[1]] = np.nan
        y = np.array(LABELS * 10)
        pipe = clone(build_pipeline())
        pipe.fit(frame, y)
        incoming = frame.iloc[:2].copy()
        incoming.loc[incoming.index[0], CATEGORICAL[0]] = 99999
        incoming.loc[incoming.index[0], CATEGORICAL[1]] = np.nan
        incoming.loc[incoming.index[0], NUMERIC[1]] = np.nan
        encoded = pipe.named_steps['preprocess'].transform(incoming)
        self.assertEqual(encoded[0, 0], 0)
        self.assertTrue(np.all(encoded >= 0) and np.all(encoded == np.floor(encoded)))
        self.assertGreaterEqual(pipe.named_steps['model'].model_.n_categories_[0], 2)
        self.assertEqual(len(pipe.predict(incoming)), 2)
        scores = cross_val_score(clone(pipe), frame, y, cv=StratifiedKFold(3, shuffle=True, random_state=42), scoring='f1_macro', error_score='raise')
        self.assertTrue(np.isfinite(scores).all())

    def test_saved_predictions_and_separation(self):
        r = ROOT/'data/experiment'; out = r/'categorical_nb'
        df = pd.read_csv(ROOT/'data/raw/data.csv', sep=';')
        train = pd.read_csv(r/'train_indices.csv').row_index.tolist()
        test = pd.read_csv(r/'test_indices.csv').row_index.tolist()
        folds = json.loads((r/'cv_folds.json').read_text())
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(train) | set(test), set(df.index))
        seen = []
        for fold in folds:
            a, b = set(fold['train_rows']), set(fold['validation_rows'])
            self.assertFalse(a & b); self.assertEqual(a | b, set(train)); self.assertFalse((a | b) & set(test))
            seen.extend(fold['validation_rows'])
        self.assertEqual(len(seen), len(set(seen))); self.assertEqual(set(seen), set(train))
        p = pd.read_csv(out/'CategoricalNB_predictions.csv')
        m = json.loads((out/'CategoricalNB_metrics.json').read_text())
        self.assertEqual(p.row_index.tolist(), test)
        self.assertEqual(p.actual.tolist(), df.loc[test, 'Target'].tolist())
        self.assertTrue(np.isclose(m['accuracy'], accuracy_score(p.actual, p.predicted)))
        self.assertEqual(m['confusion_matrix'], confusion_matrix(p.actual, p.predicted, labels=LABELS).tolist())
        report = classification_report(p.actual, p.predicted, labels=LABELS, output_dict=True, zero_division=0)
        for label in LABELS + ['macro avg', 'weighted avg']:
            for metric in ('precision', 'recall', 'f1-score', 'support'):
                self.assertAlmostEqual(m['classification_report'][label][metric], report[label][metric])

if __name__ == '__main__': unittest.main()
