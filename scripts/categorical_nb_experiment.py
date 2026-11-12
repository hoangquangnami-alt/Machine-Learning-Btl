"""Run the separate, exploratory CategoricalNB experiment on saved splits."""
import json
import platform
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from student_success.categorical_nb import build_pipeline
from student_success.data import CATEGORICAL, NUMERIC, SEED, load_data
from student_success.evaluate import evaluate
from student_success.preprocess import build_preprocessor

R = ROOT / 'data/experiment'
OUT = R / 'categorical_nb'
IMG = ROOT / 'report/course-template/images/results'
NAMES = ('A_knn', 'A_tree', 'B_knn', 'B_tree')
SCORING = {'macro_f1': 'f1_macro', **{f'dropout_{name}': make_scorer(fn, labels=['Dropout'], average='macro', zero_division=0) for name, fn in [('precision', precision_score), ('recall', recall_score), ('f1', f1_score)]}}
GRID = {'preprocess__numeric__bins__n_bins': [3, 5, 8], 'preprocess__numeric__bins__strategy': ['quantile', 'uniform'], 'model__alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data(ROOT / 'data/raw/data.csv', OUT)
    train = pd.read_csv(R/'train_indices.csv').row_index.to_numpy()
    test = pd.read_csv(R/'test_indices.csv').row_index.to_numpy()
    folds = json.loads((R/'cv_folds.json').read_text())
    assert len(set(train)) == len(train) and len(set(test)) == len(test)
    assert not set(train) & set(test) and set(train) | set(test) == set(df.index)
    positions = {row: i for i, row in enumerate(train)}
    cv = []
    seen = []
    for fold in folds:
        tr, val = fold['train_rows'], fold['validation_rows']
        assert len(set(tr)) == len(tr) and len(set(val)) == len(val)
        assert not set(tr) & set(val) and set(tr) | set(val) == set(train)
        seen.extend(val)
        cv.append((np.array([positions[i] for i in tr]), np.array([positions[i] for i in val])))
    assert len(cv) == 5 and len(seen) == len(train) and set(seen) == set(train)
    X, y = df[CATEGORICAL + NUMERIC], df.Target
    Xtr, ytr = X.loc[train], y.loc[train]
    search = GridSearchCV(build_pipeline(), GRID, scoring=SCORING, refit='macro_f1', cv=cv,
                          n_jobs=1, error_score='raise', return_train_score=False)
    search.fit(Xtr, ytr)
    pd.DataFrame(search.cv_results_).to_csv(OUT/'search_cv.csv', index=False)
    selected = json.loads((R/'selected.json').read_text())
    rows, fold_rows = [], []
    for name in (*NAMES, 'CategoricalNB'):
        if name == 'CategoricalNB':
            result = {f'test_{metric}': np.array([search.cv_results_[f'split{i}_test_{metric}'][search.best_index_] for i in range(5)]) for metric in SCORING}
        else:
            flow, algo = name.split('_')
            est = KNeighborsClassifier() if algo == 'knn' else DecisionTreeClassifier(random_state=SEED)
            pipe = Pipeline([('preprocess', build_preprocessor(flow)), ('model', est)]).set_params(**selected[name]['params'])
            result = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=SCORING, n_jobs=1, error_score='raise')
            saved = pd.read_csv(R/f'{name}_cv.csv')
            saved_row = saved.loc[saved.rank_test_macro_f1 == 1].iloc[0]
            for i in range(5):
                if not np.isclose(result['test_macro_f1'][i], saved_row[f'split{i}_test_macro_f1'], atol=1e-12):
                    raise ValueError(f'{name} fold {i} does not reproduce saved CV')
        row = {'model': name}
        for metric in SCORING:
            vals = result[f'test_{metric}']
            row[f'mean_{metric}'] = float(np.mean(vals))
            if metric == 'macro_f1': row['std_macro_f1'] = float(np.std(vals))
        rows.append(row)
        for i in range(5): fold_rows.append({'model': name, 'fold': i, **{metric: float(result[f'test_{metric}'][i]) for metric in SCORING}})
        print(name, row, flush=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT/'comparison_cv.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(OUT/'fold_metrics.csv', index=False)
    chosen = {'params': search.best_params_, 'cv_macro_f1': search.best_score_, 'best_index': int(search.best_index_), 'fit_prior': True}
    (OUT/'selected.json').write_text(json.dumps(chosen, indent=2)+'\n')
    evaluate('CategoricalNB', search.best_estimator_, X.loc[test], y.loc[test], OUT, IMG)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(comparison.model, comparison.mean_macro_f1, yerr=comparison.std_macro_f1, capsize=4)
    ax.set_ylabel('Mean training CV macro-F1 (± fold SD)')
    ax.set_ylim(0, .65); ax.tick_params(axis='x', rotation=20)
    fig.tight_layout(); fig.savefig(IMG/'categorical_nb_cv_comparison.png', dpi=160); plt.close(fig)
    log = {'python': platform.python_version(), 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__, 'matplotlib': matplotlib.__version__, 'seed': SEED, 'train_size': len(train), 'test_size': len(test), 'folds': 5, 'grid': GRID, 'selected': chosen, 'test_status': 'exploratory; development-exposed', 'command': 'python scripts/categorical_nb_experiment.py', 'bin_options': {'encode': 'ordinal', 'quantile_method': 'averaged_inverted_cdf', 'subsample': None, 'random_state': SEED}, 'baseline_cv': 'selected configurations refit on saved folds; fold macro-F1 checked against saved results'}
    (OUT/'run_log.json').write_text(json.dumps(log, indent=2)+'\n')

if __name__ == '__main__': main()
