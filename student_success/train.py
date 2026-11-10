"""Run the controlled 2 x 2 experiment."""
import argparse
import json
import platform
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from .data import CATEGORICAL, NUMERIC, LABELS, SEED, load_data, split
from .preprocess import build_preprocessor
from .evaluate import evaluate


def eda(df, train_idx, output, images):
    train = df.loc[train_idx]
    train[NUMERIC].describe().to_csv(output / 'eda_numeric_summary.csv')
    train[CATEGORICAL].apply(lambda s: s.value_counts()).to_csv(output / 'eda_categorical_counts.csv')
    train[NUMERIC].corr().to_csv(output / 'eda_numeric_correlations.csv')
    outliers = {}
    for c in NUMERIC:
        q1, q3 = train[c].quantile([.25, .75]); iqr = q3-q1
        outliers[c] = int(((train[c] < q1-1.5*iqr) | (train[c] > q3+1.5*iqr)).sum())
    (output / 'eda_outliers_iqr.json').write_text(json.dumps(outliers, indent=2)+'\n')
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    for ax, c in zip(axs.flat, NUMERIC):
        ax.hist(train[c].dropna(), bins=25); ax.set_title(c)
    fig.tight_layout(); fig.savefig(images / 'eda_histograms.png', dpi=140); plt.close(fig)
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    for ax, c in zip(axs, NUMERIC):
        ax.boxplot(train[c].dropna()); ax.set_title(c, fontsize=8)
    fig.tight_layout(); fig.savefig(images / 'eda_boxplots.png', dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); train.Target.value_counts().reindex(LABELS).plot.bar(ax=ax)
    ax.set_title('Training labels'); ax.set_ylabel('Count'); fig.tight_layout(); fig.savefig(images / 'eda_class_distribution.png', dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); train['Course'].value_counts().plot.bar(ax=ax)
    ax.set_title('Training course codes'); ax.set_ylabel('Count'); fig.tight_layout(); fig.savefig(images / 'eda_courses.png', dpi=140); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/raw/data.csv'); parser.add_argument('--output', default='results', help='Thư mục ảnh'); parser.add_argument('--tables-output', help='Thư mục CSV/JSON (mặc định data/experiment; nếu --output khác mặc định: <output>/data)')
    args = parser.parse_args(); images = Path(args.output); images.mkdir(parents=True, exist_ok=True)
    out = Path(args.tables_output) if args.tables_output else (Path('data/experiment') if images == Path('results') else images / 'data')
    out.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data, out); train_idx, test_idx = split(df)
    pd.DataFrame({'row_index': train_idx}).to_csv(out/'train_indices.csv', index=False)
    pd.DataFrame({'row_index': test_idx}).to_csv(out/'test_indices.csv', index=False)
    eda(df, train_idx, out, images)
    X = df[CATEGORICAL+NUMERIC]; y = df.Target
    cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X.loc[train_idx], y.loc[train_idx]))
    folds = [{'train_rows': train_idx[tr].tolist(), 'validation_rows': train_idx[val].tolist()} for tr, val in cv]
    (out/'cv_folds.json').write_text(json.dumps(folds)+'\n')
    scorer = make_scorer(recall_score, labels=['Dropout'], average='macro', zero_division=0)
    configs = {'knn': (KNeighborsClassifier(), {'model__n_neighbors': [5, 11, 21], 'model__weights': ['uniform', 'distance'], 'model__p': [1, 2]}),
               'tree': (DecisionTreeClassifier(random_state=SEED), {'model__max_depth': [3, 8, 12], 'model__min_samples_leaf': [1, 10], 'model__class_weight': [None, 'balanced']})}
    all_metrics = {}; selected = {}
    for flow in ('A', 'B'):
        for algorithm, (estimator, grid) in configs.items():
            name = f'{flow}_{algorithm}'
            pipe = Pipeline([('preprocess', build_preprocessor(flow)), ('model', estimator)])
            search = GridSearchCV(pipe, grid, cv=cv, scoring={'macro_f1':'f1_macro', 'dropout_recall':scorer}, refit='macro_f1', n_jobs=1, return_train_score=False)
            search.fit(X.loc[train_idx], y.loc[train_idx])
            pd.DataFrame(search.cv_results_).to_csv(out/f'{name}_cv.csv', index=False)
            selected[name] = {'params': search.best_params_, 'cv_macro_f1': search.best_score_, 'cv_dropout_recall': float(search.cv_results_['mean_test_dropout_recall'][search.best_index_])}
            all_metrics[name] = evaluate(name, search.best_estimator_, X.loc[test_idx], y.loc[test_idx], out, images)
    (out/'selected.json').write_text(json.dumps(selected, indent=2)+'\n')
    (out/'metrics.json').write_text(json.dumps(all_metrics, indent=2)+'\n')
    summary = pd.DataFrame([{'model': n, 'accuracy': m['accuracy'], 'macro_precision': m['classification_report']['macro avg']['precision'], 'macro_recall': m['classification_report']['macro avg']['recall'], 'macro_f1': m['classification_report']['macro avg']['f1-score'], 'dropout_recall': m['classification_report']['Dropout']['recall']} for n, m in all_metrics.items()])
    summary.to_csv(out/'summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 4)); summary.set_index('model')[['macro_f1','dropout_recall']].plot.bar(ax=ax, ylim=(0,1)); ax.set_ylabel('Score'); fig.tight_layout(); fig.savefig(images/'comparison.png', dpi=150); plt.close(fig)
    log = {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__, 'scikit_learn': sklearn.__version__, 'seed': SEED, 'train_size': len(train_idx), 'test_size': len(test_idx), 'cv_folds': 5, 'grid': {k: v[1] for k, v in configs.items()}, 'selected': selected}
    (out/'experiment_log.json').write_text(json.dumps(log, indent=2)+'\n')

if __name__ == '__main__':
    main()
