"""Evaluate selected models on the same held-out rows; store metrics and plots."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from .data import LABELS


def evaluate(name, model, X, y, output, images):
    pred = model.predict(X)
    report = classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=LABELS)
    metrics = {'accuracy': accuracy_score(y, pred), 'classification_report': report,
               'confusion_matrix': cm.tolist(), 'labels': LABELS}
    (output / f'{name}_metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    pd.DataFrame(report).T.to_csv(output / f'{name}_classification_report.csv')
    pd.DataFrame(cm, index=LABELS, columns=LABELS).rename_axis('actual').to_csv(output / f'{name}_confusion_matrix.csv')
    pred_df = pd.DataFrame({'row_index': X.index, 'actual': y.to_numpy(), 'predicted': pred})
    pred_df.to_csv(output / f'{name}_predictions.csv', index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name.replace('_', ' ')); fig.tight_layout(); fig.savefig(images / f'{name}_confusion_matrix.png', dpi=150); plt.close(fig)
    print(name, 'accuracy', round(metrics['accuracy'], 4), 'macro F1', round(report['macro avg']['f1-score'], 4),
          'dropout recall', round(report['Dropout']['recall'], 4))
    return metrics
