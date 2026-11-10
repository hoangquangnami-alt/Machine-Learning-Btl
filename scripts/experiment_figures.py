"""Biểu đồ thí nghiệm từ CSV/JSON đã lưu; không huấn luyện hay chọn lại mô hình."""
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
R = Path(os.environ.get('FIT_RESULTS', ROOT / 'data/experiment'))
OUT = Path(os.environ.get('FIT_IMAGES', ROOT / 'report/course-template/images/results'))
OUT.mkdir(parents=True, exist_ok=True)
selected = json.loads((R / 'selected.json').read_text())
summary = pd.read_csv(R / 'summary.csv').set_index('model')
BLUE, ORANGE, DARK, PURPLE = '#315ea8', '#cb6530', '#4a4a4a', '#6352c7'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 13,
                     'axes.titleweight': 'bold', 'axes.labelcolor': '#333333',
                     'xtick.color': '#555555', 'ytick.color': '#555555', 'figure.facecolor': 'white'})
fmt = FuncFormatter(lambda v, pos: f'{v:.2f}'.replace('.', ','))
def clean(ax, axis='y'):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color='#dedede', linewidth=.8)
    for s in ('top','right','left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')

def val(v): return f'{v:.3f}'.replace('.', ',')

# Mỗi điểm lấy macro-F1 CV cao nhất trong các cấu hình có cùng k hoặc độ sâu.
fig, axes = plt.subplots(2, 1, figsize=(10, 8.3), layout='constrained')
for ax, model, ticks, param, xlabel in [
    (axes[0], 'knn', [5,11,21], 'param_model__n_neighbors', 'Số láng giềng k'),
    (axes[1], 'tree', [3,8,12], 'param_model__max_depth', 'Độ sâu tối đa')]:
    ax.set_title('K-NN: thay số láng giềng' if model == 'knn' else 'Cây quyết định: thay độ sâu', loc='left', pad=12)
    for flow, color, marker, ls in [('A', BLUE, 'o', '-'), ('B', ORANGE, 's', '--')]:
        cv = pd.read_csv(R / f'{flow}_{model}_cv.csv')
        y = cv.groupby(param)['mean_test_macro_f1'].max().reindex(ticks).to_numpy()
        ax.plot(range(len(ticks)), y, color=color, marker=marker, linestyle=ls, linewidth=2.2, markersize=6,
                label=f'Luồng {flow}')
        idx = int(np.argmax(y))
        ax.annotate(f'{flow}: {val(y[idx])}', (idx,y[idx]), xytext=(0, 16 if flow=='A' else -24), textcoords='offset points', ha='center', color=color, fontsize=10, fontweight='bold', arrowprops=dict(arrowstyle='-',color=color,linewidth=.8))
    ax.set(xticks=range(len(ticks)), xticklabels=ticks, xlabel=xlabel, ylabel='Macro-F1 trung bình · 5 fold', ylim=(.34,.50), xlim=(-.15,len(ticks)-.85))
    ax.yaxis.set_major_formatter(fmt); clean(ax); ax.legend(loc='lower left',frameon=False,ncol=2,fontsize=9)
fig.suptitle('CV: điểm tốt nhất ở mỗi giá trị k hoặc độ sâu', x=.07, ha='left', fontsize=16, fontweight='bold')
fig.savefig(OUT/'cv_curves.png', dpi=220); plt.close(fig)

# So sánh CV/test của đúng cấu hình đã chọn; đây không phải điểm trên train fit.
models = ['A_knn','B_knn','A_tree','B_tree']
names = [f"{model[0]} · K-NN (k = {selected[model]['params']['model__n_neighbors']})" if model.endswith('knn') else f"{model[0]} · Cây (độ sâu = {selected[model]['params']['model__max_depth']})" for model in models]
fig, ax = plt.subplots(figsize=(10,4.8), layout='constrained')
ypos = np.arange(len(models))
for i, name in enumerate(models):
    cv = selected[name]['cv_macro_f1']; test = summary.loc[name,'macro_f1']
    ax.plot([cv,test], [i,i], color='#d8d8d8', linewidth=4, zorder=1)
    ax.scatter(cv,i,s=100,color=DARK,edgecolors='white',linewidth=1,zorder=3, label='CV trung bình trên train' if i==0 else None)
    ax.scatter(test,i,s=100,color=PURPLE,edgecolors='white',linewidth=1,zorder=3, label='Test (885 mẫu)' if i==0 else None)
    ax.text(max(cv,test)+.006,i,f'{val(cv)} / {val(test)}',va='center',fontsize=11,color='#444444')
ax.set(yticks=ypos, yticklabels=names, xlim=(.37,.525), xlabel='Macro-F1  ·  nhãn số: CV / test')
ax.invert_yaxis(); ax.xaxis.set_major_formatter(fmt); clean(ax,'x')
ax.legend(loc='upper left', bbox_to_anchor=(0, -.17), frameon=False, ncol=2, fontsize=10)
ax.set_title('Macro-F1: so sánh CV và test của từng tổ hợp', loc='left', pad=15, fontsize=15)
fig.savefig(OUT/'cv_test_comparison.png', dpi=220); plt.close(fig)

# Hai độ đo có ý nghĩa khác nhau, hiển thị theo từng tổ hợp trên cùng thang 0–1.
fig, ax = plt.subplots(figsize=(10,4.8), layout='constrained')
for i, name in enumerate(models):
    a = summary.loc[name,'macro_f1']; b = summary.loc[name,'dropout_recall']
    ax.plot([a,b], [i,i], color='#dddddd', linewidth=4, zorder=1)
    ax.scatter(a,i,s=100,color=BLUE,edgecolors='white',linewidth=1,zorder=3, label='Macro-F1: trung bình ba lớp' if i==0 else None)
    ax.scatter(b,i,s=100,color=ORANGE,edgecolors='white',linewidth=1,zorder=3, label='Recall: riêng lớp Dropout' if i==0 else None)
    ax.text(max(a,b)+.015,i,f'{val(a)} / {val(b)}',va='center',fontsize=11,color='#444444')
ax.set(yticks=ypos, yticklabels=names, xlim=(0,1), xlabel='Điểm trên test  ·  nhãn số: macro-F1 / recall Dropout')
ax.invert_yaxis(); ax.xaxis.set_major_formatter(fmt); clean(ax,'x')
ax.legend(loc='upper left', bbox_to_anchor=(0, -.17), frameon=False, ncol=2, fontsize=10)
ax.set_title('Điểm tổng thể và khả năng phát hiện bỏ học', loc='left', pad=15, fontsize=15)
fig.savefig(OUT/'comparison.png', dpi=220); plt.close(fig)
print('Đã tạo biểu đồ CV, CV/test và kết quả test từ kết quả đã lưu.')

# Điểm theo tổ hợp và theo lớp: trục x là hạng mục rời rạc, không phải thời gian.
model_order = ['A_knn', 'A_tree', 'B_knn', 'B_tree']
model_labels = ['A / K-NN', 'A / Cây', 'B / K-NN', 'B / Cây']
class_labels = ['Dropout\nBỏ học', 'Enrolled\nCòn học', 'Graduate\nTốt nghiệp']
class_codes = ['Dropout', 'Enrolled', 'Graduate']
colors = {'A_knn':'#4472b2', 'A_tree':'#dc8433', 'B_knn':'#6551c8', 'B_tree':'#278276'}
fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 9.8), layout='constrained')
fig.suptitle('So sánh F1 giữa các tổ hợp mô hình', x=.08, ha='left', fontsize=16, fontweight='bold')
x = np.arange(len(model_order))
cv_points = [selected[name]['cv_macro_f1'] for name in model_order]
test_points = [summary.loc[name, 'macro_f1'] for name in model_order]
for vals, color, label, linestyle, offset in [(cv_points, '#3c57aa', 'CV trên train · dùng chọn mô hình', '-', 12),
                                              (test_points, '#cb6530', 'Test đã xem · chỉ đối chiếu', '--', -18)]:
    top.plot(x, vals, color=color, marker='o', markersize=7, linewidth=2.2, linestyle=linestyle, label=label)
    for i, score in enumerate(vals):
        top.annotate(val(score), (i, score), xytext=(0, offset), textcoords='offset points', ha='center', fontsize=10, color=color, fontweight='bold')
best = max(model_order, key=lambda name: selected[name]['cv_macro_f1'])
idx = model_order.index(best)
top.scatter([idx], [cv_points[idx]], s=170, facecolors='none', edgecolors='#3c57aa', linewidths=1.8, zorder=4)
top.set_title('Macro-F1 tổng thể: chọn theo CV của train', loc='left', pad=10)
top.set(xticks=x, xticklabels=model_labels, ylabel='Macro-F1', ylim=(.33,.53), xlim=(-.3,3.3))
top.yaxis.set_major_formatter(fmt); clean(top); top.legend(loc='lower left', frameon=False, ncol=2, fontsize=10)
bottom.set_title('F1 từng lớp trên test đã xem', loc='left', pad=10)
for name in model_order:
    report = pd.read_csv(R / f'{name}_classification_report.csv', index_col=0)
    values = report.loc[class_codes, 'f1-score'].to_numpy()
    chosen = name == best
    bottom.plot(range(3), values, label=model_labels[model_order.index(name)] + (' · chọn theo CV' if chosen else ''),
                color=colors[name], marker='o' if chosen else 's', linewidth=2.8 if chosen else 1.7,
                linestyle='-' if chosen else '--', markersize=6)
    if chosen:
        for i, score in enumerate(values):
            bottom.annotate(val(score), (i, score), xytext=(10, -18 if i == 1 else 6), textcoords='offset points', fontsize=10,
                            color=colors[name], fontweight='bold')
bottom.set(xticks=range(3), xticklabels=class_labels, ylabel='F1 từng lớp', ylim=(0,.85), xlim=(-.2,2.25))
bottom.yaxis.set_major_formatter(fmt); clean(bottom)
bottom.legend(loc='upper center', bbox_to_anchor=(.5,-.18), ncol=2, frameon=False, fontsize=10)
fig.savefig(OUT/'f1_algorithms.png', dpi=220)
(ROOT/'results').mkdir(exist_ok=True)
if 'FIT_IMAGES' not in os.environ:
    fig.savefig(ROOT/'results/f1_algorithms.png', dpi=220)
plt.close(fig)
print('Đã tạo biểu đồ đường F1 theo tổ hợp và theo lớp.')
