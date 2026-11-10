"""Vẽ lại hình trong báo cáo từ dữ liệu và kết quả đã lưu, không huấn luyện lại."""
from pathlib import Path
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = Path(os.environ.get('FIT_RESULTS', ROOT / 'data/experiment'))
OUTPUT = Path(os.environ.get('FIT_IMAGES', ROOT / 'report/course-template/images/results'))
OUTPUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 12,
                     'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10})
df = pd.read_csv(ROOT / 'data/raw/data.csv', sep=';')
df.columns = df.columns.str.strip()
indices = pd.read_csv(R / 'train_indices.csv').row_index
train = df.loc[indices]
labels = ['Dropout', 'Enrolled', 'Graduate']

fig, ax = plt.subplots(figsize=(6.6, 3.8), layout='constrained')
counts = train.Target.value_counts().reindex(labels)
bars = ax.bar(labels, counts, color=['#2f6da5', '#4b83b4', '#73a6ce'])
ax.bar_label(bars, labels=[f'{x:,}'.replace(',', '.') for x in counts], padding=3)
ax.set_ylabel('Số sinh viên'); ax.set_ylim(0, 2100)
fig.savefig(OUTPUT/'class_distribution.png', dpi=220); plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(8.5, 6), layout='constrained')
for ax, col, title in zip(axes.flat,
    ['Application order','Previous qualification (grade)','Admission grade','Age at enrollment'],
    ['Thứ tự nguyện vọng','Điểm văn bằng trước','Điểm tuyển sinh','Tuổi lúc nhập học']):
    ax.hist(train[col].dropna(), bins=25, color='#397bb0', edgecolor='white', linewidth=.3)
    ax.set_title(title); ax.set_ylabel('Số sinh viên')
fig.savefig(OUTPUT/'histograms.png', dpi=220); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.0, 6.0), layout='constrained')
counts = train['Course'].value_counts().sort_values(ascending=True)
ax.barh([str(x) for x in counts.index], counts.values, color='#397bb0')
ax.set_xlabel('Số sinh viên'); ax.set_ylabel('Mã ngành')
fig.savefig(OUTPUT/'courses.png', dpi=220); plt.close(fig)

cm = pd.read_csv(R/'A_tree_confusion_matrix.csv', index_col=0).loc[labels, labels].to_numpy()
fig, ax = plt.subplots(figsize=(6.2, 5.0), layout='constrained')
ax.imshow(cm, cmap='Blues', vmin=0)
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=13,
                color='white' if cm[i,j] > cm.max()/2 else '#111111')
ax.set_xticks(range(3), labels); ax.set_yticks(range(3), labels)
ax.set_xlabel('Nhãn dự đoán'); ax.set_ylabel('Nhãn thực tế')
fig.savefig(OUTPUT/'confusion_matrix.png', dpi=220); plt.close(fig)

summary = pd.read_csv(R/'summary.csv').set_index('model').loc[['A_knn','A_tree','B_knn','B_tree']]
fig, ax = plt.subplots(figsize=(7.6, 4.4), layout='constrained')
x = np.arange(4); width = .34
for offset, col, label, color in [(-width/2,'macro_f1','Macro-F1','#356fa5'),
                                   (width/2,'dropout_recall','Recall Dropout','#dc8537')]:
    vals=summary[col].to_numpy(); bars=ax.bar(x+offset, vals, width, label=label, color=color)
    ax.bar_label(bars, labels=[f'{v:.3f}'.replace('.', ',') for v in vals], padding=3, fontsize=9)
ax.set_xticks(x, ['A / K-NN','A / Cây','B / K-NN','B / Cây'])
ax.set_ylim(0, 1); ax.set_ylabel('Điểm'); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.1f}'.replace('.', ',')))
ax.legend(loc='upper right', frameon=False)
fig.savefig(OUTPUT/'comparison.png', dpi=220); plt.close(fig)
print(f'Đã lưu 5 biểu đồ vào {OUTPUT}')

# EDA theo train và ma trận test, tái tạo trực tiếp từ dữ liệu và CSV đã lưu.
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), layout='constrained')
cols=['Application order','Previous qualification (grade)','Admission grade','Age at enrollment']
titles=['Thứ tự nguyện vọng','Điểm văn bằng trước','Điểm tuyển sinh','Tuổi nhập học']
for ax,col,title in zip(axes.flat,cols,titles):
    ax.boxplot(train[col].dropna(), vert=False, patch_artist=True,
               boxprops={'facecolor':'#8ab4d5'}, medianprops={'color':'#c35c2c','linewidth':2},
               flierprops={'marker':'.','markersize':3,'alpha':.35})
    ax.set(title=title, yticks=[]); ax.grid(axis='x', alpha=.2)
fig.suptitle('Box plot bốn đặc trưng số · train',fontsize=15,fontweight='bold')
fig.savefig(OUTPUT/'boxplots_numeric.png',dpi=220); plt.close(fig)

corr=train[cols].corr()
fig,ax=plt.subplots(figsize=(7.5,6),layout='constrained')
im=ax.imshow(corr,cmap='RdBu_r',vmin=-1,vmax=1)
ax.set_xticks(range(4),titles,rotation=25,ha='right'); ax.set_yticks(range(4),titles)
for i in range(4):
    for j in range(4):
        ax.text(j,i,f'{corr.iloc[i,j]:.2f}'.replace('.',','),ha='center',va='center',color='white' if abs(corr.iloc[i,j])>.6 else '#222')
fig.colorbar(im,ax=ax,label='Tương quan Pearson');ax.set_title('Tương quan giữa các đặc trưng số · train')
fig.savefig(OUTPUT/'correlation_heatmap.png',dpi=220);plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(11,4.2),layout='constrained')
for ax,col,title in zip(axes,['Gender','Daytime/evening attendance','Application mode'],['Giới tính (mã UCI)','Ca học (mã UCI)','Hình thức đăng ký · 5 mã phổ biến']):
    counts=train[col].value_counts().head(5)
    bars=ax.bar([str(v) for v in counts.index],counts.values,color='#397bb0');ax.bar_label(bars,padding=3,fontsize=9)
    ax.set(title=title,xlabel='Mã UCI',ylabel='Số mẫu');ax.set_ylim(0,counts.max()*1.18)
fig.savefig(OUTPUT/'categorical_distributions.png',dpi=220);plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(10,8.5),layout='constrained')
for ax,name,title in zip(axes.flat,['A_knn','A_tree','B_knn','B_tree'],['A / K-NN','A / Cây','B / K-NN','B / Cây']):
    cm=pd.read_csv(R/f'{name}_confusion_matrix.csv',index_col=0).loc[labels,labels].to_numpy()
    ax.imshow(cm,cmap='Blues',vmin=0,vmax=442)
    for i in range(3):
        for j in range(3):
            ax.text(j,i,f'{cm[i,j]}\n({cm[i,j]/cm[i].sum():.0%})',ha='center',va='center',color='white' if cm[i,j]>221 else '#111',fontsize=10)
    ax.set(xticks=range(3),xticklabels=labels,yticks=range(3),yticklabels=labels,xlabel='Dự đoán',ylabel='Thực tế',title=title)
fig.suptitle('Ma trận nhầm lẫn · số mẫu và tỷ lệ theo nhãn thật',fontsize=15,fontweight='bold')
fig.savefig(OUTPUT/'all_confusion_matrices.png',dpi=220);plt.close(fig)

# Vẽ lại các hình thí nghiệm sau các hình EDA.
import runpy
runpy.run_path(str(ROOT / 'scripts/experiment_figures.py'), run_name='__main__')
