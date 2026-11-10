"""EDA trên train và phân tích lỗi mô tả trên test đã xem; không chọn mô hình."""
from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from student_success.data import CATEGORICAL, NUMERIC, SEED
from student_success.preprocess import build_preprocessor
R=ROOT/'data/experiment'; IMG=ROOT/'report/course-template/images/results'
df=pd.read_csv(ROOT/'data/raw/data.csv',sep=';'); df.columns=df.columns.str.strip()
train=df.loc[pd.read_csv(R/'train_indices.csv').row_index].copy()
train['Nhóm tuổi']=pd.cut(train['Age at enrollment'], [0,19,24,29,200], labels=['< 20','20–24','25–29','≥ 30'])
fig,axes=plt.subplots(3,1,figsize=(10,10),layout='constrained')
for ax,col,title,maxgroups in [(axes[0],'Course','Ngành (mã UCI)',8),(axes[1],'Nhóm tuổi','Nhóm tuổi',None),(axes[2],'Application mode','Hình thức đăng ký (mã UCI)',8)]:
 counts=train[col].value_counts().head(maxgroups).index if maxgroups else train[col].dropna().unique()
 if col=='Nhóm tuổi': counts=['< 20','20–24','25–29','≥ 30']
 rows=[]
 for group in counts:
  sub=train[train[col]==group]; rows.append((str(group),len(sub),float((sub.Target=='Dropout').mean())))
 table=pd.DataFrame(rows,columns=['nhom','n','dropout_rate']);table.to_csv(R/f'eda_dropout_{col.replace(" ","_")}.csv',index=False)
 y=np.arange(len(rows));ax.barh(y,table.dropout_rate,color='#315ea8');ax.set_yticks(y,[f'{a} (n={b})' for a,b,_ in rows]);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel('Tỷ lệ Dropout trên train');
 for i,v in enumerate(table.dropout_rate): ax.text(min(v+.02,.9),i,f'{v:.0%}',va='center',fontsize=9)
 ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Tỷ lệ bỏ học theo nhóm · tập train',fontweight='bold',fontsize=16)
fig.savefig(IMG/'dropout_by_group.png',dpi=220);plt.close(fig)
name='B_tree'; pred=pd.read_csv(R/f'{name}_predictions.csv');test=df.loc[pred.row_index].copy().reset_index(drop=True)
test['actual']=pred.actual;test['predicted']=pred.predicted;test['Sai']=test.actual!=test.predicted
test['Nhóm tuổi']=pd.cut(test['Age at enrollment'],[0,19,24,29,200],labels=['< 20','20–24','25–29','≥ 30'])
for col,label in [('Course','course'),('Nhóm tuổi','age')]:
 group=test.groupby(col,observed=True).agg(n=('Sai','size'),errors=('Sai','sum'),error_rate=('Sai','mean')).query('n >= 30').sort_values('n',ascending=False)
 group.to_csv(R/f'error_groups_{label}.csv')
# Lưu cả tỷ lệ sai của B/K-NN theo cùng nhóm để đối chiếu các nhận xét trong báo cáo.
knn_pred=pd.read_csv(R/'B_knn_predictions.csv')
knn=df.loc[knn_pred.row_index].copy().reset_index(drop=True)
knn['Sai']=knn_pred.actual.ne(knn_pred.predicted)
knn['Nhóm tuổi']=pd.cut(knn['Age at enrollment'],[0,19,24,29,200],labels=['< 20','20–24','25–29','≥ 30'])
for col,label in [('Course','course'),('Nhóm tuổi','age')]:
    group=knn.groupby(col,observed=True).agg(n=('Sai','size'),errors=('Sai','sum'),error_rate=('Sai','mean')).query('n >= 30').sort_values('n',ascending=False)
    group.to_csv(R/f'B_knn_error_groups_{label}.csv')
params=json.loads((R/'selected.json').read_text())[name]['params']
est=DecisionTreeClassifier(random_state=SEED).set_params(**{k.removeprefix('model__'):v for k,v in params.items()})
pipe=Pipeline([('preprocess',build_preprocessor('B')),('model',est)]).fit(train[CATEGORICAL+NUMERIC],train.Target)
features=pipe['preprocess'].get_feature_names_out();importance=pipe['model'].feature_importances_
pd.DataFrame({'feature':features,'importance':importance}).sort_values('importance',ascending=False).to_csv(R/'B_tree_feature_importance.csv',index=False)
print('Đã lưu EDA theo nhóm, lỗi test theo nhóm và độ quan trọng của cây B.')
