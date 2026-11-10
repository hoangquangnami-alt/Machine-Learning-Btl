"""ROC-AUC OvR cho B/K-NN: OOF trên train và mô tả trên test đã xem."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
from student_success.data import CATEGORICAL,NUMERIC,SEED
from student_success.preprocess import build_preprocessor
R=ROOT/'data/experiment';x=pd.read_csv(ROOT/'data/raw/data.csv',sep=';');x.columns=x.columns.str.strip()
tr=pd.read_csv(R/'train_indices.csv').row_index;te=pd.read_csv(R/'test_indices.csv').row_index
params=json.loads((R/'selected.json').read_text())['B_knn']['params'];params={k.removeprefix('model__'):v for k,v in params.items()}
pipe=Pipeline([('preprocess',build_preprocessor('B')),('model',KNeighborsClassifier(**params))])
labels=np.sort(x.Target.unique());X=x[CATEGORICAL+NUMERIC];y=x.Target
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
oof=cross_val_predict(pipe,X.loc[tr],y.loc[tr],cv=cv,method='predict_proba',n_jobs=1)
pipe.fit(X.loc[tr],y.loc[tr]);test=pipe.predict_proba(X.loc[te]);assert list(pipe.classes_)==list(labels)
rows=[]
for name,ids,probs in [('train_oof',tr,oof),('test_viewed',te,test)]:
 binary=label_binarize(y.loc[ids],classes=labels)
 auc=roc_auc_score(binary,probs,average=None)
 rows.append({'dataset':name,**{str(l):v for l,v in zip(labels,auc)},'macro_ovr':auc.mean()})
out=pd.DataFrame(rows);out.to_csv(R/'B_knn_roc_auc.csv',index=False);print(out.to_string(index=False))
