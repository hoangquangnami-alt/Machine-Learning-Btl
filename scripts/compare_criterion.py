"""So sánh Gini và entropy bằng CV trên train cho cây B đã chọn; không dùng test."""
from pathlib import Path
import sys,json
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from student_success.data import CATEGORICAL,NUMERIC,SEED
from student_success.preprocess import build_preprocessor
R=ROOT/'data/experiment'
df=pd.read_csv(ROOT/'data/raw/data.csv',sep=';');df.columns=df.columns.str.strip()
idx=pd.read_csv(R/'train_indices.csv').row_index
params=json.loads((R/'selected.json').read_text())['B_tree']['params']
params={k.removeprefix('model__'):v for k,v in params.items()}
fold=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
rows=[]
for criterion in ['gini','entropy']:
 model=Pipeline([('preprocess',build_preprocessor('B')),('model',DecisionTreeClassifier(random_state=SEED,criterion=criterion,**params))])
 scores=cross_val_score(model,df.loc[idx,CATEGORICAL+NUMERIC],df.loc[idx,'Target'],cv=fold,scoring='f1_macro',n_jobs=1)
 rows.append({'criterion':criterion,'macro_f1_cv_mean':scores.mean(),'macro_f1_cv_std':scores.std()})
pd.DataFrame(rows).to_csv(R/'B_tree_criterion_cv.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
