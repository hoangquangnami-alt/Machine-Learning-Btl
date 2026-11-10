"""Baselines, leakage sensitivity, ablations (CV on train) and descriptive bootstrap on viewed test."""
from pathlib import Path
import os
import json, sys
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from student_success.data import CATEGORICAL, NUMERIC, EXCLUDED, SEED, LABELS
from student_success.preprocess import build_preprocessor
R=Path(os.environ.get('FIT_RESULTS', ROOT/'data/experiment')); df=pd.read_csv(ROOT/'data/raw/data.csv',sep=';');df.columns=df.columns.str.strip()
tr=pd.read_csv(R/'train_indices.csv').row_index;te=pd.read_csv(R/'test_indices.csv').row_index
X=df[CATEGORICAL+NUMERIC];y=df.Target
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
rows=[]
for name,est in [('knn',KNeighborsClassifier()),('tree',DecisionTreeClassifier(random_state=SEED))]:
 p=Pipeline([('preprocess',build_preprocessor('A')),('model',est)])
 score=cross_validate(p,X.loc[tr],y.loc[tr],cv=cv,scoring={'f1':'f1_macro','accuracy':'accuracy'},n_jobs=1)
 p.fit(X.loc[tr],y.loc[tr]); pred=p.predict(X.loc[te]); truth=y.loc[te]
 rows.append({'model':name,'cv_f1_mean':score['test_f1'].mean(),'cv_f1_std':score['test_f1'].std(), 'test_accuracy':accuracy_score(truth,pred), 'test_macro_f1':f1_score(truth,pred,average='macro'), 'test_macro_precision':precision_score(truth,pred,average='macro',zero_division=0),'test_macro_recall':recall_score(truth,pred,average='macro',zero_division=0),'test_dropout_recall':recall_score(truth,pred,labels=['Dropout'],average='macro',zero_division=0)})
 print('baseline',name,flush=True)
pd.DataFrame(rows).to_csv(R/'default_baselines.csv',index=False)
# Sensitivity: add all excluded fields to a clone of A; macroeconomic and outcome fields as numeric.
base=build_preprocessor('A'); all36=clone(base);all36.transformers.append(('excluded','passthrough',EXCLUDED))
leak=[]
for name,prep,cols in [('admission_18',base,CATEGORICAL+NUMERIC),('all_36',all36,CATEGORICAL+NUMERIC+EXCLUDED)]:
 pipe=Pipeline([('preprocess',prep),('model',DecisionTreeClassifier(random_state=SEED))])
 scores=cross_validate(pipe,df.loc[tr,cols],y.loc[tr],cv=cv,scoring={'f1':'f1_macro','accuracy':'accuracy'},n_jobs=1)
 leak.append({'features':name,'f1_cv_mean':scores['test_f1'].mean(),'f1_cv_std':scores['test_f1'].std(),'accuracy_cv_mean':scores['test_accuracy'].mean()})
 print('leakage',name,flush=True)
pd.DataFrame(leak).to_csv(R/'feature_availability_cv.csv',index=False)
# Ablation: fixed B model hyperparameters; change one feature group at a time.
selected=json.loads((R/'selected.json').read_text())
def only_age(x): return x[:,[0]]
def only_delta(x): return x[:,[1]]
ablations=[]
for algorithm in ['knn','tree']:
 params={k.removeprefix('model__'):v for k,v in selected[f'B_{algorithm}']['params'].items()}
 for variant in ['full_B','no_rare_grouping','no_age_band','no_grade_difference','no_engineered']:
  prep=build_preprocessor('B')
  if variant=='no_rare_grouping': prep.set_params(categorical__onehot__min_frequency=None,categorical__onehot__handle_unknown='ignore')
  if variant=='no_engineered': prep.transformers=[t for t in prep.transformers if t[0]!='engineered']
  elif variant in ('no_age_band','no_grade_difference'):
   for name,transformer,cols in prep.transformers:
    if name=='engineered': transformer.steps.insert(1,('select',FunctionTransformer(only_delta if variant=='no_age_band' else only_age)))
  est=(KNeighborsClassifier(**params) if algorithm=='knn' else DecisionTreeClassifier(random_state=SEED,**params))
  pipe=Pipeline([('preprocess',prep),('model',est)])
  scores=cross_validate(pipe,X.loc[tr],y.loc[tr],cv=cv,scoring='f1_macro',n_jobs=1)['test_score']
  ablations.append({'model':f'B_{algorithm}','variant':variant,'macro_f1_cv_mean':scores.mean(),'macro_f1_cv_std':scores.std(),**{f'fold_{i}':v for i,v in enumerate(scores)}})
  print('ablation',algorithm,variant,flush=True)
pd.DataFrame(ablations).to_csv(R/'ablation_cv.csv',index=False)
# Stratified paired bootstrap on the already-viewed held-out predictions (fixed estimators, no refit).
predictions={name:pd.read_csv(R/f'{name}_predictions.csv') for name in ['A_knn','A_tree','B_knn','B_tree']}
truth=predictions['B_knn'].actual.to_numpy(); assert all(np.array_equal(q.actual.to_numpy(),truth) for q in predictions.values())
rng=np.random.default_rng(SEED);indices=[np.flatnonzero(truth==label) for label in LABELS]
pred={name:q.predicted.to_numpy() for name,q in predictions.items()}; samples={name:[] for name in pred}
for _ in range(2000):
 ids=np.concatenate([rng.choice(x,size=len(x),replace=True) for x in indices])
 for name in pred: samples[name].append(f1_score(truth[ids],pred[name][ids],average='macro',zero_division=0))
ci=[]
for name,arr in samples.items():
 low,high=np.quantile(arr,[.025,.975]);ci.append({'model':name,'macro_f1':f1_score(truth,pred[name],average='macro'),'ci_low':low,'ci_high':high,'iterations':2000,'seed':SEED})
pd.DataFrame(ci).to_csv(R/'test_bootstrap_ci.csv',index=False)
diff=np.array(samples['B_knn'])-np.array(samples['A_knn']);pd.DataFrame([{'comparison':'B_knn-A_knn','difference':f1_score(truth,pred['B_knn'],average='macro')-f1_score(truth,pred['A_knn'],average='macro'),'ci_low':np.quantile(diff,.025),'ci_high':np.quantile(diff,.975)}]).to_csv(R/'test_bootstrap_paired.csv',index=False)
print('bootstrap 2000 hoàn tất',flush=True)
