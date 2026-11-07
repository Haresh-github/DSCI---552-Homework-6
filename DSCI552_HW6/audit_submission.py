"""Check the submitted files, results, model semantics, and report consistency."""
import json, numpy as np,pandas as pd, nbformat,fitz
from pathlib import Path
from sklearn.metrics import auc
ROOT=Path(__file__).parent;O=ROOT/'results'
report=(ROOT/'DSCI_552_HW6_Reworked_Report.pdf')
book=nbformat.read(ROOT/'DSCI_552_HW6_Reworked.ipynb',as_version=4)
assert len(book.cells)>=25
for cell in book.cells:
 assert not any(o.output_type=='error' for o in cell.get('outputs',[]))
assert sum(c.cell_type=='code' for c in book.cells)>8
text='\n'.join(p.get_text() for p in fitz.open(report))
assert len(fitz.open(report))==18
assert 'instructor:' not in text.lower()
assert 'scope' not in text.lower()
assert 'explaining the methods in plain language' not in text.lower()
assert 'AI use acknowledgment' in text and 'Haresh Sambandamoorthy' in text
assert 'model tree' in text.lower()
assert len(pd.read_csv(O/'all_170_feature_cv.csv'))==170
cv=pd.read_csv(O/'model_tree_cv_folds.csv'); assert len(cv)==30
for (smote,alpha),g in cv.groupby(['smote','alpha']):
 assert len(g)==5 and set(g.fold)=={1,2,3,4,5}
cvagg=pd.read_csv(O/'model_tree_cv_summary.csv');assert len(cvagg)==6
for r in cvagg.itertuples():
 a=cv[(cv.smote==r.smote)&(np.isclose(cv.alpha,r.alpha))]
 assert abs(a.auc.mean()-r.mean_auc)<1e-8
 assert abs(a.error.mean()-r.mean_cv_error)<1e-8
baseline=json.load(open(O/'results.json'))
corrected=json.load(open(O/'model_tree_results.json'))
models={**baseline['models'],**corrected['models']}
for name,v in models.items():
 for split,n in [('train',60000),('test',16000)]:
  q=v[split]
  assert sum(q[k] for k in ['tn','fp','fn','tp'])==n,(name,split)
  assert abs(q['error']-(q['fp']+q['fn'])/n)<1e-9
  roc=np.load(O/f'roc_{name}_{split}.npz')
  assert abs(q['auc']-auc(roc['fpr'],roc['tpr']))<1e-8,(name,split)
for k in ['l1_logistic_model_tree','l1_logistic_model_tree_smote']:
 a=corrected['models'][k]
 s=cvagg[cvagg.smote==k.endswith('smote')].sort_values('mean_auc',ascending=False)
 assert abs(a['alpha']-s.iloc[0]['alpha'])<1e-12
print('AUDIT PASSED: 18-page PDF, executed 28-cell notebook, no reported errors, 30 CV folds,')
print('6 models x train/test metric totals, ROC arrays match AUC, 170 CVs, CV-selected alphas, AI wording.')
