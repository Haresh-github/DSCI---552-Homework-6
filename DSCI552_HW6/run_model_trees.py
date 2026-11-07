"""Homework 6 parts 1(e/f): true multivariate model trees with L1 logistic split models.
The original standard XGBoost experiment remains separate as a comparison, because
XGBoost gbtree does not provide logistic-regression hyperplanes at internal nodes.
"""
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore',category=ConvergenceWarning)
import os,time,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from l1_logistic_model_tree import L1LogisticModelTree

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
OUT=ROOT/'results'
FIG=ROOT/'figures'
OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
SEED=42
ALPHAS=[0.0001,0.0003,0.001]
MAX_DEPTH=3
FOLDS=5
SMOTE_NEIGHBORS=5

def log_signed(x):
    arr=np.asarray(x,dtype=np.float32)
    return (np.sign(arr)*np.log1p(np.abs(arr))).astype('float32')

def pipeline(alpha,smote):
    steps=[('imputer',SimpleImputer(strategy='median')),
        ('signed_log',FunctionTransformer(log_signed)),
        ('scaler',StandardScaler())]
    if smote:
        steps.append(('smote',SMOTE(random_state=SEED,k_neighbors=SMOTE_NEIGHBORS)))
    steps.append(('model_tree',L1LogisticModelTree(alpha=alpha,max_depth=MAX_DEPTH,
        min_samples_split=600,min_samples_leaf=150,min_gain=1e-5,
        max_iter=15,random_state=SEED)))
    return Pipeline(steps)

def metrics(y,p):
    pred=(p>=0.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
                error=float((fp+fn)/len(y)),auc=float(roc_auc_score(y,p)),
                recall=float(tp/max(1,tp+fn)),precision=float(tp/max(1,tp+fp)),
                cost=int(10*fp+500*fn))

def main():
    tr=pd.read_csv(DATA/'aps_failure_training_set.csv',skiprows=20,na_values='na',low_memory=False)
    te=pd.read_csv(DATA/'aps_failure_test_set.csv',skiprows=20,na_values='na',low_memory=False)
    X=tr.drop(columns=['class']).to_numpy(dtype=np.float32)
    T=te.drop(columns=['class']).to_numpy(dtype=np.float32)
    y=(tr['class']=='pos').to_numpy(dtype=np.int8)
    yt=(te['class']=='pos').to_numpy(dtype=np.int8)
    assert X.shape==(60000,170) and T.shape==(16000,170)
    assert y.sum()==1000 and yt.sum()==375
    cv=StratifiedKFold(n_splits=FOLDS,shuffle=True,random_state=SEED)
    splits=list(cv.split(X,y))
    rows=[]
    for with_smote in (False,True):
        for alpha in ALPHAS:
            start=time.time()
            for fold,(trn,val) in enumerate(splits,1):
                pipe=pipeline(alpha,with_smote)
                pipe.fit(X[trn],y[trn])
                prob=pipe.predict_proba(X[val])[:,1]
                met=metrics(y[val],prob)
                rec={'smote':with_smote,'alpha':alpha,'fold':fold,**met,
                     'nodes':int(pipe.named_steps['model_tree'].node_count_),
                     'internal_nodes':int(pipe.named_steps['model_tree'].internal_nodes_)}
                rows.append(rec)
                print('CV:',rec['smote'],rec['alpha'],fold,'AUC',round(rec['auc'],5),
                      'error',round(rec['error'],5),'splits',rec['internal_nodes'],flush=True)
            print('alpha',alpha,'smote',with_smote,'time',round(time.time()-start,1),flush=True)
    cv_table=pd.DataFrame(rows)
    cv_table.to_csv(OUT/'model_tree_cv_folds.csv',index=False,float_format='%.9g')
    summary=cv_table.groupby(['smote','alpha'],as_index=False).agg(
      mean_auc=('auc','mean'),sd_auc=('auc','std'),mean_cv_error=('error','mean'),
      mean_recall=('recall','mean'),mean_cost_per_fold=('cost','mean'))
    summary.to_csv(OUT/'model_tree_cv_summary.csv',index=False,float_format='%.9g')
    outputs={'description':'L1 logistic oblique tree, five-fold CV, no test-based tuning',
             'alpha_grid':ALPHAS,'folds':FOLDS,'max_depth':MAX_DEPTH,
             'smote_full_training_set':True,'cv_summary':summary.to_dict(orient='records'),
             'models':{}}
    for with_smote,name in [(False,'l1_logistic_model_tree'),(True,'l1_logistic_model_tree_smote')]:
        candidates=summary[summary.smote==with_smote].sort_values(
             ['mean_auc','mean_cv_error'],ascending=[False,True]);best=candidates.iloc[0]
        alpha=float(best.alpha)
        model=pipeline(alpha,with_smote)
        model.fit(X,y)
        stored={'alpha':alpha,'cv_auc':float(best.mean_auc),
                'cv_auc_sd':float(best.sd_auc), 'cv_error':float(best.mean_cv_error),
                'cv_recall':float(best.mean_recall),
                'internal_nodes':int(model.named_steps['model_tree'].internal_nodes_),
                'node_count':int(model.named_steps['model_tree'].node_count_)}
        for split,xx,yy in [('train',X,y),('test',T,yt)]:
            prob=model.predict_proba(xx)[:,1]
            a=metrics(yy,prob)
            fpr,tpr,_=roc_curve(yy,prob)
            np.savez_compressed(OUT/f'roc_{name}_{split}.npz',fpr=fpr,tpr=tpr)
            stored[split]=a
        outputs['models'][name]=stored
        print('FINAL:',name,'alpha=',alpha,'cv_auc=',round(best.mean_auc,6),
            'test=',stored['test'],'split_count=',stored['internal_nodes'],flush=True)
    (OUT/'model_tree_results.json').write_text(json.dumps(outputs,indent=2,allow_nan=False))
    fig,ax=plt.subplots(figsize=(8.2,4.4))
    for with_smote,label,style in [(False,'Without SMOTE','o-'),(True,'SMOTE inside folds','s--')]:
        s=summary[summary.smote==with_smote]
        ax.plot(s.alpha,s.mean_auc,style,label=label,color='black',alpha=.8 if with_smote else .55)
    ax.set_xscale('log')
    ax.set_xticks(ALPHAS,labels=[f'{a:g}' for a in ALPHAS]);ax.set_xlabel('L1 penalty alpha')
    ax.set_ylabel('Mean validation ROC-AUC');ax.set_title('Five-fold cross-validation for L1 logistic model trees')
    ax.legend();ax.grid(alpha=.2)
    fig.savefig(FIG/'16_model_tree_alpha_cv.png',dpi=160,bbox_inches='tight');plt.close(fig)
    print('All corrected model-tree files saved.',flush=True)

if __name__=='__main__':main()
