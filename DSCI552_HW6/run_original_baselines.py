# %% [markdown]
# 1(a) and 1(b): Load and examine the APS failure data
# The original training and test sets are kept separate throughout.

# %%
from pathlib import Path
import json
import math
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from joblib import dump
warnings.filterwarnings('ignore', category=FutureWarning)
SEED = 42
ROOT = Path.cwd()
DATA = ROOT / 'data'
FIG = ROOT / 'figures'
OUT = ROOT / 'results'
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
sns.set_theme(style='whitegrid',font_scale=0.87)
train = pd.read_csv(DATA/'aps_failure_training_set.csv', skiprows=20, na_values=['na'], low_memory=False)
test  = pd.read_csv(DATA/'aps_failure_test_set.csv', skiprows=20, na_values=['na'], low_memory=False)
features = [col for col in train.columns if col != 'class']
X_train_raw = train[features].astype('float32')
X_test_raw = test[features].astype('float32')
y_train = (train['class']=='pos').astype('int8').to_numpy()
y_test = (test['class']=='pos').astype('int8').to_numpy()
assert len(features) == 170 and train.shape == (60000,171) and test.shape == (16000,171)
assert (y_train==1).sum()==1000 and (y_test==1).sum()==375
results = {'data': {'train_rows':len(train),'test_rows':len(test),'features':len(features),
 'train_positive':int(y_train.sum()),'train_negative':int((1-y_train).sum()),
 'test_positive':int(y_test.sum()),'test_negative':int((1-y_test).sum()),
 'train_missing_cells':int(X_train_raw.isna().sum().sum()),
 'test_missing_cells':int(X_test_raw.isna().sum().sum()),
 'train_missing_pct':float(X_train_raw.isna().to_numpy().mean()*100),
 'test_missing_pct':float(X_test_raw.isna().to_numpy().mean()*100),
 'features_over_50pct_missing':int((X_train_raw.isna().mean()>0.5).sum())}}
print('Dataset:',len(train),'train,',len(test),'test,',len(features),'features.')
print('Class distribution:', train['class'].value_counts().to_dict())
print('Missing training entries: {:.2f}%'.format(results['data']['train_missing_pct']))

# %% [markdown]
# 1(b)(i)-(ii): Missing values and coefficients of variation
# Missing values are filled with the *training-set* median of each feature.
# The test set is transformed with the same medians, never refitted.

# %%
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train_raw).astype('float32')
X_test_imp = imputer.transform(X_test_raw).astype('float32')
means = X_train_imp.mean(axis=0, dtype=np.float64)
stds = X_train_imp.astype(np.float64).std(axis=0,ddof=1)
cv = np.divide(stds,means,out=np.full_like(stds,np.nan),where=np.abs(means)>1e-12)
cv_table = pd.DataFrame({'feature':features, 'mean':means, 'sample_sd':stds,
 'cv':cv, 'missing_pct':X_train_raw.isna().mean().to_numpy()*100}).sort_values('cv',ascending=False)
cv_table.to_csv(OUT/'all_170_feature_cv.csv',index=False,float_format='%.9g')
selected = cv_table['feature'].head(math.isqrt(170)).tolist()
results['data']['cv_top_13']=selected
results['data']['cv_top_5']=[{'feature':r.feature,'cv':float(r.cv)} for r in cv_table.head(5).itertuples()]
print('Top 13 CV features:', ', '.join(selected))

# %% [markdown]
# 1(b)(iii)-(v): Correlations, scatter plots, box plots, and class counts
# Plots show only a selection for readability; all 170 features are retained in every model.

# %%
def savefig(name, fig):
    fig.savefig(FIG/name,dpi=150,bbox_inches='tight',facecolor='white')
    plt.close(fig)

fig,ax=plt.subplots(figsize=(13.5,11.5))
corr=pd.DataFrame(X_train_imp,columns=features).corr()
sns.heatmap(corr,ax=ax,cmap='coolwarm',center=0,vmin=-1,vmax=1,
 xticklabels=False,yticklabels=False,cbar_kws={'label':'Pearson correlation'})
ax.set_title('Correlation among all 170 APS features')
ax.set_xlabel('170 features (full matrix)');ax.set_ylabel('170 features (full matrix)')
savefig('01_correlation_all_features.png',fig)

fig,ax=plt.subplots(figsize=(11,4.3))
top=cv_table.head(13).iloc[::-1]
ax.barh(top.feature,top.cv,color='#445c8c')
ax.set_xlabel('Coefficient of variation = sample SD / mean')
ax.set_title('13 features with the highest coefficient of variation')
savefig('02_top_13_cv.png',fig)

# Scatter plots use a fixed balanced visual sample so the rare positives remain visible.
rng=np.random.default_rng(SEED)
pos_idx=rng.choice(np.where(y_train==1)[0],size=250,replace=False)
neg_idx=rng.choice(np.where(y_train==0)[0],size=250,replace=False)
feature_ix=[features.index(name) for name in selected]
fig,axes=plt.subplots(4,4,figsize=(14,12))
for ax,name,ix in zip(axes.ravel(),selected,feature_ix):
    for idx,label,color in [(neg_idx,'Negative','#37588d'),(pos_idx,'Positive','#c04f43')]:
        vals=np.sign(X_train_imp[idx,ix])*np.log1p(np.abs(X_train_imp[idx,ix]))
        ax.scatter(np.arange(len(idx)),vals,s=11,c=color,alpha=.62,label=label)
    ax.set_title(name);ax.set_ylabel('signed log(1 + value)');ax.set_xlabel('Sample within class')
for ax in axes.ravel()[len(selected):]:ax.axis('off')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=2,bbox_to_anchor=(.5,1.012))
fig.suptitle('Scatter plots for the 13 highest-CV features (250 examples per class)',y=1.047,fontsize=15)
fig.tight_layout()
savefig('03_scatter_top_13_by_class.png',fig)

# Box plots: all rows, two classes per feature. log scale handles rare large values.
frame=pd.DataFrame(X_train_imp[:,feature_ix],columns=selected)
frame['Class']=np.where(y_train==1,'Positive','Negative')
frame_long=frame.melt(id_vars='Class',var_name='Feature',value_name='Value')
frame_long['Signed log value']=np.sign(frame_long.Value)*np.log1p(np.abs(frame_long.Value))
fig,ax=plt.subplots(figsize=(14,6))
sns.boxplot(data=frame_long,x='Feature',y='Signed log value',hue='Class',
           showfliers=False,ax=ax,palette={'Negative':'#37588d','Positive':'#c04f43'})
ax.tick_params(axis='x',rotation=45)
ax.set_title('Class-by-class box plots for all 13 selected features (outliers hidden for readability)')
savefig('04_boxplots_top_13_by_class.png',fig)
del frame,frame_long

# Pairwise scatter plots for the first 5 features; one balanced sample, not feature selection.
small_features=selected[:5]
vis_idx=np.r_[neg_idx,pos_idx]
small=pd.DataFrame(X_train_imp[vis_idx][:,[features.index(x) for x in small_features]],columns=small_features)
small=np.sign(small)*np.log1p(np.abs(small))
small['Class']=['Negative']*len(neg_idx)+['Positive']*len(pos_idx)
g=sns.pairplot(small,hue='Class',vars=small_features,corner=True,
 plot_kws={'alpha':.42,'s':13},diag_kind='hist',palette={'Negative':'#37588d','Positive':'#c04f43'})
g.figure.suptitle('Pairwise scatter plots: first 5 of 13 highest-CV features',y=1.02)
g.figure.savefig(FIG/'05_pairwise_scatter.png',dpi=125,bbox_inches='tight',facecolor='white')
plt.close(g.figure)

fig,ax=plt.subplots(figsize=(6,4))
ax.bar(['Negative','Positive'],[(y_train==0).sum(),(y_train==1).sum()],color=['#37588d','#c04f43'])
for i,n in enumerate([(y_train==0).sum(),(y_train==1).sum()]):ax.text(i,n+200,str(n),ha='center')
ax.set_ylabel('Number of training observations');ax.set_title('APS failure class balance')
savefig('06_class_balance.png',fig)
print('Visualization and CV tables saved.')

# %% [markdown]
# 1(c)-(d): Random forests, first without class balancing, then with balanced class weights.
# Out-of-bag (OOB) error is measured using training rows not sampled by a tree.

# %%
def assess(model,name):
    assessments={}
    for split,xx,yy in [('train',X_train_raw,y_train),('test',X_test_raw,y_test)]:
        probs=model.predict_proba(xx)[:,1]
        pred=(probs>=.5).astype(int)
        tn,fp,fn,tp=confusion_matrix(yy,pred,labels=[0,1]).ravel()
        fpr,tpr,_=roc_curve(yy,probs)
        assessments[split]={'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
            'error':float((fp+fn)/len(yy)),'auc':float(roc_auc_score(yy,probs)),
            'recall':float(recall_score(yy,pred,zero_division=0)),
            'precision':float(precision_score(yy,pred,zero_division=0)),
            'cost':int(10*fp+500*fn)}
        np.savez_compressed(OUT/f'roc_{name}_{split}.npz',fpr=fpr,tpr=tpr)
    results['models'][name]=assessments
    print(name, 'train error {:.4%}, test error {:.4%}, test AUC {:.4f}, test recall {:.1%}'.format(
         assessments['train']['error'],assessments['test']['error'],assessments['test']['auc'],assessments['test']['recall']))
    return assessments

results['models']={}
rf_params=dict(n_estimators=180,random_state=SEED,n_jobs=3,bootstrap=True,
               oob_score=True,max_features='sqrt',min_samples_leaf=1)
for name,weights in [('Random Forest',None),('Weighted Random Forest','balanced_subsample')]:
    t0=time.time()
    model=SkPipeline([('imputer',SimpleImputer(strategy='median')),
                      ('rf',RandomForestClassifier(**rf_params,class_weight=weights))])
    model.fit(X_train_raw,y_train)
    a=assess(model,name.replace(' ','_').lower())
    oob_err=float(1-model.named_steps['rf'].oob_score_)
    results['models'][name.replace(' ','_').lower()]['oob_error']=oob_err
    results['models'][name.replace(' ','_').lower()]['fit_seconds']=round(time.time()-t0,1)
    print('  OOB error {:.4%}; test error {:.4%}; {:.1f} sec'.format(oob_err,a['test']['error'],time.time()-t0))
    del model

# %% [markdown]
# 1(e)-(f): XGBoost with L1 penalty. Five-fold cross-validation chooses reg_alpha.
# Standard XGBoost uses one feature at a time in its tree splits. The L1 setting
# shrinks leaf weights; it does not fit a separate logistic model at every node.
# SMOTE is performed inside each training fold only, after imputation and scaling.

# %%
from sklearn.metrics import roc_auc_score
ALPHAS=[0.1,1.0,10.0]
FOLDS=5
cv=StratifiedKFold(n_splits=FOLDS,shuffle=True,random_state=SEED)
base_xgb=dict(n_estimators=130,max_depth=4,learning_rate=0.085,
              subsample=0.90,colsample_bytree=0.85,objective='binary:logistic',
              eval_metric='logloss',tree_method='hist',n_jobs=3,random_state=SEED,
              reg_lambda=1.0)
def make_pipeline(alpha,smote=False):
    steps=[('imputer',SimpleImputer(strategy='median')),
           ('scaler',StandardScaler())]
    if smote:steps.append(('smote',SMOTE(random_state=SEED,k_neighbors=5)))
    steps.append(('xgb',XGBClassifier(**base_xgb,reg_alpha=alpha)))
    return ImbPipeline(steps)

cv_rows=[]
for smote in [False,True]:
    print('Beginning 5-fold CV:', 'SMOTE inside folds' if smote else 'without SMOTE',flush=True)
    for alpha in ALPHAS:
        t0=time.time()
        for i,(idx_fit,idx_val) in enumerate(cv.split(X_train_raw,y_train),start=1):
            pipe=make_pipeline(alpha,smote=smote)
            pipe.fit(X_train_raw.iloc[idx_fit],y_train[idx_fit])
            probs=pipe.predict_proba(X_train_raw.iloc[idx_val])[:,1]
            pred=(probs>=0.5).astype(int)
            score_auc=roc_auc_score(y_train[idx_val],probs)
            score_err=float(np.mean(pred!=y_train[idx_val]))
            tn,fp,fn,tp=confusion_matrix(y_train[idx_val],pred,labels=[0,1]).ravel()
            cv_rows.append({'smote':smote,'alpha':alpha,'fold':i,
                'auc':float(score_auc),'error':score_err,'recall':float(tp/(tp+fn)),
                'cost':int(10*fp+500*fn)})
            print('  smote={} alpha={:g} fold={} AUC={:.4f} err={:.3%}'.format(smote,alpha,i,score_auc,score_err),flush=True)
            del pipe
        print('  alpha {:g} complete in {:.1f}s'.format(alpha,time.time()-t0),flush=True)
cv_df=pd.DataFrame(cv_rows)
cv_df.to_csv(OUT/'xgboost_cv_folds.csv',index=False)
cv_summary=cv_df.groupby(['smote','alpha'],as_index=False).agg(
    mean_auc=('auc','mean'),sd_auc=('auc','std'),mean_cv_error=('error','mean'),
    mean_recall=('recall','mean'),mean_cost_per_fold=('cost','mean'))
cv_summary.to_csv(OUT/'xgboost_cv_summary.csv',index=False,float_format='%.7g')
results['xgb_cv']=cv_summary.to_dict('records')
for smote,name in [(False,'xgboost'),(True,'xgboost_smote')]:
    subset=cv_summary.loc[cv_summary.smote==smote]
    best=subset.sort_values(['mean_auc','mean_cv_error'],ascending=[False,True]).iloc[0]
    alpha=float(best['alpha'])
    model=make_pipeline(alpha,smote=smote)
    model.fit(X_train_raw,y_train)
    a=assess(model,name)
    results['models'][name].update({'alpha':alpha,'cv_auc':float(best.mean_auc),
            'cv_error':float(best.mean_cv_error),'cv_auc_sd':float(best.sd_auc),
            'cv_recall':float(best.mean_recall)})
    print('{} selected alpha = {}, CV error {:.4%}, test error {:.4%}'.format(name,alpha,best.mean_cv_error,a['test']['error']))
    del model

# %% [markdown]
# Final comparisons for 1(c)-(f). All confusion matrices use [negative, positive]
# as the actual and predicted class order. Errors use the original unbalanced test set.

# %%
fig,axes=plt.subplots(2,2,figsize=(12,10))
for ax,key,title in zip(axes.ravel(),
 ['random_forest','weighted_random_forest','xgboost','xgboost_smote'],
 ['Random forest','Weighted random forest','XGBoost','XGBoost + SMOTE']):
    for split,color,style in [('train','#37588d','--'),('test','#bd4c44','-')]:
        z=np.load(OUT/f'roc_{key}_{split}.npz')
        ax.plot(z['fpr'],z['tpr'],style,c=color,lw=1.7,label='{} AUC {:.4f}'.format(split.title(),results['models'][key][split]['auc']))
    ax.plot([0,1],[0,1],'k:',lw=.8)
    ax.set_title(title);ax.set_xlabel('False positive rate');ax.set_ylabel('True positive rate');ax.legend(loc='lower right')
fig.suptitle('Training and test ROC curves',fontsize=15,y=1.01)
fig.tight_layout()
savefig('07_roc_curves.png',fig)

fig,axes=plt.subplots(4,2,figsize=(9.5,15))
for row,(key,title) in enumerate(zip(
 ['random_forest','weighted_random_forest','xgboost','xgboost_smote'],
 ['Random forest','Weighted random forest','XGBoost','XGBoost + SMOTE'])):
    for col,split in enumerate(['train','test']):
        a=results['models'][key][split]
        arr=np.array([[a['tn'],a['fp']],[a['fn'],a['tp']]])
        sns.heatmap(arr,annot=True,fmt='d',cmap='Blues',cbar=False,ax=axes[row,col],
            xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'])
        axes[row,col].set_xlabel('Predicted');axes[row,col].set_ylabel('Actual')
        axes[row,col].set_title('{}: {} confusion matrix'.format(title,split))
fig.tight_layout()
savefig('08_confusion_matrices.png',fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.3))
for smote,label,color in [(False,'No SMOTE','#37588d'),(True,'SMOTE within folds','#c04f43')]:
    subset=cv_summary[cv_summary.smote==smote].sort_values('alpha')
    axes[0].plot(subset.alpha,subset.mean_auc,'-o',label=label,color=color)
    axes[1].plot(subset.alpha,100*subset.mean_cv_error,'-o',label=label,color=color)
axes[0].set_ylabel('Mean validation ROC-AUC')
axes[1].set_ylabel('Mean validation error (%)')
for ax in axes:
    ax.set_xlabel('XGBoost L1 penalty (alpha)');ax.set_xticks(ALPHAS);ax.legend()
fig.suptitle('Five-fold cross-validation on training data',y=1.02)
fig.tight_layout()
savefig('09_cv_by_alpha.png',fig)

fig,ax=plt.subplots(figsize=(8,4.5))
keys=['random_forest','weighted_random_forest','xgboost','xgboost_smote']
names=['RF','Weighted RF','XGB','XGB + SMOTE']
vals=[results['models'][k]['test']['cost'] for k in keys]
ax.bar(names,vals,color=['#45638d','#8a9ab4','#77a59c','#c04f43'])
for i,v in enumerate(vals):ax.text(i,v+max(vals)*.018,f'{v:,}',ha='center')
ax.set_ylabel('APS cost = 10 x false positives + 500 x false negatives')
ax.set_title('Optional APS cost comparison on original test set')
savefig('10_test_cost.png',fig)

# ISLR 9.7.3 (a)-(h): hand-checkable separating line, margins, support vectors,
# a nonoptimal separating line and a new red point that breaks separability.
red=np.array([[3,4],[2,2],[4,4],[1,4]],dtype=float)
blue=np.array([[2,1],[4,3],[4,1]],dtype=float)
xx=np.linspace(.5,4.7,150)
fig,axes=plt.subplots(1,2,figsize=(12,5.4))
for ax in axes:
    ax.scatter(red[:,0],red[:,1],color='#c04f43',s=70,label='Red')
    ax.scatter(blue[:,0],blue[:,1],color='#37588d',s=70,label='Blue')
    for i,(x,y) in enumerate(np.r_[red,blue],start=1):ax.annotate(str(i),(x,y),xytext=(5,5),textcoords='offset points',fontsize=9)
    ax.plot(xx,xx-.5,'k-',label='Best line')
    ax.plot(xx,xx,'k--',lw=1,alpha=.65)
    ax.plot(xx,xx-1,'k--',lw=1,alpha=.65)
    ax.set_xlim(.5,4.8);ax.set_ylim(.4,5)
    ax.set_aspect('equal','box');ax.set_xlabel('$X_1$');ax.set_ylabel('$X_2$')
for j in [2,3]:axes[0].scatter([red[j-1,0]],[red[j-1,1]],s=200,facecolors='none',edgecolors='black',lw=1.4)
for j in [0,1]:axes[0].scatter([blue[j,0]],[blue[j,1]],s=200,facecolors='none',edgecolors='black',lw=1.4)
axes[0].set_title('Best line, margins, and circled support vectors')
axes[0].legend(loc='upper left',fontsize=8)
axes[1].plot(xx,.75*xx+.3,'-.',color='#7c9c5d',label='Non-optimal separator')
axes[1].scatter([3],[2],s=130,color='#ffb957',edgecolors='black',marker='*',zorder=5,label='New red point (3, 2)')
axes[1].set_title('Non-optimal line and extra point breaking separation')
axes[1].legend(loc='upper left',fontsize=7)
fig.tight_layout()
savefig('11_islr_9_7_3_diagram.png',fig)

# Export a single clean table of all train/test metrics.
summary_rows=[]
for key in keys:
    for split in ['train','test']:
        a=results['models'][key][split]
        summary_rows.append({'model':key,'split':split,**a})
pd.DataFrame(summary_rows).to_csv(OUT/'all_model_results.csv',index=False)
with open(OUT/'results.json','w') as f:json.dump(results,f,indent=2)
print('\nALL MODEL RESULTS')
print(pd.DataFrame(summary_rows).to_string(index=False))
print('\nSaved all plots and all model results in ./figures and ./results.')
