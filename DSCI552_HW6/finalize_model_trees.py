import json,time
from run_model_trees import *
files=sorted(OUT.glob('model_cv_*.csv'))
assert len(files)==6,files
cv_table=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
assert len(cv_table)==30
cv_table.to_csv(OUT/'model_tree_cv_folds.csv',index=False,float_format='%.9g')
summary=cv_table.groupby(['smote','alpha'],as_index=False).agg(
  mean_auc=('auc','mean'),sd_auc=('auc','std'),mean_cv_error=('error','mean'),
  mean_recall=('recall','mean'),mean_cost_per_fold=('cost','mean'))
summary.to_csv(OUT/'model_tree_cv_summary.csv',index=False,float_format='%.9g')
print('SUMMARY\n',summary.to_string(index=False),flush=True)
outputs={'description':'L1 logistic oblique tree with 5-fold CV (all original 60k training rows)',
         'alpha_grid':ALPHAS,'folds':FOLDS,'max_depth':MAX_DEPTH,
         'smote_full_training_set':True,'cv_summary':summary.to_dict(orient='records'),
         'models':{}}
tr=pd.read_csv(DATA/'aps_failure_training_set.csv',skiprows=20,na_values='na',low_memory=False)
te=pd.read_csv(DATA/'aps_failure_test_set.csv',skiprows=20,na_values='na',low_memory=False)
X=tr.drop(columns=['class']).to_numpy(dtype=np.float32)
T=te.drop(columns=['class']).to_numpy(dtype=np.float32)
y=(tr['class']=='pos').to_numpy(dtype=np.int8)
yt=(te['class']=='pos').to_numpy(dtype=np.int8)
for smote,name in [(False,'l1_logistic_model_tree'),(True,'l1_logistic_model_tree_smote')]:
    best=summary[summary.smote==smote].sort_values(['mean_auc','mean_cv_error'],ascending=[False,True]).iloc[0]
    alpha=float(best.alpha)
    model=pipeline(alpha,smote)
    t0=time.time(); model.fit(X,y)
    stored={'alpha':alpha,'cv_auc':float(best.mean_auc),'cv_auc_sd':float(best.sd_auc),
            'cv_error':float(best.mean_cv_error),'cv_recall':float(best.mean_recall),
            'internal_nodes':int(model.named_steps['model_tree'].internal_nodes_),
            'node_count':int(model.named_steps['model_tree'].node_count_),
            'training_seconds':round(time.time()-t0,2)}
    for split,xx,yy in [('train',X,y),('test',T,yt)]:
        prob=model.predict_proba(xx)[:,1]
        a=metrics(yy,prob)
        fpr,tpr,_=roc_curve(yy,prob)
        np.savez_compressed(OUT/f'roc_{name}_{split}.npz',fpr=fpr,tpr=tpr)
        stored[split]=a
    outputs['models'][name]=stored
    print('FINAL',name, json.dumps(stored),flush=True)
(OUT/'model_tree_results.json').write_text(json.dumps(outputs,indent=2,allow_nan=False))
fig,ax=plt.subplots(figsize=(8.4,4.8))
for smote,label,style in [(False,'Without SMOTE','o-'),(True,'SMOTE within folds','s--')]:
    s=summary[summary.smote==smote]
    ax.plot(s.alpha,s.mean_auc,style,label=label,color='black',alpha=.6 if smote else 1)
ax.set_xscale('log');ax.set_xticks(ALPHAS,labels=[f'{a:g}' for a in ALPHAS]);ax.set_xlabel('L1 penalty alpha')
ax.set_ylabel('Mean validation AUC');ax.set_title('L1 model trees: five-fold cross-validation')
ax.legend();ax.grid(alpha=.2)
fig.savefig(FIG/'16_model_tree_alpha_cv.png',dpi=160,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11.5,4.2))
for ax,split in zip(axes,['train','test']):
 for name,label,style in [('l1_logistic_model_tree','Model tree','-'),('l1_logistic_model_tree_smote','Model tree + SMOTE','--')]:
    roc=np.load(OUT/f'roc_{name}_{split}.npz')
    val=outputs['models'][name][split]['auc']
    ax.plot(roc['fpr'],roc['tpr'],style,color='black',lw=1.6,alpha=.65 if name.endswith('smote') else 1,label=f'{label}: AUC {val:.3f}')
 ax.plot([0,1],[0,1],':',color='gray');ax.set(xlabel='False positive rate',ylabel='True positive rate',title=split.title()+' ROC curves');ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(FIG/'17_model_tree_roc.png',dpi=160,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(8.5,7.0))
for row,name in enumerate(['l1_logistic_model_tree','l1_logistic_model_tree_smote']):
 for col,split in enumerate(['train','test']):
  vals=outputs['models'][name][split]
  mat=np.array([[vals['tn'],vals['fp']],[vals['fn'],vals['tp']]])
  a=axes[row,col];a.imshow(mat,cmap='Greys',vmin=0,vmax=59000)
  for i in range(2):
   for j in range(2):a.text(j,i,f'{mat[i,j]:,}',ha='center',va='center',color='white' if mat[i,j]>29000 else 'black')
  a.set(xticks=[0,1],yticks=[0,1],xticklabels=['negative','positive'],yticklabels=['negative','positive'],xlabel='Predicted',ylabel='True',title=f'{"SMOTE" if row else "Original"} - {split}')
fig.tight_layout();fig.savefig(FIG/'18_model_tree_confusion.png',dpi=160,bbox_inches='tight');plt.close(fig)
