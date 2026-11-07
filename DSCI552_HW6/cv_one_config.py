import sys,time,json
from run_model_trees import *
alpha=float(sys.argv[1]);smote=sys.argv[2]=='True'
tr=pd.read_csv(DATA/'aps_failure_training_set.csv',skiprows=20,na_values='na',low_memory=False)
X=tr.drop(columns=['class']).to_numpy(dtype=np.float32);y=(tr['class']=='pos').to_numpy(dtype=np.int8)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
rows=[]
for i,(trn,val) in enumerate(cv.split(X,y),1):
    tic=time.time();model=pipeline(alpha,smote);model.fit(X[trn],y[trn])
    z=model.predict_proba(X[val])[:,1]
    row=dict(smote=smote,alpha=alpha,fold=i,**metrics(y[val],z),nodes=model.named_steps['model_tree'].node_count_,internal_nodes=model.named_steps['model_tree'].internal_nodes_)
    rows.append(row)
    print('alpha',alpha,'smote',smote,'fold',i,'auc',round(row['auc'],5),'time',round(time.time()-tic,2),flush=True)
filename=f'model_cv_{"smote" if smote else "original"}_{str(alpha).replace(".","p")}.csv'
pd.DataFrame(rows).to_csv(OUT/filename,index=False)
print('SAVED',filename,flush=True)
