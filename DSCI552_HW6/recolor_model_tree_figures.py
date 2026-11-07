from pathlib import Path
import shutil, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
ROOT=Path(__file__).parent
FIG=ROOT/'figures'
SRC=FIG
OUT=ROOT/'results'
# Replace grayscale copies with the original colored plots. These figures use exactly the same data.
# Original data/preprocessing scripts generate the other 15 colored figures.
# Improve distinction in the 3 model-tree plots originally issued in monochrome.
models=json.loads((OUT/'model_tree_results.json').read_text())['models']
cv=pd.read_csv(OUT/'model_tree_cv_summary.csv')
blue='#2769A5';orange='#E1812C';deep='#143A60'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(8.8,5.0),dpi=165)
for smote,label,color,marker in [(False,'Without SMOTE',blue,'o'),(True,'SMOTE within each fold',orange,'s')]:
 s=cv[cv.smote==smote].sort_values('alpha')
 ax.plot(s.alpha,s.mean_auc,linewidth=2.5,color=color,marker=marker,markersize=6,label=label)
ax.set_xscale('log');ax.set_xticks([.0001,.0003,.001],labels=['0.0001','0.0003','0.001']);ax.set(xlabel='L1 penalty (alpha)',ylabel='Mean validation ROC-AUC',title='Model tree: five-fold cross-validation')
ax.grid(alpha=.18);ax.legend(loc='best');fig.tight_layout();fig.savefig(FIG/'16_model_tree_alpha_cv.png',dpi=165,bbox_inches='tight');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11.9,4.6),dpi=165)
for ax,split in zip(axes,['train','test']):
 for name,label,color,style in [('l1_logistic_model_tree','Without SMOTE',blue,'-'),('l1_logistic_model_tree_smote','With SMOTE',orange,'--')]:
  curve=np.load(OUT/f'roc_{name}_{split}.npz')
  val=models[name][split]['auc']
  ax.plot(curve['fpr'],curve['tpr'],color=color,ls=style,lw=2.1,label=f'{label}: AUC {val:.3f}')
 ax.plot([0,1],[0,1],color='#8A96A1',ls=':',lw=1.4)
 ax.set(xlabel='False positive rate',ylabel='True positive rate',title=f'{split.title()} set ROC curves',xlim=(-.01,1.01),ylim=(-.01,1.01));ax.legend(fontsize=8,loc='lower right');ax.grid(alpha=.12)
fig.tight_layout();fig.savefig(FIG/'17_model_tree_roc.png',dpi=165,bbox_inches='tight');plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(8.6,7.1),dpi=165)
for row,(name,label) in enumerate([('l1_logistic_model_tree','Without SMOTE'),('l1_logistic_model_tree_smote','With SMOTE')]):
 for col,split in enumerate(['train','test']):
  val=models[name][split];mat=np.array([[val['tn'],val['fp']],[val['fn'],val['tp']]])
  ax=axes[row,col]
  cmap='Blues' if row==0 else 'Oranges'
  im=ax.imshow(np.log1p(mat),cmap=cmap,vmin=0,vmax=np.log1p(59000))
  for i in range(2):
   for j in range(2):
    ax.text(j,i,f'{mat[i,j]:,}',color='white' if mat[i,j]>=30000 else '#172B40',ha='center',va='center',fontsize=10,fontweight='bold')
  ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'],xlabel='Predicted class',ylabel='Actual class',title=f'{label} | {split.title()}')
fig.suptitle('Model-tree confusion matrices',fontsize=12,y=1.01)
fig.tight_layout();fig.savefig(FIG/'18_model_tree_confusion.png',dpi=165,bbox_inches='tight');plt.close(fig)
print('Restored color in',len(list(FIG.glob('*.png'))),'plots')
