"""Oblique classification tree with L1-logistic hyperplanes at every internal node.
Each split has form: sum(beta[j] * standardized_feature[j]) > threshold.
All splits are learned on training folds, never on held-out observations.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier

@dataclass
class _Node:
    probability: float
    count: int
    positives: int
    depth: int
    coef: np.ndarray | None = None
    intercept: float = 0.0
    threshold: float = 0.0
    left: object = None
    right: object = None

class L1LogisticModelTree(ClassifierMixin, BaseEstimator):
    """Binary oblique tree. Uses penalized logistic regression to choose each split.

    At every internal node, fit an L1-penalized logistic regression using all
    input features. Choose an impurity-reducing cut in its linear decision
    score (a multivariate split). Terminal nodes output smoothed class rates.
    alpha represents the L1 penalty per observation in the average loss:
    minimize mean logistic loss + alpha * ||beta||_1.
    The logistic-loss model is fit by stochastic gradient descent with an L1 penalty.
    """
    def __init__(self, alpha=0.0005, max_depth=3, min_samples_split=600,
                 min_samples_leaf=150, min_gain=1e-5, max_iter=15, random_state=42):
        self.alpha=alpha
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf
        self.min_gain=min_gain
        self.max_iter=max_iter
        self.random_state=random_state

    @staticmethod
    def _best_threshold(scores, y, min_leaf):
        sort=np.argsort(scores,kind='mergesort')
        sorted_scores=scores[sort]
        sorted_y=y[sort]
        n=len(y)
        cut=np.flatnonzero(sorted_scores[:-1] < sorted_scores[1:])+1
        cut=cut[(cut>=min_leaf)&((n-cut)>=min_leaf)]
        if len(cut)==0:return None
        p=np.cumsum(sorted_y, dtype=np.float64)
        l1=p[cut-1]
        r1=p[-1]-l1
        lsize=cut.astype(np.float64)
        rsize=n-lsize
        frac=p[-1]/n
        parent_gini=2*frac*(1-frac)
        left_rate=l1/lsize
        right_rate=r1/rsize
        weighted_gini=(lsize/n)*2*left_rate*(1-left_rate)+(rsize/n)*2*right_rate*(1-right_rate)
        gain=parent_gini-weighted_gini
        ix=np.argmax(gain)
        j=cut[ix]
        return (float((sorted_scores[j-1]+sorted_scores[j])*0.5),float(gain[ix]))

    def _grow(self,X,y,depth):
        n=len(y); positives=int(y.sum())
        leaf=_Node((positives+1)/(n+2),n,positives,depth)
        if depth>=self.max_depth or n<self.min_samples_split or positives<2 or positives==n:
            return leaf
        gate=SGDClassifier(loss='log_loss',penalty='l1',alpha=self.alpha,
                           fit_intercept=True,tol=1e-3,max_iter=self.max_iter,
                           random_state=self.random_state,shuffle=True)
        gate.fit(X,y)
        coefs=gate.coef_.ravel().astype(np.float32,copy=True)
        if not np.any(coefs):return leaf
        score=np.asarray(X@coefs+gate.intercept_[0],dtype=np.float64)
        best=self._best_threshold(score,y,self.min_samples_leaf)
        if best is None or best[1]<self.min_gain:return leaf
        threshold,gain=best
        left=score<=threshold
        right=~left
        if int(left.sum())<self.min_samples_leaf or int(right.sum())<self.min_samples_leaf:
            return leaf
        leaf.coef=coefs
        leaf.intercept=float(gate.intercept_[0]); leaf.threshold=threshold
        leaf.left=self._grow(X[left],y[left],depth+1)
        leaf.right=self._grow(X[right],y[right],depth+1)
        return leaf

    def fit(self,X,y):
        X=np.asarray(X,dtype=np.float32)
        y=np.asarray(y,dtype=np.int8)
        if X.ndim!=2 or not np.isfinite(X).all():raise ValueError('X must be finite 2D array')
        if not set(np.unique(y)).issubset({0,1}):raise ValueError('y must be binary')
        self.n_features_in_=X.shape[1]
        self.classes_=np.array([0,1])
        self.root_=self._grow(X,y,0)
        self.node_count_,self.internal_nodes_=self._count_nodes(self.root_)
        return self
    @staticmethod
    def _count_nodes(node):
        if node.coef is None:return (1,0)
        n1,i1=L1LogisticModelTree._count_nodes(node.left)
        n2,i2=L1LogisticModelTree._count_nodes(node.right)
        return (1+n1+n2,1+i1+i2)

    def predict_proba(self,X):
        X=np.asarray(X,dtype=np.float32)
        if X.shape[1]!=self.n_features_in_:raise ValueError('Wrong number of features')
        p=np.empty(X.shape[0],dtype=np.float64)
        def traverse(node, idx):
            if len(idx)==0:return
            if node.coef is None:
                p[idx]=node.probability
                return
            score=X[idx]@node.coef + node.intercept
            mask=score <= node.threshold
            traverse(node.left,idx[mask]);traverse(node.right,idx[~mask])
        traverse(self.root_,np.arange(X.shape[0]))
        return np.column_stack((1-p,p))

    def predict(self,X):
        return (self.predict_proba(X)[:,1]>=0.5).astype(np.int8)
