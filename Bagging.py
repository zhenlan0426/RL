# -*- coding: utf-8 -*-
"""
Created on Tue May 03 21:32:43 2016

@author: zhenlanwang
"""

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class Bagging_KClass(BaseEstimator, ClassifierMixin):
        
    def __init__(self,BaseEst,M_est,FixedPara,subSample,subFeature,RandomInit):
        # RandomInit is a string that randomly initiate the base para. 
        # to be called by exec(). the name within should be consistent with the ones
        # used in BasePara
        self.BaseEst=BaseEst
        self.M_est=M_est
        self.estimator_=[]
        self.FixedPara=FixedPara
        self.subSample=subSample
        self.subFeature=subFeature
        self.subFeatureIndex=[]
        self.RandomInit=RandomInit
        self.Likelihood=[]
        
    def fit(self,X,y):
        n , d = X.shape
        n_sample , d_sample = int(n*self.subSample), int(d*self.subFeature)
        n_test = n-n_sample
        k = len(np.unique(y))

        for i in range(self.M_est):
            # sampling
            IndexSample=np.random.permutation(n) 
            IndexFeature=np.random.permutation(d) 
            y_model=y[IndexSample[:n_sample]]
            X_model=X[IndexSample[:n_sample],:][:,IndexFeature[:d_sample]]
            y_test=y[IndexSample[n_sample:]]
            X_test=X[IndexSample[n_sample:],:][:,IndexFeature[:d_sample]]
            self.subFeatureIndex.append(IndexFeature[:d_sample])
            
            # estimation
            exec(self.RandomInit) 
            self.estimator_.append(self.BaseEst(**self.BasePara).fit(X_model,y_model))            

            # OOB fit
            yp=self.estimator_[-1].predict_proba(X_test)*k/2 #k/2 is here to prevent underflow
            target=yp[np.arange(n_test),y_test]            
            self.Likelihood.append(np.prod(np.where(target==0,0.01,target))) # floor. otherwise likelihood will be zero
        
        return self
        
        
    def predict_proba(self,X):
        w=np.array(self.Likelihood)
        w=w/np.sum(w)
        yp=w[0]*self.estimator_[0].predict_proba(X[:,self.subFeatureIndex[0]])        
        
        for i in range(1,self.M_est):
            yp+=w[i]*self.estimator_[i].predict_proba(X[:,self.subFeatureIndex[i]])
        
        return yp
        
    def predict(self,X):
        w=np.array(self.Likelihood)
        w=w/np.sum(w)
        yp=w[0]*self.estimator_[0].predict_proba(X[:,self.subFeatureIndex[0]])        
        
        for i in range(1,self.M_est):
            yp+=w[i]*self.estimator_[i].predict_proba(X[:,self.subFeatureIndex[i]])
        
        return np.argmax(yp,1)
            
''' tested on knn bagging. does not work well as the "best" para for knn has order of magnitude            
higher likelihood. As a result, the weight is so dominant by the "best" para that the ensemble
essentially equal to the "best" para """
            
            
            
            
            
            
            