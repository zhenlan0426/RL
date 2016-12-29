#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 17:19:37 2016

@author: will
"""

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeRegressor     
import gym
from copy import deepcopy
import matplotlib.pyplot as plt


env = gym.envs.make("Breakout-v0")


k = 4 # 4 possible action 0 (stay), 1 (fire), 2 (right) and 3 (left)
learning_rate = 1e-2
depth = 12
max_features = 0.05
NbaseLearner = 50
subfold = 10


sampleSize = 10000 # additional sampling given updated q function
L,W = 80,80
lookback = 3 # how many frames to look back to make state transition markovian
iter_times = 100 # major loop
eps = np.linspace(0.5,0.1,iter_times) # epsilon greedy policy
DataSize = 100000 # cached experience
discount = 0.99
batchSize = 10000
trainIterTime = 2 
featureNumber = L*W*lookback
featureLowCutoff = 100
featureHighCutoff = DataSize - featureLowCutoff


                 



class GBM(BaseEstimator, ClassifierMixin):
    # This version of GBM is for multi-objective, with mask as for each observation
    # not all objectives target is known
    def __init__(self,BaseEst,M_est,learnRate,BasePara,subFold):
        self.BaseEst=BaseEst
        self.M_est=M_est
        self.learnRate=learnRate
        #self.estimator_=[]
        self.BasePara=BasePara
        self.subFold=subFold
        #self.init_ = .0

    # Incorporate mask for RL     
    def fit(self,X,y,mask):
        self.estimator_=[]
        n=y.shape[0]
        k_ = np.max(mask) + 1 # action starts at 0
        self.init_ = np.array([np.mean(y[mask==i]) for i in range(k_)])
        yhat = np.broadcast_to(self.init_, (n,k_))[np.arange(n),mask]
        kf = KFold(n, n_folds=self.subFold)
        for i in range(self.M_est):
            index=np.random.permutation(n) # shuffle index for subsampling
            X,y,yhat,mask = X[index,:],y[index],yhat[index],mask[index]
            for train,test in kf:
                target = np.zeros((test.shape[0],k_))
                target[np.arange(test.shape[0]),mask[test]] = y[test] - yhat[test]

                self.estimator_.append(self.BaseEst(**self.BasePara).fit(X[test],target))
                yhat[train]+=self.learnRate*self.estimator_[-1].\
                                predict(X[train])[np.arange(train.shape[0]),mask[train]]
        return self  
          
    def predict(self,X):
        yhat=np.reshape(self.init_,(1,-1))
        for clf in self.estimator_:
            yhat= yhat + self.learnRate*clf.predict(X)
        return yhat
        

def processImg(img):
    # precess make problem easier for agent by leveraging "human intelligence"
    # such as removing color and bounding box and reducing resolution. This greatly
    # reduce the dimention of this problem. However, this limits the agent's applicability.
    img = img[35:195] # crop the bounding box
    img = img[::2,::2,0] # downsample
    img[img != 0] = 1 # remove color information
    return img.astype(np.float).flatten()


def insert_delete(old, new,first_dim):
    # only support insert_delete in the first or the last dim
    # of the tensor
    
    if first_dim:
        return np.append(old[new.shape[0]:],new,0)
    else:
        return np.append(old[...,new.shape[-1]:],new,-1)
        
        
def copyGBM(model1,model2):
    model2.estimator_ = deepcopy(model1.estimator_)
    model2.init_ = deepcopy(model1.init_)
    
modelList = [GBM(DecisionTreeRegressor,NbaseLearner,\
            learning_rate,{'max_depth':depth,'splitter':'random','max_features':max_features},subfold),\
            GBM(DecisionTreeRegressor,NbaseLearner,\
            learning_rate,{'max_depth':depth,'splitter':'random','max_features':max_features},subfold) ]
                
                
# 1.0 init sampling 
S_tot = np.zeros((DataSize,featureNumber),dtype=np.int32)
S_next_tot = np.zeros((DataSize,featureNumber),dtype=np.int32)
A_tot = np.zeros(DataSize,dtype=np.int32)
R_tot = np.zeros(DataSize,dtype=np.float32)
Done_tot = np.ones(DataSize,dtype=bool) # init = True to avoide q evaluation


counter = 0
done = True
while counter < DataSize: # do not update S_next_tot due to MC-training
    if done == True:
        s = np.tile(processImg(env.reset()), lookback)
        discount_tot = np.zeros(DataSize)
        
    a = np.random.randint(k)    
    s_next, r, done, _ = env.step(a)
    discount_tot[counter] = 1
    
    S_tot[counter] = s
    A_tot[counter] = a 
    R_tot += r * discount_tot
    discount_tot *= discount
    if not done:
        s = insert_delete(s,processImg(s_next),True)
    counter += 1       
# 1.0 End of init sampling    
    
    
# 1.1 init training. MC-training instead of TD is used as at this stage bias of 
# q function dominates sample variance. For init training, iterate over the 
# entire Dataset once. 
# train_op is a list of Graph node for SGD-update, one for each model
temp = np.sum(S_tot==0,0)
featureIndex = [(temp>featureLowCutoff) * (temp<featureHighCutoff)]*2
modelList[0].fit(S_tot[:,featureIndex[0]],R_tot,A_tot) 
copyGBM(modelList[0],modelList[1])

modelList[0].subFold = 2
modelList[1].subFold = 2

# Main loop    
S = np.zeros((sampleSize,featureNumber),dtype=np.int32)
S_next = np.zeros((sampleSize,featureNumber),dtype=np.int32)
A = np.zeros(sampleSize,dtype=np.int32)
R = np.zeros(sampleSize,dtype=np.float32)
Done = np.zeros(sampleSize,dtype=bool)

for iter_ in range(iter_times):
    
    # 2.1 Sample Step
    counter = 0 
    done = True
    while counter < sampleSize:
        if done == True:
            s = np.tile(processImg(env.reset()), lookback)

        a = np.argmax(modelList[0].predict(np.expand_dims(s[featureIndex[0]],0))) \
                if np.random.rand() > eps[iter_] else np.random.randint(k) 
        s_next, r, done, _ = env.step(a)

        S[counter] = s
        A[counter] = a
        R[counter] = r
        Done[counter] = done
        if not done:
            s = insert_delete(s,processImg(s_next),True)
            S_next[counter] = s
        counter += 1    
    print "iteration:{}, reward:{}".format(iter_, np.mean(R))
    # 2.1 End of Sample Step
    
    # add newly sampled data to Dataset
    S_tot = insert_delete(S_tot,S,True)
    S_next_tot = insert_delete(S_next_tot,S_next,True)
    A_tot = insert_delete(A_tot,A,True)
    R_tot = insert_delete(R_tot,R,True)
    Done_tot = insert_delete(Done_tot,Done,True)
    
    # 2.2 Training
    index = np.random.permutation(DataSize) 
    for i in range(trainIterTime): 
        batchIndex = index[i*batchSize:(i+1)*batchSize]
        target = R_tot[batchIndex] + np.where(Done_tot[batchIndex],.0,\
                    modelList[0].predict(S_next_tot[batchIndex][:,featureIndex[0]])\
                    [range(batchSize),np.argmax(modelList[1].predict(S_next_tot[batchIndex][:,featureIndex[1]]),1)])
        
        updateModel = np.random.randint(2)
        temp = np.sum(S_tot[batchIndex]==0,0)
        featureIndex[updateModel] = (temp>featureLowCutoff) * (temp<featureHighCutoff)
        
        modelList[updateModel].fit(S_tot[batchIndex][:,featureIndex[updateModel]],target,A_tot[batchIndex]) 

    # 2.2 End of Training
    

    
done = True
for i in range(10000):
    if done:
        s = np.tile(processImg(env.reset()), lookback)
    a =np.argmax(modelList[0].predict(np.expand_dims(s[featureIndex[0]],0)))
    s_next, r, done, _ = env.step(a)
    plt.imshow(s_next[:,:,0])
    plt.show()
    if not done:
        s = insert_delete(s,processImg(s_next),True)
        
for i in range(100000):
    plt.imshow(S_tot[i].reshape(lookback,W,L)[0])
    plt.show()

a=modelList[0].predict(S_tot[index[-50:]][:,featureIndex[0]])
b=modelList[1].predict(S_tot[index[-50:]][:,featureIndex[1]])