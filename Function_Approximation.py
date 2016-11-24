#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 09:17:02 2016

@author: will
"""
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeRegressor     
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.kernel_approximation import RBFSampler
from blackJack import BlackjackEnv
import sklearn.pipeline
import sklearn.preprocessing

def qLearnWithFA(env, episodes, estimator, discount=1.0, trials=1000 ,\
                 epsilon=0.2,randomTrials=10,sampleSize=1000):
    # always resample from env, instead of following the path so that we know the total size
    # of experience. Use TD(0) off-policy update and experience replay, where the "on-policy"
    # follows epsilon-greedy of current estimator(s,a).
    # estimator needs to have fit WITH MASK as defined in GBM below/predict method and could do multiple output prediction, one
    # for each action.

    sDim = env.sDim # needs to implement this attribute in env
    aDim = env.aDim # needs to implement this attribute in env
    experience = np.zeros((episodes*trials,2*sDim+aDim+2)) # each dim corresponds to(s,a,r,s',done)
    
    index_ = 0 # track number of experience
    for i in range(episodes):
        for j in range(trials): # "on-policy" exploration
            s=env.reset() # restart s each trial
            
            if i < randomTrials: # random sample at the begining              
                a = env.action_space.sample() 
            else: # epsilon-greedy         
                a = np.argmax(estimator.predict(np.reshape(np.array(s),(1,-1))),1)[0] \
                        if np.random.rand()>epsilon else env.action_space.sample()
            s_next, r, done, _ = env.step(a)

            experience[index_,:sDim]=s
            experience[index_,sDim:sDim+aDim]=a
            experience[index_,sDim+aDim]=r
            experience[index_,sDim+aDim+1:sDim+aDim+1+sDim]=s_next
            experience[index_,sDim+aDim+1+sDim]=done                    
            index_+=1 
        
        # off-policy update estimator
        if index_ >= sampleSize:
            shuffleIndex=np.random.permutation(index_) # shuffle index for subsampling
            experience[:index_] = experience[shuffleIndex]
            X = experience[:sampleSize,:sDim]
            y = np.where(experience[:sampleSize,sDim+aDim+1+sDim]==1,\
                         experience[:sampleSize,sDim+aDim],\
                         experience[:sampleSize,sDim+aDim]+\
                         discount*np.max(estimator.predict(experience[:sampleSize,sDim+aDim+1:sDim+aDim+1+sDim]),1)
                         )
            mask = experience[:sampleSize,sDim] # works only for aDim=1
            estimator.fit(X,y,mask.astype('int64'))
        
    return estimator

class GBM(BaseEstimator, ClassifierMixin):
    def __init__(self,BaseEst,M_est,learnRate,BasePara,subFold):
        self.BaseEst=BaseEst
        self.M_est=M_est
        self.learnRate=learnRate
        self.estimator_=[]
        self.BasePara=BasePara
        self.subFold=subFold
        self.init_ = .0

    # Incorporate mask for RL     
    def fit(self,X,y,mask):
        n=y.shape[0]
        k_ = np.max(mask) + 1 # action starts at 0
        self.init_ = np.array([np.mean(y[mask==i]) for i in range(k_)])
        yhat = np.broadcast_to(self.init_, (n,k_))[np.arange(n),mask]
        kf = KFold(n, n_folds=self.subFold)
        for i in range(self.M_est):
            index=np.random.permutation(n) # shuffle index for subsampling
            X,y,yhat,mask = X[index,:],y[index],yhat[index],mask[index]
            for _,test in kf:
                target = np.zeros((test.shape[0],k_))
                target[np.arange(test.shape[0]),mask[test]] = y[test] - yhat[test]

                self.estimator_.append(self.BaseEst(**self.BasePara).fit(X[test],target))
                yhat+=self.learnRate*self.estimator_[-1].predict(X)[np.arange(n),mask]
        return self  
          
    def predict(self,X):
        yhat=np.reshape(self.init_,(1,-1))
        for clf in self.estimator_:
            yhat= yhat + self.learnRate*clf.predict(X)
        return yhat
    
#q_FA = qLearnWithFA(env1, 1000, GBM(DecisionTreeRegressor,100,0.1,{'max_depth':4,'splitter':'random','max_features':1},2))


def qLearnWithFA2(env, episodes, estimator, transformer, discount=1.0, trials=1000 ,\
                 epsilon=0.2,randomTrials=10,sampleSize=1000):
    # always resample from env, instead of following the path so that we know the total size
    # of experience. Use TD(0) off-policy update and experience replay, where the "on-policy"
    # follows epsilon-greedy of current estimator(s,a).
    # estimator needs to have fit/predict method and could do multiple output prediction, one
    # for each action. Use a list of estimator, one for each action.
    # this version uses partial_fit of SGDRegressor
    # transformer is a function that takes s as input, return a features of x to be used in later modeling
    A = env.action_space.n
    estimator_lst = [estimator for i in range(A)]
    sDim = env.sDim # needs to implement this attribute in env
    aDim = env.aDim # needs to implement this attribute in env
    experience = np.zeros((episodes*trials,2*sDim+aDim+2)) # each dim corresponds to(s,a,r,s',done)
    reshape_ = lambda x: np.reshape(x,(1,-1))
    
    index_ = 0 # track number of experience
    for i in range(episodes):
        for j in range(trials): # "on-policy" exploration
            s=env.reset() # restart s each trial
            
            if i < randomTrials: # random sample at the begining              
                a = env.action_space.sample() 
            else: # epsilon-greedy         
                if np.random.rand()>epsilon:
                    s_transf = transformer(reshape_(s))
                    a = np.argmax([estimator_lst[d].predict(s_transf)[0] for d in range(A)]) 
                else:
                    a = env.action_space.sample()
            s_next, r, done, _ = env.step(a)

            experience[index_,:sDim]=s
            experience[index_,sDim:sDim+aDim]=a
            experience[index_,sDim+aDim]=r
            experience[index_,sDim+aDim+1:sDim+aDim+1+sDim]=s_next
            experience[index_,sDim+aDim+1+sDim]=done                    
            index_+=1 
        
        # updates estimator from experience replay
        if index_ >= sampleSize:
            shuffleIndex=np.random.permutation(index_) # shuffle index for subsampling
            experience[:index_] = experience[shuffleIndex]

            for p in range(sampleSize): 
                if experience[p,sDim+aDim+1+sDim]==1 or \
                    np.any([model.coef_==None for model in estimator_lst]): # done or not fit
                    
                    estimator_lst[int(experience[p,sDim])].partial_fit(\
                        transformer(reshape_(experience[p,:sDim]))\
                        ,[experience[p,sDim+aDim]])
                else:
                    
                    X_tranf = transformer(reshape_(experience[p,sDim+aDim+1:sDim+aDim+1+sDim]))
                    estimator_lst[int(experience[p,sDim])].partial_fit(\
                        transformer(reshape_(experience[p,:sDim]))\
                        ,[experience[p,sDim+aDim] + discount * \
                         np.max([estimator_lst[d].predict(X_tranf)[0] for d in range(A)])])                    
                    

    return estimator_lst

    
def qLearnWithFA3(env, episodes, estimator, transformer, discount=1.0, trials=1000 ,\
                 epsilon=0.2,randomTrials=10,sampleSize=1000):
    # always resample from env, instead of following the path so that we know the total size
    # of experience. Use TD(0) off-policy update and experience replay, where the "on-policy"
    # follows epsilon-greedy of current estimator(s,a).
    # estimator needs to have fit/predict method and could do multiple output prediction, one
    # for each action. Use a list of estimator, one for each action.
    # this version uses batch fit to get least square estimator directly. ##
    # transformer is a function that takes s as input, return a features of x to be used in later modeling
    A = env.action_space.n
    estimator_lst = [estimator for i in range(A)]
    sDim = env.sDim # needs to implement this attribute in env
    aDim = env.aDim # needs to implement this attribute in env
    experience = np.zeros((episodes*trials,2*sDim+aDim+2)) # each dim corresponds to(s,a,r,s',done)
    reshape_ = lambda x: np.reshape(x,(1,-1))
    
    index_ = 0 # track number of experience
    for i in range(episodes):
        for j in range(trials): # "on-policy" exploration
            s=env.reset() # restart s each trial
            
            if i < randomTrials: # random sample at the begining              
                a = env.action_space.sample() 
            else: # epsilon-greedy         
                if np.random.rand()>epsilon:
                    s_transf = transformer(reshape_(s))
                    a = np.argmax([estimator_lst[d].predict(s_transf)[0] for d in range(A)]) 
                else:
                    a = env.action_space.sample()
            s_next, r, done, _ = env.step(a)

            experience[index_,:sDim]=s
            experience[index_,sDim:sDim+aDim]=a
            experience[index_,sDim+aDim]=r
            experience[index_,sDim+aDim+1:sDim+aDim+1+sDim]=s_next
            experience[index_,sDim+aDim+1+sDim]=done                    
            index_+=1 
        
        # updates estimator from experience replay
        if index_ >= sampleSize:
            shuffleIndex=np.random.permutation(index_) # shuffle index for subsampling
            experience[:index_] = experience[shuffleIndex]
            sampleData = experience[:sampleSize]

            for d in np.random.permutation(A):
                sampleD = sampleData[sampleData[:,sDim]==d]
                X = transformer(sampleD[:,:sDim])
                X_next = transformer(sampleD[:,sDim+aDim+1:sDim+aDim+1+sDim])
                y = np.where(sampleD[:,sDim+aDim+1+sDim]==1,\
                                 sampleD[:,sDim+aDim],\
                                 sampleD[:,sDim+aDim]+\
                                 discount*np.max(np.array([model.predict(X_next) for model in estimator_lst]),0)\
                                 ) \
                    if np.all([hasattr(model, 'coef_') for model in estimator_lst])\
                    else sampleD[:,sDim+aDim]
                estimator_lst[d].fit(X,y)
                                    

    return estimator_lst


    
''''''''''''''''''''''''''''''''''''''   
''' Test qLearnWithFA2 on Blackjack '''
''''''''''''''''''''''''''''''''''''''

#env1 = BlackjackEnv()
#estimator = SGDRegressor(learning_rate='constant')
#estimator3 = LinearRegression()
#
#''' prepare transformer '''
#observation_examples = np.array([env1.observation_space.sample() for x in range(10000)])
#scaler = sklearn.preprocessing.StandardScaler()
#scaler.fit(observation_examples)
#
#featurizer = sklearn.pipeline.FeatureUnion([
#        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
#        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
#        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
#        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
#        ])
#featurizer.fit(scaler.transform(observation_examples))
#transformer = lambda x: featurizer.transform(scaler.transform(x))
#
#q2 = qLearnWithFA2(env1, 1000, estimator, transformer, discount=1.0, trials=1000 ,\
#                 epsilon=0.2,randomTrials=10,sampleSize=1000)
#
#q3 = qLearnWithFA3(env1, 1000, estimator3, transformer, discount=1.0, trials=1000 ,\
#                 epsilon=0.2,randomTrials=10,sampleSize=1000)
