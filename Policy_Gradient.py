# -*- coding: utf-8 -*-
      
import numpy as np 
import gym
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA




class Policy():
    # pi(a|s) follows a normal distribution, with mean transformer(s)*weight,
    # and variance sigma. 
    
    def __init__(self,d):
        # transformer is a function that creates features out of state
        # it should include a column of one as intercept
        # sigma is the std of the normal distribution and r is learning rate
        # parameter moved to fit method to allow for more flexible decay
        self.d = d
        self.weight = np.random.randn(d)/np.sqrt(d)
    
    def predict(self,s_transf):
        # s should be np.array of the shape (N,d0)
        return np.dot(s_transf,self.weight)
    
    def predict_sample(self,s_transf,sigma):
        # sigma_factor could be used to decay sigma
        return np.random.randn(s_transf.shape[0])*sigma + self.predict(s_transf)
    
    def partial_fit(self,s_transf,a,td_err,discount,r,sigma):
        # the first dim of s_transf, a, and td_err is N, # of sample
        # in order to do a mini-batch update
        self.weight += r/sigma**2 * np.mean((a-np.dot(s_transf,self.weight)).reshape(-1,1)*\
                                      discount.reshape(-1,1)*\
                                      td_err.reshape(-1,1)*s_transf,0)
        return self

    
class Value():
    # estimate value function as transformer(s)*weight
    
    def __init__(self,d):
        # transformer is a function that creates features out of state
        # it should include a column of one as intercept
        # sigma is the std of the normal distribution and r is learning rate
        # parameter moved to fit method to allow for more flexible decay
        self.d = d
        #self.weight = np.random.randn(d)/np.sqrt(d)
        self.weight = np.zeros(d)
        
    def predict(self,s_transf):
        # s should be np.array of the shape (N,d0)
        return np.dot(s_transf,self.weight)
    
    def partial_fit(self,s_transf,td_err,r):
        # the first dim of s_transf, a, and td_err is N, # of sample
        self.weight += r * np.mean(td_err.reshape(-1,1)*s_transf,0)
        return self    

class EnvList():
    
    def __init__(self, n, EnvList_,d_raw):
        # d_raw is the dim of state before transformer
        # EnvList_ should be like [envMake for i in range(n)]
        self.EnvList_ = EnvList_
        self.n = n
        self.d = d_raw
        
    def step(self,a):
        S_next,R,Done = np.zeros((self.n,self.d)),np.zeros(self.n),np.zeros(self.n)
        for i in range(self.n):
            S_next[i], R[i], Done[i], _ = self.EnvList_[i].step(a[i])
            if Done[i]: # reset terminal state
                self.EnvList_[i].reset()
        
        return S_next,R,Done
    
    def currentState(self):
        return np.array([env.state for env in self.EnvList_])

def ActorCritic(policy_,value_,transformer,envList_,iterTimes,learnR,sigma,discount,n):  
    discount_vec = np.zeros(n)
    for i in range(iterTimes):
        s_transf = transformer(envList_.currentState())
        action = policy_.predict_sample(s_transf,sigma)
        s_next,r,done = envList_.step(action.reshape(-1,1))
        td_err = discount*np.where(done,.0,value_.predict(transformer(s_next))) + \
                                   r - value_.predict(s_transf)
        value_.partial_fit(s_transf,td_err,learnR)
        policy_.partial_fit(s_transf,action,td_err,discount_vec,learnR,sigma)
        discount_vec = np.where(done, 1.0, discount_vec*discount)
        
    return policy_,value_
    
    
    
    
    
'''''''''''''''''''''''''''''''''''''''''''''''   
'''''' Test on MountainCarContinuous-v0''''''
'''''''''''''''''''''''''''''''''''''''''''''''    
env1 = gym.envs.make("MountainCarContinuous-v0")
iterTimes = 10000
r = 1e-2
sigma = 1
discount =1
n = 100

observation_examples = np.array([env1.observation_space.sample() for x in range(100000)])    
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf0", RBFSampler(gamma=8.0, n_components=50)),                                        
        ("rbf1", RBFSampler(gamma=4.0, n_components=50)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=50)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=50)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=50))
        ])    
pca = PCA(n_components=0.99,whiten=True)        
pca.fit(featurizer.fit_transform(observation_examples))
d = pca.n_components_ + 1 # extra one for intercept

transformer = lambda x: np.c_[np.ones(x.shape[0]),\
                                  pca.transform(featurizer.transform(x))]

envList_ = EnvList(n,[gym.envs.make("MountainCarContinuous-v0") for i in range(n)],2)
policy_ = Policy(d)
value_ = Value(d)

policy_,value_ = ActorCritic(policy_,value_,transformer,envList_,iterTimes,r,sigma,discount,n)
