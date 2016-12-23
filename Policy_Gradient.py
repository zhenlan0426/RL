# -*- coding: utf-8 -*-
      
import numpy as np 
import gym
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA
import  matplotlib.pyplot as plt
from sklearn.utils.extmath import cartesian


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
        S_next,R,Done = np.zeros((self.n,self.d)),np.zeros(self.n),np.zeros(self.n,dtype=bool)
        for i in range(self.n):
            S_next[i], R[i], Done[i], _ = self.EnvList_[i].step(a[i])
            if Done[i]: # reset terminal state
                self.EnvList_[i].reset()
                self.EnvList_[i].state = self.EnvList_[i].observation_space.sample()
        return S_next,R,Done
    
    def currentState(self):
        return np.array([env.state for env in self.EnvList_])

def ActorCritic(policy_,value_,transformer,envList_,iterTimes,learnR,sigma,discount,n):  
    discount_vec = np.zeros(n)
    cumR = 0
    monitorFreq = 1000
    
    for i in range(iterTimes):
        s_transf = transformer(envList_.currentState())
        action = policy_.predict_sample(s_transf,sigma)
        s_next,r,done = envList_.step(action.reshape(-1,1))
        td_err = discount*np.where(done,.0,value_.predict(transformer(s_next))) + \
                                   r - value_.predict(s_transf)
        value_.partial_fit(s_transf,td_err,learnR)
        policy_.partial_fit(s_transf,action,td_err,discount_vec,learnR,sigma)
        discount_vec = np.where(done, 1.0, discount_vec*discount)
        cumR += np.sum(r)
        if i%monitorFreq == 0:
            print "iteration {} with cumulative reward {}".format(i,cumR/monitorFreq/n)
            cumR = 0
    return policy_,value_
    
    
def mc_eval(env, episodes, policy, discount=1.0):
    # eval expected value over both s and a
    # policy is a fun maps from s to a
    val = 0
    for i in range(episodes):
        s=env.reset()
        done = False
        factor = 1 # discount
        while not done:         
            s, r, done, _ = env.step(policy(s))
            val += factor*r
            factor *= discount
    return val/episodes     
    
    
def mc_eval_batch(envList, episodes, policy, discount=1.0):
    # same as mc_eval, but does it in batch for speed
    # seems to give biased estimate due to algo favors 
    # batch the finish fast
    val = 0
    count = 0 
    n = envList.n
    cumR = np.zeros(n)
    factor = np.ones_like(n)
    
    while count<episodes:
        s = envList.currentState()
        s, r, done = envList.step(policy(s))
        cumR += factor*r
        factor = np.where(done,1.0,factor*discount)
        val += np.sum(np.where(done,cumR,.0))
        count += np.sum(done)
        cumR[done] = .0

        
    return val/count        
    
'''''''''''''''''''''''''''''''''''''''''''''''   
'''''' Test on MountainCarContinuous-v0''''''
'''''''''''''''''''''''''''''''''''''''''''''''    

# performance depends on r, sigma, n used 
env1 = gym.envs.make("MountainCarContinuous-v0")
iterTimes = 10000
r = 1e-4 # 1e-4 is the result of tuning
sigma = 0.1
discount =1
n = 1000

observation_examples = np.array([env1.observation_space.sample() for x in range(100000)])    
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf0", RBFSampler(gamma=8.0, n_components=25)),                                        
        ("rbf1", RBFSampler(gamma=4.0, n_components=25)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=25)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=25)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=25)),
        ("rbf5", RBFSampler(gamma=0.25, n_components=25))
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

mc_eval(env1, 10, lambda x:policy_.predict(transformer(x.reshape(1,-1))), discount=1.0)

#mc_eval_batch(EnvList(n,[gym.envs.make("MountainCarContinuous-v0") for i in range(n)],2),\
#               1000, lambda x:policy_.predict(transformer(x)).reshape(-1,1), discount=1.0)


x = cartesian([np.linspace(-1,0.6,100),np.linspace(-0.07,0.07,100)])
policy_x = policy_.predict(transformer(x))
value_x = value_.predict(transformer(x))

CS3 = plt.imshow(value_x.reshape(100,100))
plt.colorbar(CS3)
plt.show()
CS3 = plt.imshow(policy_x.reshape(100,100))
plt.colorbar(CS3)


''' render final policy '''
done = False
env1.reset()
policy = lambda x:policy_.predict(transformer(x.reshape(1,-1)))
while not done:
    env1.render(mode='human')
    _,_,done,_= env1.step(policy(env1.state))
env1.close()













sampleSize = 10000
eps = 0.1
L,W = 96,96
yhat
sess
lookback = 3
iter_times = 1000 
DataSize = 100000
discount = 1
batchSize = 20
trainIterTime = 1000
switchModel = 0.2 # control how freq each model get updated 
# Build Computation Graph

# End of Build Computation Graph


# 1.0 init sampling 
S_tot = np.zeros((DataSize,L,W,lookback))
S_next_tot = np.zeros((DataSize,L,W,lookback))
A_tot = np.zeros(DataSize)
R_tot = np.zeros(DataSize)
Done_tot = np.ones(DataSize,dtype=bool) # init = True to avoide q evaluation


counter = 0
done = True
while counter < DataSize: # do not update S_next_tot due to MC-training
    if done == True:
        s = np.broadcast_to(rgb2grey(env.reset()), (L,W,lookback))
        discount_tot = np.zeros(DataSize)
        
    a = env.action_space.sample()    
    s_next, r, done, _ = env.step(a)
    discount_tot[counter] = 1
    
    S_tot[counter] = s
    A_tot[counter] = a 
    R_tot += r * discount_tot
    discount_tot *= discount
    if not done:
        s = insert_delete(s,rgb2grey(s_next),False)
    counter += 1       
# 1.0 End of init sampling    
    
    
# 1.1 init training. MC-training instead of TD is used as at this stage bias of 
# q function dominates sample variance. For init training, iterate over the 
# entire Dataset once. 
# train_op is a list of Graph node for SGD-update, one for each model

index = np.random.permutation(DataSize) 
for i in range(int(DataSize/batchSize)): 
    batchIndex = index[i:(i+1)*batchSize]
    sess.run(train_op[np.random.randint(3)], {X_:S_tot[batchIndex], y_:R_tot[batchIndex], 
                       a_:A_tot[batchIndex]})


    
# init sample data    
S = np.zeros((sampleSize,L,W,lookback))
S_next = np.zeros((sampleSize,L,W,lookback))
A = np.zeros(sampleSize)
R = np.zeros(sampleSize)
Done = np.zeros(sampleSize,dtype=bool)

for iter_ in range(iter_times):
    
    # 2.1 Sample Step
    counter = 0 
    done = True
    while counter < sampleSize:
        if done == True:
            s = np.broadcast_to(rgb2grey(env.reset()), (L,W,lookback))

        # yhat_ is a list of node in Tensor Graph
        a = np.argmax(sess.run(yhat_[np.random.randint(3)], {S_:s}), 1) 
                if np.random.rand() > eps else env.action_space.sample() 
        s_next, r, done, _ = env.step(a)

        S[counter] = s
        A[counter] = a
        R[counter] = r
        Done[counter] = done
        if not done:
            s = insert_delete(s,rgb2grey(s_next),False)
            S_next[counter] = s
        counter += 1    
    # 2.1 End of Sample Step
    
    # add newly sampled data to Dataset
    S_tot = insert_delete(S_tot,S,True)
    S_next_tot = insert_delete(S_next_tot,S_next,True)
    A_tot = insert_delete(A_tot,A,True)
    R_tot = insert_delete(R_tot,R,True)
    Done_tot = insert_delete(Done_tot,Done,True)
    
    # 2.2 Training
    index = np.random.permutation(DataSize) 
    modelIndex = np.random.permutation(3)
    for i in range(trainIterTime): 
        batchIndex = index[i:(i+1)*batchSize]
        if np.random.rand()<switchModel:
            modelIndex = np.random.permutation(3)
        yMax,yNext = sess.run([yhat_[i] for i in modelIndex[1:]], {S_:S_next_tot[batchIndex]})
        target = R_tot[batchIndex] + np.where(Done_tot[batchIndex],.0,\
                                        discount*yNext[range(batchSize),np.argmax(yMax,1)])
        sess.run(train_op[modelIndex[0]], {X_:S_tot[batchIndex], y_:target, 
                           a_:A_tot[batchIndex]})    
    # 2.2 End of Training
