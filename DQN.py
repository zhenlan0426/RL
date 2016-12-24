#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 08:28:05 2016

@author: will
"""

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

def rgb2grey(img):
    # last dim is channel
    return 0.2125*img[...,0] + 0.7154*img[...,1] + 0.0721*img[...,2]

def insert_delete(old, new,first_dim):
    # only support insert_delete in the first or the last dim
    # of the tensor
    n = new.shape[0]
    if first_dim:
        return np.append(old[n:],new,0)
    else:
        return np.append(old[...,n:],new,-1)
    
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
        a = np.argmax(sess.run(yhat_[np.random.randint(3)], {S_:s}), 1) \
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
