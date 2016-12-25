#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 08:28:05 2016

@author: will
"""
import numpy as np
import tensorflow as tf
from tflearn.activations import leaky_relu
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import fully_connected
import gym


env = gym.envs.make("Breakout-v0")


k = 4 # 4 possible action 0 (stay), 1 (fire), 2 (right) and 3 (left)
learning_rate = 1e-4
sampleSize = 10000 # additional sampling given updated q function
L,W = 80,80
lookback = 3 # how many frames to look back to make state transition markovian
iter_times = 500 # major loop
eps = np.linspace(0.5,0.1,iter_times) # epsilon greedy policy
DataSize = 100000 # cached experience
discount = 1
batchSize = 20
trainIterTime = 1000 # how many batch data will be sampled for each training 2.2
switchModel = 0.1 # control how freq each model get updated 




# TO DO Data aug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def processImg(img):
    # precess make problem easier for agent by leveraging "human intelligence"
    # such as removing color and bounding box and reducing resolution. This greatly
    # reduce the dimention of this problem. However, this limits the agent's applicability.
    img = img[35:195] # crop the bounding box
    img = np.expand_dims(img[::2,::2,0],-1) # downsample
    img[img != 0] = 1 # remove color information
    return img.astype(np.float)


def insert_delete(old, new,first_dim):
    # only support insert_delete in the first or the last dim
    # of the tensor
    
    if first_dim:
        return np.append(old[new.shape[0]:],new,0)
    else:
        return np.append(old[...,new.shape[-1]:],new,-1)
        
        
# Build Computation Graph
tf.reset_default_graph()
X_ = tf.placeholder(shape=[None, L, W, lookback], dtype=tf.float32, name="X")
y_ = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
a_ = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

network = conv_2d(X_, 32, 3, activation=leaky_relu)
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation=leaky_relu)
network = max_pool_2d(network, 2)
network = conv_2d(network, 128, 3, activation=leaky_relu)
network = max_pool_2d(network, 2)
#network = conv_2d(network, 256, 3, activation=leaky_relu)
#network = tf.reduce_max(network, axis=(1,2))

yhat0 = fully_connected(network, 128, activation=leaky_relu)
yhat0 = fully_connected(yhat0, k, activation=leaky_relu)

yhat1 = fully_connected(network, 128, activation=leaky_relu)
yhat1 = fully_connected(yhat1, k, activation=leaky_relu)

yhat2 = fully_connected(network, 128, activation=leaky_relu)
yhat2 = fully_connected(yhat2, k, activation=leaky_relu)

yhat_ = [yhat0,yhat1,yhat2]


linearIndices = tf.range(batchSize) * k + a_
optimizer1 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer3 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)


loss0 = tf.reduce_mean(tf.squared_difference(y_, \
                tf.gather(tf.reshape(yhat0, [-1]), linearIndices)))
train0 = optimizer1.minimize(loss0)

loss1 = tf.reduce_mean(tf.squared_difference(y_, \
                tf.gather(tf.reshape(yhat1, [-1]), linearIndices)))
train1 = optimizer2.minimize(loss1)

loss2 = tf.reduce_mean(tf.squared_difference(y_, \
                tf.gather(tf.reshape(yhat2, [-1]), linearIndices)))
train2 = optimizer3.minimize(loss2)

train_op = [train0,train1,train2]

# End of Build Computation Graph


sess = tf.Session()
sess.run(tf.global_variables_initializer())

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
        s = np.broadcast_to(processImg(env.reset()), (L,W,lookback))
        discount_tot = np.zeros(DataSize)
        
    a = env.action_space.sample()    
    s_next, r, done, _ = env.step(a)
    discount_tot[counter] = 1
    
    S_tot[counter] = s
    A_tot[counter] = a 
    R_tot += r * discount_tot
    discount_tot *= discount
    if not done:
        s = insert_delete(s,processImg(s_next),False)
    counter += 1       
# 1.0 End of init sampling    
    
    
# 1.1 init training. MC-training instead of TD is used as at this stage bias of 
# q function dominates sample variance. For init training, iterate over the 
# entire Dataset once. 
# train_op is a list of Graph node for SGD-update, one for each model

index = np.random.permutation(DataSize) 
for i in range(int(DataSize/batchSize)): 
    batchIndex = index[i*batchSize:(i+1)*batchSize]
    sess.run(train_op[np.random.randint(3)], {X_:S_tot[batchIndex], y_:R_tot[batchIndex], 
                       a_:A_tot[batchIndex]})


    
# Main loop    
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
            s = np.broadcast_to(processImg(env.reset()), (L,W,lookback))

        # yhat_ is a list of node in Tensor Graph
        a = np.argmax(sess.run(yhat_[np.random.randint(3)], {X_:np.expand_dims(s,0)}), 1) \
                if np.random.rand() > eps[iter_] else env.action_space.sample() 
        s_next, r, done, _ = env.step(a)

        S[counter] = s
        A[counter] = a
        R[counter] = r
        Done[counter] = done
        if not done:
            s = insert_delete(s,processImg(s_next),False)
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
    modelIndex = np.random.permutation(3)
    for i in range(trainIterTime): 
        batchIndex = index[i*batchSize:(i+1)*batchSize]
        if np.random.rand()<switchModel:
            modelIndex = np.random.permutation(3)
        yMax,yNext = sess.run([yhat_[i] for i in modelIndex[1:]], \
                              {X_:S_next_tot[batchIndex]})
        target = R_tot[batchIndex] + np.where(Done_tot[batchIndex],.0,\
                                        discount*yNext[range(batchSize),np.argmax(yMax,1)])
        sess.run(train_op[modelIndex[0]], {X_:S_tot[batchIndex], y_:target, 
                           a_:A_tot[batchIndex]})    
    # 2.2 End of Training
    
    
    
    
    
    
    
    
name_ = [node.name for node in tf.trainable_variables()]   
paras = sess.run(tf.trainable_variables())
for i in range(20):
    plt.hist(paras[i].flatten())
    plt.title(name_[i])
    plt.show()
    
for i in range(100000):
    plt.imshow(S_tot[i,:,:,0])
    plt.show()