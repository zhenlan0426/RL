#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 08:28:05 2016

@author: will
"""
import numpy as np
import tensorflow as tf
from tflearn.activations import leaky_relu
from tflearn.layers.conv import conv_2d#, max_pool_2d
from tflearn.layers.core import fully_connected
import gym
import matplotlib.pyplot as plt
import cPickle

env = gym.envs.make("Breakout-v0")

""" do not use env.action_space.sample. It allows for 6 actions """
k = 4 # 4 possible action 0 (stay), 1 (fire), 2 (right) and 3 (left)
learning_rate = 1e-3
sampleSize = 10000 # additional sampling given updated q function
L,W = 80,72
lookback = 3 # how many frames to look back to make state transition markovian
iter_times = 1000 # major loop
eps = np.linspace(1,0.1,iter_times) # epsilon greedy policy
DataSize = 200000 # cached experience
discount = 1
batchSize = 50
trainIterTime = 400 # how many batch data will be sampled for each training 2.2
switchModel = 0.01 # control how freq each model get updated 
skipAction = 16
FlipMapping = {0:0,1:1,2:3,3:2}


# TO DO Data aug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def processImg(img):
    """Breakout-v0"""
    # precess make problem easier for agent by leveraging "human intelligence"
    # such as removing color and bounding box and reducing resolution. This greatly
    # reduce the dimention of this problem. However, this limits the agent's applicability.
    img = img[35:195,8:-8] # crop the bounding box
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


def copy_model_parameters(scopeFrom, scopeTo):

    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(scopeFrom)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(scopeTo)]
    e2_params = sorted(e2_params, key=lambda v: v.name)
    return [e2_v.assign(e1_v) for e1_v, e2_v in zip(e1_params, e2_params)]
    
def CNN(X_,y_,a_,scope):
    with tf.variable_scope(scope):
        network = conv_2d(X_, 32, 8, activation=leaky_relu)
        network = conv_2d(network, 32, 6, activation=leaky_relu)
        network = conv_2d(network, 32, 4, activation=leaky_relu)
    
        yhat0 = fully_connected(network, 64, activation=leaky_relu)
        yhat0 = fully_connected(yhat0, k)
            
        linearIndices = tf.range(batchSize) * k + a_
        #optimizer1 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optimizer1 = tf.train.RMSPropOptimizer(0.00025, 0.95, 0.0, 1e-4)
        
        loss0 = tf.reduce_mean(tf.squared_difference(y_, \
                        tf.gather(tf.reshape(yhat0, [-1]), linearIndices)))
        train0 = optimizer1.minimize(loss0)
    
    return yhat0, train0

# create three sets of parameters, one for each q function
yhat0, train0 = CNN(X_,y_,a_,'est0')
yhat1, train1 = CNN(X_,y_,a_,'est1')
copyParas = copy_model_parameters('est0', 'est1')

yhat_ = [yhat0,yhat1]
train_op = [train0,train1]
# End of Build Computation Graph


sess = tf.Session()
sess.run(tf.global_variables_initializer())

#################### 1.0 init sampling ####################
S_tot = np.zeros((DataSize,L,W,lookback),dtype=np.float32)
S_next_tot = np.zeros((DataSize,L,W,lookback),dtype=np.float32)
A_tot = np.zeros(DataSize,dtype=np.int32)
R_tot = np.zeros(DataSize,dtype=np.float32)
Done_tot = np.ones(DataSize,dtype=bool) # init = True to avoide q evaluation


counter = 0
done = True
while counter < DataSize: # do not update S_next_tot due to MC-training
    if done == True:
        s = np.broadcast_to(processImg(env.reset()), (L,W,lookback))
        discount_tot = np.zeros(DataSize)
        
    a = np.random.randint(k)    
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
for _ in range(1):
    index = np.random.permutation(DataSize) 
    for i in range(int(DataSize/batchSize)): 
        batchIndex = index[i*batchSize:(i+1)*batchSize]
        sess.run(train_op[0], {X_:S_tot[batchIndex], y_:R_tot[batchIndex], 
                           a_:A_tot[batchIndex]})


    
####################  Main loop  ####################   
S = np.zeros((sampleSize,L,W,lookback),dtype=np.float32)
S_next = np.zeros((sampleSize,L,W,lookback),dtype=np.float32)
A = np.zeros(sampleSize,dtype=np.int32)
R = np.zeros(sampleSize,dtype=np.float32)
Done = np.zeros(sampleSize,dtype=bool)


for iter_ in range(iter_times):
    
    # 2.1 Sample Step
    counter = 0 
    done = True
    while counter < sampleSize:
        if done == True:
            s = np.broadcast_to(processImg(env.reset()), (L,W,lookback))

        # yhat_ is a list of node in Tensor Graph
        if counter%skipAction == 0:
            a = np.argmax(sess.run(yhat_[0], {X_:np.expand_dims(s,0)}), 1)[0] \
                    if np.random.rand() > eps[iter_] else np.random.randint(k) 
        s_next, r, done, _ = env.step(a)

        S[counter] = s
        A[counter] = a
        R[counter] = r
        Done[counter] = done
        if not done:
            s_temp = insert_delete(s,processImg(s_next),False)
            S_next[counter] = s_temp
            
        S[counter+1] = s[:,::-1,:]
        A[counter+1] = FlipMapping[a]
        R[counter+1] = r
        Done[counter+1] = done
        if not done:
            S_next[counter+1] = s_temp

        S[counter+2] = s
        A[counter+2] = a
        R[counter+2] = r
        Done[counter+2] = done
        if not done:
            S_next[counter+2] = s_temp[:,::-1,:]

        S[counter+3] = s[:,::-1,:]
        A[counter+3] = FlipMapping[a]
        R[counter+3] = r
        Done[counter+3] = done
        if not done:
            S_next[counter+3] = s_temp[:,::-1,:]
        
        if not done:
            s = s_temp
        counter += 4    
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
        yMax = np.zeros(batchSize, dtype=np.float32) # run model only on Not-done x
        yMax[~Done_tot[batchIndex]] = np.max(sess.run(yhat_[1],\
                                        {X_:S_next_tot[batchIndex][~Done_tot[batchIndex]]}),1)
        target = R_tot[batchIndex] + discount * yMax
        sess.run(train_op[0], {X_:S_tot[batchIndex], y_:target, 
                           a_:A_tot[batchIndex]})    
    sess.run(copyParas)
    # 2.2 End of Training
    
    
    
#saver = tf.train.Saver()
#save_path = saver.save(sess, r"/DQN1.ckpt")    
    
    
    
    
name_ = [node.name for node in tf.trainable_variables()]   
paras = sess.run(tf.trainable_variables())
with open(r"paras.pkl", "wb") as output_file:
    cPickle.dump(paras, output_file)
    
    
for i in range(18):
    plt.hist(paras[i].flatten())
    plt.title(name_[i])
    plt.show()
    
for i in range(100000):
    plt.imshow(S_tot[i,:,:,0])
    plt.show()

done = True
for i in range(10000):
    if done:
        s = np.broadcast_to(processImg(env.reset()), (L,W,lookback))
    a =np.argmax(sess.run(yhat_[0], {X_:np.expand_dims(s,0)}), 1)
    s_next, r, done, _ = env.step(a)
    plt.imshow(s[:,:,0])
    plt.show()
    if not done:
        s = insert_delete(s,processImg(s_next),False)
        
a=sess.run(yhat_, {X_:S_tot[index[:10]]})
for i in range(10):
    plt.imshow(S_tot[index[i]])
    plt.show()