# -*- coding: utf-8 -*-
"""
Created on Tue Nov 08 13:51:10 2016

"""
import numpy as np


def covt_P(env):
    # convert the dict of dict, env.P to two tenor of the shape (dS,dA), (dS,dS,dA)
    # one for Reward, one for P. The dims are (From State, Action State),
    # (To State, From State, Action State). Reward function does not depend on next state
    ds = env.nS
    dA = env.nA
    R = np.zeros((ds,dA))
    P = np.zeros((ds,ds,dA))
    for s,j in env.P.iteritems():
        for a,q in j.iteritems():
            for prob, nextS, reward, _ in q:
                R[s,a] = reward 
                P[nextS,s,a] = prob
    
    return R, P


def policy_eval(policy, env, V0=None, discount_factor=1.0, error=0.0001, maxIter=1000,R=None,P=None):
    # policy is a matrix of the shape (S, A)
    # env implments OpenAI env API
    # V0 is vector of shape(env.nS,) representing value function. So is the returned value
    
    if V0 == None:
        V0 = np.zeros(env.nS)
    
    if R == None:
        R, P = covt_P(env) 
        
    R_pi = np.sum(R * policy,1)
    P_pi = discount_factor * np.einsum('tfa,fa->ft',P,policy)
    
    for i in range(maxIter):
        V1 = R_pi + np.dot(P_pi,V0)
        if np.max(np.abs(V1-V0))<error:
            print 'policy eval stops after {} iter'.format(i)
            break
        else:
            V0 = V1
    return V1
    
def policy_Greedy(V,env,discount_factor=1.0,R=None,P=None):
    # calculate greedy policy by one-step look-ahead given current value function
    if R == None:
        R, P = covt_P(env) 
    
    q = np.zeros((env.nS,env.nA))
    q[range(env.nS),np.argmax(np.einsum('tfa,t->fa',P,V) * discount_factor + R, 1)] = 1
    return q
    
    
def policy_iter(env,V0=None,policy0=None,discount_factor=1.0,error=0.0001, maxIter1=100, maxIter2=1000):
    R, P = covt_P(env)
    if V0 == None:
        V0 = np.zeros(env.nS)
    if policy0 == None:
        policy0 = np.ones((env.nS,env.nA))/env.nA
        
    for i in range(maxIter1):
        V0 = policy_eval(policy0, env, V0=V0, discount_factor=discount_factor,\
                        error=error, maxIter=maxIter2,R=R,P=P)
        policy1 = policy_Greedy(V0,env,discount_factor=discount_factor,R=R,P=P)
        print np.sum(policy1!=policy0)
        if np.sum(policy1!=policy0)==0:
            print 'policy iter stops after {} iter'.format(i)
            break
        else:
            policy0 = policy1
    
    return V0, policy1
    
    
def value_iter(env,V0=None,discount_factor=1,error=0.0001, maxIter=1000):
    R, P = covt_P(env)
    if V0 == None:
        V0 = np.zeros(env.nS)
    for i in range(maxIter):
        V1 = np.max(np.einsum('tfa,t->fa',P,V0) * discount_factor + R, 1)
        if np.max(np.abs(V1-V0))<error:
            print 'value iter stops after {} iter'.format(i)
            break
        else:
            V0 = V1
    return V1, policy_Greedy(V1,env,discount_factor=discount_factor,R=R,P=P)        
  
