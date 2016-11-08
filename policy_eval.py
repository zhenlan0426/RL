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


def policy_eval(policy, env, V0=None, discount_factor=1.0, error=0.0001, maxIter=1000):
    # policy is a matrix of the shape (S, A)
    # env implments OpenAI env API
    # V0 is vector of shape(env.nS,) representing value function. So is the returned value
    
    if V0 == None:
        V0 = np.zeros(env.nS)
    
    R, P = covt_P(env)
    R_pi = np.sum(R * policy,1)
    P_pi = discount_factor * np.einsum('tfa,fa->ft',P,policy)
    
    for i in range(maxIter):
        V1 = R_pi + np.dot(P_pi,V0)
        if np.max(np.abs(V1-V0))<error:
            break
        else:
            V0 = V1
    return V1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
