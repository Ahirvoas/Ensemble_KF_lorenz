# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:26:39 2018

@author: hirvoasa
"""
import numpy as np
from algos.utils import inv_svd,sqrt_svd

def EnKF(dx, T, dy, xb, B, Q, R, Ne, f, H, obs, prng):
    sqQ = sqrt_svd(Q)
    sqR = sqrt_svd(R)
    sqB = sqrt_svd(B)
    Xa = np.zeros([dx, Ne, T+1])
    Xf = np.zeros([dx, Ne, T])
    # Initialize ensemble
    for i in range(Ne):
        Xa[:,i,0] = xb + sqB.dot(prng.normal(size=dx))
    for t in range(T):
        
        #print(Xf[:,:,t].ndim)
        Xf[:,:,t] = f(Xa[:,:,t]) + sqQ.dot(prng.normal(size=(dx, Ne)))
        Y = H.dot(Xf[:,:,t]) 

        # Update
        if np.isnan(obs[0,t]):
            Xa[:,:,t+1] = Xf[:,:,t]
        else:
            Pfxx = np.cov(Xf[:,:,t])
            K = Pfxx.dot(H.T).dot(inv_svd(H.dot(Pfxx).dot(H.T) + R))
            innov = (np.tile(obs[:,t], (Ne, 1))).T+ sqR.dot(prng.normal(size=(dy, Ne))) - Y
            Xa[:,:,t+1] = Xf[:,:,t] + K.dot(innov)
    return Xa, Xf
