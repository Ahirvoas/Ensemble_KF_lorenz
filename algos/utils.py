# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 17:38:23 2018

@author: hirvoasa
"""
import numpy as np

def inv_svd(A):
    "Returns the inverse matrix by SVD"
    U, s, V = np.linalg.svd(A, full_matrices=True)
    invs = 1./s
    n = np.size(s)
    invS = np.zeros((n,n))
    invS[:n, :n] = np.diag(invs)
    invA=np.dot(V.T,np.dot(invS, U.T))
    return invA

def sqrt_svd(A):
   "Returns the square root matrix by SVD"

   U, s, V = np.linalg.svd(A)#, full_matrices=True)

   sqrts = np.sqrt(s)
   n = np.size(s)
   sqrtS = np.zeros((n,n))
   sqrtS[:n, :n] = np.diag(sqrts)

   sqrtA=np.dot(V.T,np.dot(sqrtS, U.T))

   return sqrtA

def gen_truth(f, x0, T, Q, prng):
  sqQ = sqrt_svd(Q)
  Nx = x0.size
  Xt = np.zeros((Nx, T+1))
  Xt[:,0] = x0
  for k in range(T):
    Xt[:,k+1] = f(Xt[:,k]) + sqQ.dot(prng.normal(size=Nx))
  return Xt

def gen_obs(h, Xt, R, prng):
  sqR = sqrt_svd(R)
  No = sqR.shape[0]
  T = Xt.shape[1] -1
  Yo = np.zeros((No, T))
  Yo[:] = np.nan
  for k in range(T):
    Yo[:,k] = h(Xt[:,k+1]) + sqR.dot(prng.normal(size=No))
  return Yo

def RMSE(E):
  return np.sqrt(np.mean(E**2))

