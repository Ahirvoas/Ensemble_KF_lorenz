# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 17:23:51 2018

@author: hirvoasa
"""
import numpy as np
def Lorenz_Single(xold,deltat,sigma,rho,beta):
    x=np.zeros(xold.shape)
    x[0] = xold[0]+deltat*sigma*(xold[1]-xold[0]) 
    x[1] = xold[1]+deltat*(rho*xold[0]-xold[1]-xold[0]*xold[2]) 
    x[2] = xold[2]+deltat*(-beta*xold[2]+xold[0]*xold[1])
    return x

def Lorenz_Ens(xold,deltat,sigma,rho,beta):
    x=np.zeros(xold.shape)
    x[0,:] = xold[0,:]+deltat*sigma*(xold[1,:]-xold[0,:]) 
    x[1,:] = xold[1,:]+deltat*(rho*xold[0,:]-xold[1,:]-xold[0,:]*xold[2,:]) 
    x[2,:] = xold[2,:]+deltat*(-beta*xold[2,:]+xold[0,:]*xold[1,:])
    return x