# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 17:06:09 2018

@author: hirvoasa
Lorenz model. Wrapper to python code
"""

class M:
   
    def __init__(self,deltat=0.15,sigma=10.0,rho=29.0,beta=8./3):
        "Lorenz-63 parameters"
        self.deltat=deltat
        self.sigma=sigma
        self.rho=rho
        self.beta=beta
    
    def Lorenz(self,xold):
        "Time integration of Lorenz-63 (single and ensemble)"
        import models.Lorenz_models as PyFor
        import numpy as np    
        x=np.zeros(xold.shape)
        if xold.ndim==1:
            x=PyFor.Lorenz_Single(xold,self.deltat,self.sigma,self.rho,self.beta) 
        else:
            x=PyFor.Lorenz_Ens(xold,self.deltat,self.sigma,self.rho,self.beta)
        return x


