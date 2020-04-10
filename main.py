#Numpy is your best friend
import numpy as np 
from numpy.random import RandomState
from numpy.linalg import cholesky
#Tools for plotting
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.colors import cnames
import matplotlib.animation as animation

import models.Lorenz_wrapper as wrap #Wrapper to choose
from algos.utils import  RMSE, gen_truth, gen_obs #Useful tools for data studies
from algos.EnKF import EnKF #Algorithme of Ensemble Kalman Filter
prng = RandomState(6) # random number generator
# Background values
x_b = 3
y_b = -3
z_b = 21

# Simulation Parameters 
deltat = 1/100.
max_time=10.0
res = int(max_time / deltat)

dx = 3 # dimension of the state
dy = 3 # dimension of the observations

Ne=150 #Number of Elements for EnKF

sigma_true=10.0;rho_true=29.0;beta_true=8./3# physical true parameters

#Dynamical True Model Python version
fmdl_true=wrap.M(deltat=deltat)
m_true = lambda x: fmdl_true.Lorenz(x) 
#Dynamical Perturbated Model Python version
fmdl_perturb=wrap.M(deltat=deltat,sigma=sigma_true,rho=rho_true,beta=beta_true)
m_perturb = lambda x: fmdl_perturb.Lorenz(x)
#Observation Function
H = np.eye(3)
#H = H[(0,2),:] # first and third variables are observed
h = lambda x: H.dot(x)  # observation model
# Setting covariances for true
sig2_Q =0.0; sig2_R =0.0 # parameters
Q_true = np.eye(dx) *sig2_Q # model covariance
R_true = np.eye(dy) *sig2_R # observation covariance
#R_true = R_true[(0,2),:]

# Setting covariances for perturbated model
sig2_Q_perturb =0.001; sig2_R_perturb =0.5 # parameters
sig2_B_perturb=np.array([2,1.5,0.1]) # parameters
Q_perturb = np.eye(dx) *sig2_Q_perturb # model covariance
R_perturb = np.eye(dy) *sig2_R_perturb # observation covariance
#R_perturb = R_perturb[(0,2),:]
print(R_perturb)
# Prior state
x0_true = np.r_[1.5, -1.5, 21]
x0_perturb = np.r_[x_b, y_b, z_b]
B = np.eye(dx) *sig2_B_perturb
# Latent State Generation
X_true = gen_truth(m_true, x0_true, res, Q_true, prng) #Latent State Generation from true model
X_perturb = gen_truth(m_perturb, x0_perturb, res, Q_perturb, prng) #Latent State Generation from perturbated model
# Observations Generation²
Y_true = gen_obs(h, X_true, R_true, prng) #Observation from true model
Y_perturb = gen_obs(h, X_perturb, R_perturb, prng) #Observation from perturbated model
fig = plt.figure(figsize=(20,10))
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
# Prepare the axes limits
ax.set_xlim((-25, 25))
ax.set_ylim((-35, 35))
ax.set_zlim((5, 55))
ax.set_title('Lorenz attractor')
# Second Plot
lines_ref = ax.plot(Y_true[0,:], Y_true[1,:], Y_true[2,:], 'b-',label='Latent Variable')
ax.view_init(30, -20)
legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.savefig('Lorenz_model.png')
plt.show() 
# First Plot
f, axarr = plt.subplots(3, sharex=True)
axarr[0].plot(Y_true[0,:], 'b+', label='X Observation')
axarr[0].plot(X_true[0,:],'r-', label='X Latent Variable')
axarr[0].legend(loc="upper right")
axarr[1].plot(X_true[1,:], 'b+', label='Y Observation')
axarr[1].plot(X_true[1,:],'r-', label='Y Latent Variable')
axarr[1].legend(loc="upper right")
axarr[2].plot(Y_true[2,:], 'b+', label='Z Observation')
axarr[2].plot(X_true[2,:],'r-', label='Z Latent Variable')
axarr[2].legend(loc="upper right")     
fig = plt.figure(figsize=(20,10))
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
# Prepare the axes limits
ax.set_xlim((-25, 25))
ax.set_ylim((-35, 35))
ax.set_zlim((5, 55))
ax.set_title('Lorenz attractor')
# Second Plot
lines_ref = ax.plot(Y_true[0,:], Y_true[1,:], Y_true[2,:], 'b-',label='Latent Variable')
lines_fore = ax.plot(X_true[0,:], Y_perturb[1,:], Y_perturb[2,:], 'r+',label='Observation')
ax.view_init(30, -20)
legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.savefig('Observation_Lorenz_model.png')
plt.show() 