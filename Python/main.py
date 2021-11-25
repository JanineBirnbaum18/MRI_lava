# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:40:53 2021

@author: birnb
"""

# Imports
import copy
import sys
import os

import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
from scipy import interpolate as sciinterp

from read_dicom import *
from build_mesh import *
from gfmodel import *

from time import process_time

SESSION = sys.argv[1]
path_mag = sys.argv[2]
nens = int(sys.argv[3])
nitr = int(sys.argv[4])


# EnKF subroutine
def EnKF(A_EnKF, data, dataerr):
    A = A_EnKF.copy() # A matrix
    M = len(data) # Number of the measurement
    N = A_EnKF.shape[1] # Number of the ensemble
    Ndim = A_EnKF.shape[0] # Number of the measurements + parameters
    Np = Ndim - M # Number of the parameters

    OneN = np.ones([N, N]) / N # 1_N matrix

    A_bar = A @ OneN # A bar matrix: A_bar = A OneN
    A_prm = A - A_bar # A' = A - A_bar

    # H matrix
    H1 = np.zeros([Np, M])
    H2 = np.identity(M)
    H = np.vstack((H1, H2)).transpose()

    # Measurement Matrix
    d = np.kron(np.ones((N, 1)), data).transpose()

    # E-matrix: method I (using uncertainty)
    E = np.kron(np.ones((N, 1)), dataerr).transpose()

    # measurement + pertubation
    D = d + E

    # DHA = D - HA
    DHA = D - H @ A

    # HApE = HA' + E
    HApE = H @ A_prm + E

    # Singular value decomposition
    U, S, V = np.linalg.svd(HApE)
    SIGinv = (S @ S.T) ** (-1)
    
    # calculate the analysis ensemble
    X1 = SIGinv * U.transpose()
    X2 = X1 @ DHA
    X3 = U @ X2
    X4 = (H @ A_prm).transpose() @ X3
    Aa = A + A_prm @ X4

    return Aa

# Forward model 
def forward_model(mu, x_scale, y_scale, vels, mesh, spl, spl2d_u, spl2d_v, mask, frame, g, eta_max, endframe):
    rhoh, K, tau_y, n = mu
    print(x_scale,y_scale,vels,mesh,spl,spl2d_u,spl2d_v,mask,
          frame>endframe,g,rhoh,K,tau_y,n,eta_max)
    u_sol, v_sol, P_sol = gfmodel(x_scale,y_scale,vels,mesh,spl,spl2d_u,spl2d_v,mask,
                                  frame>endframe,g,rhoh,K,tau_y,n,eta_max)    
    return u_sol, v_sol, P_sol

# Read data
dat = pd.read_excel('C:/Users/birnb/Documents/Columbia Research/MRI/Data/Experiments list.xlsx')
gb = dat.groupby('Session')

path_full = 'C:/Users/birnb/Desktop/MRI/'+ SESSION

if 'cross' in path_mag: 
    rot = False # set to true to flip orientation of image
    xleft = 60
    windowSize = 3 # size for initial smoothing kernel
    sigma_x = 8 # gaussian smoothing in x-direction
    sigma_y = 0.05 # gaussian smoothing in y-direction
    thresh2 = 130 # rethreshold after smoothing (0,255)
    thresh = 0.002 # minimum magnitude of signal intensity
    
else: 
    rot = True # set to true to flip orientation of image
    xleft = 3
    windowSize = 3 # size for initial smoothing kernel
    sigma_x = 8 # gaussian smoothing in x-direction
    sigma_y = 0.05 # gaussian smoothing in y-direction
    thresh2 = 130 # rethreshold after smoothing (0,255)
    thresh = 0.004 # minimum magnitude of signal intensity

session = gb.get_group(SESSION)
df = session.loc[session['String']==path_mag]
dirs = np.array(['Vel_along','Vel_across','Vel_vert'])
dir_names = {'Vel_along':'Along-channel','Vel_across':'Across-channel','Vel_vert':'Vertical'}
venc = df[dirs].iloc[0].values
vel_stretch = venc[~np.isnan(venc)] 
vel_dir = [dir_names[a] for a in dirs[~np.isnan(venc)]]
velocity = len(vel_stretch)
if path_mag.split('_')[-1].isnumeric():
    if (int(path_mag.split('_')[-1])<3):
        path_vel = []
        path_mag_part = '_'.join(path_mag.split('_')[:-1])
        str_end = (int(path_mag.split('_')[-1])+1)*velocity-1
    else: 
        path_vel = [path_mag + '_P']
        path_mag_part = path_mag
        str_end = 0
else: 
    path_vel = [path_mag + '_P']
    path_mag_part = path_mag
    str_end = 0
while len(path_vel)<velocity:
    path_vel = np.append(path_vel,path_mag_part + '_P_' + str(str_end))
    str_end += 1

#frames = np.arange(df['Start frame'].values,df['End frame'].values,dtype=int)
frames = np.arange(df['Start frame'].values,len(os.listdir(
             path_full + '/' + path_mag + '/' + os.listdir(path_full + '/' + path_mag + '/')[0])),dtype=int)
gateframe = frames[0] + 1 # frame of gate removal (used to calculate time from release)
H = df['Height (cm)'].values/100
L = 0.2
phi_gas = df['Gas vol %'].values[0]/100
phi_solid = df['Particle vol %'].values[0]/100
eta_liq = df['Liquid viscosity (cSt*1000)'].values[0]*1000/975
delta_t = 10
endframe = df['End frame'].values[0]
    
x_shift = df['x_shift'].values[0] # position of MRI image left edge with respect to channel back (cm)
y_shift = df['y_shift'].values[0] # position of MRI image bottom edge with respect to channel bottom (cm)
t_shift = df['t_shift'].values[0] # time before gateframe when gate is actually removed


mags, masks, vels, frames, ts, ptsx_dict, ptsy_dict, spls, x_scale, y_scale, frames_dat = read_dicom(SESSION, path_mag, path_vel)

# fig, ax = plt.subplots(figsize=(20,15),nrows=np.int32(np.ceil(mags.shape[0]/5)),ncols=5)
masks_transparent = masks.copy()
masks_transparent[masks==1] = np.nan
# dx = 0.4/mags.shape[2] # need to get dx from dicom files
# x_scale = np.arange(mags.shape[2])*dx
# y_scale = np.arange(mags.shape[1])*dx
X,Y = np.meshgrid(x_scale,y_scale)
# for i, axi in enumerate(ax.flatten()):
#     axi.pcolormesh(x_scale,y_scale,mags[i,:,:],shading='auto',vmin=-0.01,vmax=0.02)
#     #axi.imshow(np.flipud(masks_transparent[i,:,:]),cmap='Greys',alpha=0.2)
#     axi.plot(ptsx_dict[i],ptsy_dict[i],'r')
    
g = 9.81
eta_max = 10**4

# Prepare ensemble
# number of the parameters
npar = 4 #
# Initial parameter guess
# Density at top and bottom (Kg/m^3)
# rho0c0 = np.random.standard_normal(nens)*975*phi_gas/5 + 975*(1-phi_gas/2)
rhohc0 = np.random.standard_normal(nens)*975*phi_gas/5 + 975*(1-phi_gas)

# Consistency, K (Pas)
Kc0 = np.abs(np.random.standard_normal(nens)*eta_liq/10 + eta_liq*(1-phi_solid/0.6)**(-2.5))
# yield strength, tau_y (Pa)
tauy_init = 10**(70*(phi_solid - 0.3))
tauyc0 = np.abs(np.random.standard_normal(nens)*(tauy_init/10+0.1) + tauy_init + 0.1)
# flow index, n
nc0 = np.abs(np.random.standard_normal(nens)*0.1 + 1)

# Inversion
# number of measurements
nobs = mags.shape[1]*mags.shape[2] # forward model output should match input data
# allocate the A matrix
A = np.zeros([npar + 2*nobs, nens])
# storing the parameters into the A matrix
#A[0, :] = rho0c0
A[0, :] = rhohc0
A[1, :] = Kc0
A[2, :] = tauyc0
A[3, :] = nc0

# allocate the root mean square error
RMSE = np.zeros([len(frames_dat),nens])
# allocate the parameter estimation
param_est = np.zeros([npar, nens, len(frames_dat)])
# storing the initial parameter
param_est[:, :, 0] = A[0:npar, :]
# store error from each iteration to check for convergence
conv = np.zeros([len(frames_dat),nitr])

print(frames_dat)
# for all time steps 
for step in (frames_dat[1:]-gateframe):
    step = int(step)
    print('frame = {}'.format(step+gateframe))
    Dat_u = vels[step,:,:,0].copy().flatten()
    Dat_u[masks[step,:,:].flatten()==0] = 0
    Dat_v = vels[step,:,:,1].copy().flatten()
    Dat_v[masks[step,:,:].flatten()==0] = 0
    
    vels_smooth = vels.copy()
    vels_smooth[masks==0,:] = 0
    
    umin = numpy.percentile(vels_smooth[step,:,:,0]*masks[step,:,:],1)
    umax = numpy.percentile(vels_smooth[step,:,:,0]*masks[step,:,:],99)

    vels_smooth[step,vels_smooth[step,:,:,0]<umin,0] = 0
    vels_smooth[step,vels_smooth[step,:,:,0]>umax,0] = 0

    vmin = numpy.percentile(vels_smooth[step,:,:,1]*masks[step,:,:],1)
    vmax = numpy.percentile(vels_smooth[step,:,:,1]*masks[step,:,:],99)

    vels_smooth[step,vels_smooth[step,:,:,1]<vmin,1] = 0
    vels_smooth[step,vels_smooth[step,:,:,1]>vmax,1] = 0
    
    spl2d_u = sciinterp.RectBivariateSpline(x_scale,y_scale,vels_smooth[step,:,:,0].transpose(),kx=3,ky=3,s=2.5e2)
    spl2d_v = sciinterp.RectBivariateSpline(x_scale,y_scale,vels_smooth[step,:,:,1].transpose(),kx=3,ky=3,s=2.5e2)
    
    Err_u = np.ones(nobs)*np.nanmean(np.sqrt((vels[step,:,:,0] - spl2d_u.ev(X,Y))**2))
    Err_v = np.ones(nobs)*np.nanmean(np.sqrt((vels[step,:,:,1] - spl2d_v.ev(X,Y))**2))
    
    mesh = build_mesh(ptsx_dict[step],ptsy_dict[step])
    
    # for all iterations
    for iteration in np.arange(0, nitr):
        print('itr = {}'.format(iteration))
        
        # calculate forecast ensemble (Step 2)
        for i in np.arange(nens):
            u_sol, v_sol, P_sol = forward_model(A[:4,i], x_scale, y_scale, vels[step,:,:,:], mesh, spls[step], spl2d_u, 
                                                spl2d_v, masks[step,:,:], frames[step], g, eta_max,endframe)
            u_sol[np.isnan(u_sol)] = 0
            v_sol[np.isnan(v_sol)] = 0

            # store the data into A matrix
            A[npar:npar+nobs, i] = u_sol.flatten()
            A[npar+nobs:npar+2*nobs, i] = v_sol.flatten()
            
            # Root mean square error (RMSE)
            RMSE[step,i] = (np.nansum((u_sol.flatten() - Dat_u) ** 2) / (len(Dat_u) - 1))**(1/2) + (np.nansum((v_sol.flatten() - Dat_v) ** 2) / (len(Dat_v) - 1))**(1/2) # could change to include measurement uncertainty
        
            #print('RMSE = {}'.format(RMSE[i].mean()))
            if (i%20 == 0): 
                print('Ensemble = {}'.format(i))
      
        # EnKF analysis (Step 3 & 4)
        A = EnKF(A, np.append(Dat_u,Dat_v), np.append(Err_u,Err_v))
        conv[step,iteration] = np.mean(RMSE[step,:])
        
    # store the parameter estimation
    param_est[:,:,step] = A[0:npar, :]
    
print('All done')

# Write results to files

param_est.tofile('param_est_' + path_mag + '.txt')
conv.tofile('conv_' + path_mag + '.txt')
