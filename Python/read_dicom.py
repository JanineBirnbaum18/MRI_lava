# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 12:56:02 2021

@author: birnb
"""

# Imports
import numpy as np
import pandas as pd
from pydicom import dcmread
from zipfile import ZipFile
import os
import cv2
from scipy import interpolate as sciinterp
from matplotlib.collections import PolyCollection
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import sys

def read_dicom(SESSION, path_mag, path_vel):
    # retrive metadata
    df = pd.read_excel('C:/Users/birnb/Documents/Columbia Research/MRI/Data/Experiments list.xlsx')
    
    dat = df.loc[df.Session == SESSION].loc[df.String==path_mag]
    
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
    
    dirs = np.array(['Vel_along','Vel_across','Vel_vert'])
    dir_names = {'Vel_along':'Along-channel','Vel_across':'Across-channel','Vel_vert':'Vertical'}
    venc = dat[dirs].iloc[0].values
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
    frames = np.arange(dat['Start frame'].values[0],
                       len(os.listdir(path_full + '/' + path_mag + '/' + os.listdir(path_full + '/' + path_mag + '/')[0])),
                       dtype=int)
    frames_dat = []
    gateframe = frames[0] + 1 # frame of gate removal (used to calculate time from release)
    H = dat['Height (cm)'].values[0]/100
    L = 0.2
    phi_gas = dat['Gas vol %'].values[0]/100
    phi_solid = dat['Particle vol %'].values[0]/100
    eta_liq = dat['Liquid viscosity (cSt*1000)'].values[0]*1000/975
    delta_t = 10
        
    x_shift = dat['x_shift'].values[0]/100 # position of MRI image left edge with respect to channel back (cm)
    y_shift = dat['y_shift'].values[0]/100 # position of MRI image bottom edge with respect to channel bottom (cm)
    t_shift = dat['t_shift'].values[0] # time before gateframe when gate is actually removed

    # visualize data
    ds = dcmread(path_full + '/' + path_mag + '/' + os.listdir(path_full + '/' + path_mag + '/')[0] + '/' + os.listdir(
                 path_full + '/' + path_mag + '/' + os.listdir(path_full + '/' + path_mag + '/')[0])[gateframe])
    gatetime = ds.AcquisitionTime
    
    if rot:
        mags = np.zeros((frames.size,ds.pixel_array.shape[1], ds.pixel_array.shape[0]))
    else: 
        mags = np.zeros((frames.size,ds.pixel_array.shape[0], ds.pixel_array.shape[1]))
    masks = np.zeros_like(mags)
    
    if velocity>0: 
        if rot: 
            vels = np.zeros((frames.size,ds.pixel_array.shape[1], ds.pixel_array.shape[0],velocity))
        else:
            vels = np.zeros((frames.size,ds.pixel_array.shape[0], ds.pixel_array.shape[1],velocity))
    
    ts = np.zeros(frames.size)
    ptsx_dict = {}
    ptsy_dict = {}
    spls = {}
    
    fig, ax = plt.subplots()
    
    im1 = []
    
    # read each frame
    for i,frame in enumerate(frames):
        #print('frame = ' + str(frame))
        ds = dcmread(path_full + '/' + path_mag + '/' + os.listdir(path_full + '/' + path_mag + '/')[0] + '/' + os.listdir(
                     path_full + '/' + path_mag + '/' + os.listdir(path_full + '/' + path_mag + '/')[0])[frame])
        dt = float(ds.AcquisitionTime) - float(gatetime)
        
        mag = ds.pixel_array.astype(float)
        mag = mag/(2**ds.BitsStored)
        if rot:
            dx = ds.PixelSpacing[0]/10**3
            dy = ds.PixelSpacing[1]/10**3
            mag = mag.transpose()
        else: 
            dx = ds.PixelSpacing[1]/10**3
            dy = ds.PixelSpacing[0]/10**3
        x_scale = np.arange(mag.shape[1])*dx
        y_scale = np.arange(mag.shape[0])*dy
        xmax = x_scale[-1]
        height = y_scale[-1]
        mag = np.flipud(mag)
        
        mags[i,:,:] = mag
        
        # Read image
        im_mag = np.zeros_like(mag).astype(np.uint8)
        im_mag[mag<thresh] = 255
     
        # Remove low signal regions inside flow
        im_floodfill = im_mag.copy()
        cv2.floodFill(im_floodfill, None, (0, 0), 0)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_out = im_mag & im_floodfill_inv
    
        # Remove high signal regions outside flow
        im_floodfill = im_out.copy()
        cv2.floodFill(im_floodfill, None, (xleft, int(y_shift/dy)+1), 255) # select interior point
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_out = ~im_out & (~im_floodfill_inv)
        
        # Smooth outline
        
        im_blur = cv2.blur(im_out,(windowSize,windowSize), borderType=0)
        
        if dt>=0:
            im_blur = cv2.GaussianBlur(im_blur,(int(2*np.ceil(2*sigma_x)+1),int(2*np.ceil(2*sigma_y)+1)),sigma_x,sigma_y,
                                  borderType=0) 
    
        # Rethreshold
        mask = im_blur>thresh2
        #mask = np.flipud(mask)
        cs = ax.contour(x_scale, y_scale, mask, levels=1,colors='w',linewidths=2)
    
        # Get flow outline
        p = cs.collections[0].get_paths()
        c_x = np.array([])
        c_y = np.array([])
        for p_i in p:
            v = p_i.vertices
            c_x = np.append(c_x, v[:,0])
            c_y = np.append(c_y, v[:,1])
            
        try:
            np.max(c_x)
            frames_dat = np.append(frames_dat,frame)
        except: 
            print('No contour')
            continue
        
            
        ind = (c_x>0.001) & (c_y>c_y[np.argmax(c_x)])
        if np.mean(c_x[ind][1:] - c_x[ind][:-1])<0:
            c_x = c_x[::-1]
            c_y = c_y[::-1]
            ind = ind[::-1]
        for j, (x_0, x_1) in enumerate(zip(c_x, c_x[1:])):
            if (x_1<x_0) & (j>0) :
                c_x[j+1] = c_x[j]   
        for j, (x_0, x_1) in enumerate(zip(c_x, c_x[1:])):
            if (x_1<=x_0):
                ind[j] =  False
                
        surf_x = np.append(0,c_x[ind])
        surf_y = np.append(np.max(c_y),c_y[ind])
        if frame<int(dat['End frame'].values[0]):
            surf_x = np.append(surf_x,np.max(c_x)+0.005)
            surf_y = np.append(surf_y,0)
            
        spl = sciinterp.UnivariateSpline(surf_x, surf_y,s=9)
        spls[i] = spl

        left_x = np.min([np.min(c_x[ind]),2*dx+0.005])
        top_x = np.linspace(left_x+0.0025,np.max(c_x[ind])-0.0025,80)
        top_y = spl(top_x)
        top_y[:np.argmax(spl(top_x))] = np.max(spl(top_x))
        left_y = np.append(np.linspace(y_shift, np.max(spl(c_x)),10), np.max(spl(c_x)))
        right_y = np.append(np.linspace(y_shift, spl(np.max(c_x)),11), spl(np.max(c_x)))[::-1]

        if frame<int(dat['End frame'].values[0]):
            right_y = right_y[2:]
            
        ptsx = np.append(np.append(np.append(np.ones_like(left_y)*left_x,top_x),np.ones_like(right_y)*(np.max(c_x[ind])+0.005)),top_x[::-1])
        ptsy = np.append(np.append(np.append(left_y,top_y),right_y),np.ones_like(top_x)*y_shift)
        
        ptsx_dict[i] = ptsx
        ptsy_dict[i] = ptsy

        X,Y = np.meshgrid(x_scale,y_scale)
        path = mpltPath.Path(np.array([ptsx,ptsy]).transpose())
        mask = path.contains_points(np.array([X.flatten(),Y.flatten()]).transpose()).reshape(X.shape)
        masks[i,:,:] = mask

        # Plot velocities
        for k in np.arange(velocity):
            ds = dcmread(path_full + '/' + path_vel[k] + '/' + os.listdir(path_full + '/' + path_vel[k])[0] + '/' + os.listdir(
                path_full + '/' + path_vel[k] + '/' + os.listdir(path_full + '/' + path_vel[k] + '/')[0])[frame])
            vel = ds.pixel_array.astype(float)
            vel = -vel_stretch[k]*(vel-(2**(ds.BitsStored-1)))/(2**ds.BitsStored-1)
            if rot:
                vel = vel.transpose()
            vel = np.flipud(vel)
            #vel[~mask] = np.nan
            
            vels[i,:,:,k] = vel
            
        ts[i] = dt+t_shift

    return mags, masks, vels, frames, ts, ptsx_dict, ptsy_dict, spls, x_scale, y_scale, frames_dat
            
