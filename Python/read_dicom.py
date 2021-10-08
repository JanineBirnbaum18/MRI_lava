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

def read_dicom(SESSION,path_to_download, path_mag, path_vel):
    # retrive metadata
    df = pd.read_excel('C:/Users/birnb/Documents/Columbia Research/MRI/Data/Experiments list.xlsx')
    dat = df.loc[df.Session == SESSION].loc[df.String==path_mag]
    wd = os.getcwd()
    
    rot = 'Along' in dat.Slice.values
    velocity = 0
    vel_stretch = []
    vel_dir = []
    vel_titles = {'Vel_along':'Along-channel', 'Vel_across':'Across-channel', 'Vel_vert':'Vertical'}
    for col in ['Vel_along','Vel_across','Vel_vert']:
        if not np.isnan(dat[col].values[0]):
            velocity += 1 #number of velocity encodings
            vel_stretch = np.append(vel_stretch,dat[col].values[0]/100) # convert to m/s
            vel_dir = np.append(vel_dir,vel_titles[col])
    gateframe = int(dat['Start frame'].values[0]) + 1 # frame of gate removal (used to calculate time from release)

    # Image processing parameters
    thresh = 0.004 # minimum magnitude of signal intensity
    windowSize = 3 # size for initial smoothing kernel
    sigma_x = 10 # gaussian smoothing in x-direction
    sigma_y = 0.05 # gaussian smoothing in y-direction
    thresh2 = 130 # rethreshold after smoothing (0,255)
    
    y_shift = dat['y_shift'].values[0]/100 # position of MRI image bottom edge with respect to channel bottom (cm), convert to m
    t_shift = dat['t_shift'].values[0] # time before gateframe when gate is actually removed (s)

    # temporarily unzip files to local directory
    try: 
        os.remove('temp')
    except: 
        print('Nothing to delete')

    try: 
        os.mkdir('temp')
    except: 
        print('Already exists')
    comp_mag = path_to_download + '/' + path_mag + '/Files/' + [f.split('.zip')[0] for f in os.listdir(path_to_download + '/' + path_mag + '/Files/') if ('.zip' in f)][0]

    with ZipFile(comp_mag + '.zip', 'r') as zf:
        zf.extractall(wd + '/temp/mag')

    for i in np.arange(velocity):
        comp_vel = path_to_download + '/' + path_vel[i] + '/Files/' + [f.split('.zip')[0] for f in os.listdir(path_to_download + '/' + path_vel[i] + '/Files/') if ('.zip' in f)][0]

        with ZipFile(comp_vel + '.zip', 'r') as zf:
            zf.extractall(wd + '/temp/vel' + str(i))
    frames_total = len(os.listdir('temp/mag/' + os.listdir('temp/mag')[0]))
    
    frames = np.arange(int(dat['Start frame'].values[0])+1, frames_total)
    
    # visualize data
    ds = dcmread('temp/mag/' + os.listdir('temp/mag')[0] + '/' + os.listdir('temp/mag/' + os.listdir('temp/mag')[0])[gateframe])
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

    for i,frame in enumerate(frames):
        ds = dcmread('temp/mag/' + os.listdir('temp/mag')[0] + '/' + os.listdir('temp/mag/' + os.listdir('temp/mag')[0])[frame])
        dt = float(ds.AcquisitionTime) - float(gatetime)
        mag = ds.pixel_array.astype(float)
        mag = mag/(2**ds.BitsStored)
        if rot:
            mag = mag.transpose()
        mag = np.flipud(mag)
    
        dx = 0.4/mag.shape[1]
        x_scale = np.arange(mag.shape[1])*dx
        y_scale = np.arange(mag.shape[0])*dx
    
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
        cv2.floodFill(im_floodfill, None, (5, 10), 255) # select interior point
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_out = ~im_out & (~im_floodfill_inv)
    
        # Smooth outline
    
        im_blur = cv2.blur(im_out,(windowSize,windowSize), borderType=0)
    
        if dt>=0:
            im_blur = cv2.GaussianBlur(im_blur,(int(2*np.ceil(2*sigma_x)+1),int(2*np.ceil(2*sigma_y)+1)),sigma_x,sigma_y,
                              borderType=0) 

        # Rethreshold
        mask = im_blur>thresh2
        mask[y_scale<y_shift,:] = False
        
        # Get flow outline
        cs = ax.contour(x_scale, y_scale, mask, levels=1)
        p = cs.collections[0].get_paths()
        c_x = []
        c_y = []
        for p_i in p:
             v = p_i.vertices
             c_x = np.append(c_x, v[:,0])
             c_y = np.append(c_y, v[:,1])
        
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

        top_x = np.linspace(np.min(c_x[ind]+0.0025),np.max(c_x[ind])-0.0025,80)
        top_y = spl(top_x)
        top_y[:np.argmax(spl(top_x))] = np.max(spl(top_x))
        left_y = np.append(np.linspace(y_shift, np.max(spl(c_x)),10), np.max(spl(c_x)))
        right_y = np.append(np.linspace(y_shift, spl(np.max(c_x)),11), spl(np.max(c_x)))[::-1]

        if frame<int(dat['End frame'].values[0]):
            right_y = right_y[2:]

        ptsx = np.append(np.append(np.append(np.ones_like(left_y)*np.min(c_x[ind]),top_x),np.ones_like(right_y)*(np.max(c_x[ind])+0.005)),top_x[::-1])
        ptsy = np.append(np.append(np.append(left_y,top_y),right_y),np.ones_like(top_x)*y_shift)
        
        ptsx_dict[i] = ptsx
        ptsy_dict[i] = ptsy

        X,Y = np.meshgrid(x_scale,y_scale)
        path = mpltPath.Path(np.array([ptsx,ptsy]).transpose())
        mask = path.contains_points(np.array([X.flatten(),Y.flatten()]).transpose()).reshape(X.shape)
        masks[i,:,:] = mask
        
    
        # Plot velocities
        for k in np.arange(velocity):
            ds = dcmread('temp/vel' + str(k) + '/' + os.listdir('temp/vel' + str(k))[0] + '/' + os.listdir(
            'temp/vel' + str(k) + '/'+ os.listdir('temp/vel' + str(k))[0])[frame])
            vel = ds.pixel_array.astype(float)
            vel = -vel_stretch[k]*(vel-(2**(ds.BitsStored-1)))/(2**ds.BitsStored-1)
            if rot:
                vel = vel.transpose()
            vel = np.flipud(vel)
            vels[i,:,:,k] = vel
        
        ts[i] = dt+t_shift
    return mags, masks, vels, frames, ts, ptsx_dict, ptsy_dict, spls
            
