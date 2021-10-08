# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:59:45 2021

@author: birnb
"""

import getfem as gf
from getfem import *
import numpy as np
from scipy import interpolate as sciinterp

def gfmodel(x_scale,y_scale,vels,mesh,spl,spl2d_u,spl2d_v,mask,flow_front,g,rho0,rhoh,K,tau_y,n,eta_max): 
    
    gfmesh = gf.Mesh('empty', 2)
    
    #for pt in mesh.points:
    #    gfmesh.add_point(pt)
    for e in mesh.elements:
        pts = np.array([mesh.points[e[0]], mesh.points[e[1]], mesh.points[e[2]]]).transpose()
        gfmesh.add_convex(gf.GeoTrans('GT_PK(2,1)'),pts)
        
    fb_right = gfmesh.outer_faces_with_direction([ 1., 0.], 0.1) # outward vector in [x,y], tolerance angle theta (rad)
    fb_left = gfmesh.outer_faces_with_direction([-1., 0.], 0.1)
    fb_top = gfmesh.outer_faces_with_direction([0.,  1.], 1)
    fb_bot = gfmesh.outer_faces_with_direction([0., -1.], 0.1)
    #fb_topleft = mesh.outer_faces_in_box([-0.01, L_y-L_y/(2*ny)], [L_x/nx*1.5, 1.01]) # top left
    
    bounds = [fb_right, fb_left, fb_top, fb_bot]
    RIGHT_BOUND=1; LEFT_BOUND=2; TOP_BOUND=3; BOTTOM_BOUND=4;
    
    for i,bound in enumerate(bounds): 
        gfmesh.set_region(i+1, bound)
        # Define variable fields and approximations
    # velocity
    mfu = gf.MeshFem(gfmesh, 2) # vector field
    mfu.set_fem(Fem('FEM_PK(2,2)')) # continuous piecewise linear
    #mfu.set_fem(Fem('FEM_PK_WITH_CUBIC_BUBBLE(2,1)')) # continuous piecewise quadratic
    # pressure
    mfp = gf.MeshFem(gfmesh, 1) # scalar field
    mfp.set_fem(Fem('FEM_PK(2,1)')) # continuous piecewise linear
    
    mim = gf.MeshIm(gfmesh, 4)
    
    # use model blocks to assemble problem - can alternatively be done manually
    md=gf.Model('real'); # real vs complex system
    md.add_fem_variable('u', mfu)
    md.add_fem_variable('p', mfp)
    
    D = mfp.basic_dof_nodes()
    ones = np.ones(D.shape[1])
    x = D[0,:]
    y = D[0,:]
    
    # add coefficients
    #md.add_initialized_data('mu', [mu])
    md.add_initialized_data('lambda', [0])
    md.add_initialized_data('g', [0,g])
    md.add_initialized_fem_data('rho',mfp,[(rhoh-rho0)/spl(x)*y + rho0])
    
    mu_exp = 'min(' + str(eta_max) + ', ' + str(tau_y) + '/t + ' + str(K) + '*pow(t, ' + str(n-1) + '))'
    gf.asm_define_function('mu', 1, mu_exp)
    
    md.add_nonlinear_term(mim, "mu(Norm(Grad_u - Grad_u.*Id(2)))*((Grad_u + Grad_u'):Grad_Test_u) + rho*(g.Test_u)")
    md.add_linear_incompressibility_brick(mim, 'u', 'p') 
    
    # Apply BCs
    u_dat_smooth = spl2d_u.ev(D[0,:],D[1,:])
    u_dat_smooth[u_dat_smooth>np.max(vels[:,:,0])] = np.max(vels[:,:,0])
    
    v_dat_smooth = spl2d_v.ev(D[0,:],D[1,:])
    v_dat_smooth[v_dat_smooth>np.max(vels[:,:,1])] = np.max(vels[:,:1])
    
    md.add_initialized_fem_data('leftdata', mfp, [u_dat_smooth,v_dat_smooth])
    md.add_initialized_fem_data('botdata', mfp, [0*ones,0*ones])
    
    md.add_Dirichlet_condition_with_multipliers(mim, 'u', 0, LEFT_BOUND, dataname='leftdata')
    md.add_Dirichlet_condition_with_multipliers(mim, 'u', 0, BOTTOM_BOUND, dataname='botdata')
    
    md.add_initialized_fem_data('topdata', mfp, [[0*ones, 0*ones],
                                                 [0*ones, 0*ones]])
    md.add_normal_source_term_brick(mim, 'u', 'topdata', TOP_BOUND);
    
    if flow_front: 
        md.add_initialized_fem_data('rightdata', mfp, [[0*ones, 0*ones],
                                                 [0*ones, 0*ones]])
        md.add_normal_source_term_brick(mim, 'u', 'rightdata', RIGHT_BOUND);
    else: 
        md.add_initialized_fem_data('rightdata',mfp, [u_dat_smooth,v_dat_smooth])
        md.add_Dirichlet_condition_with_multipliers(mim, 'u', 0, RIGHT_BOUND, dataname='rightdata')
        
    # solve all in one
    try: 
        md.solve('max_res', 1E-9, 'max_iter', 100, 'very_noisy'); # Newton iteration for nonlinearity
    except:
        print("didn't solve")
    # Retrieve variables
    u = md.variable('u')
    P = md.variable('p')
    
    X, Y = np.meshgrid(x_scale,y_scale)
    [ui,vi] = compute(mfu,u,'interpolate on',[X.flatten(),Y.flatten()])
    u_sol= ui.copy().reshape(X.shape)
    u_sol[mask==0] = np.nan

    v_sol= vi.copy().reshape(X.shape)
    v_sol[mask==0] = np.nan
    
    pi = compute(mfp,P,'interpolate on',[X.flatten(),Y.flatten()])
    
    P_sol= pi.copy().reshape(X.shape)
    P_sol[mask==0] = np.nan

    return u_sol, v_sol, P_sol
    