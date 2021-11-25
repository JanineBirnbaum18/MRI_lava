# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:13:22 2021

@author: birnb
"""
import numpy as np
from meshpy.triangle import MeshInfo, build
from matplotlib import pyplot as plt

def build_mesh(ptsx,ptsy):
    mesh_info = MeshInfo()
    mesh_info.set_points(list(zip(ptsx,ptsy)))
    mesh_info.set_facets(list(zip(np.arange(len(ptsx)),np.roll(np.arange(len(ptsy)),-1))))
    mesh = build(mesh_info, max_volume=0.00001)
    
    # fig, ax = plt.subplots(figsize=(15,10))

    # for i, p in enumerate(mesh.points):
    #     ax.scatter(p[0],p[1],50,'k',zorder=2)

    # for i, t in enumerate(mesh.elements):
    #     pts = np.array([mesh.points[t[0]], mesh.points[t[1]], mesh.points[t[2]], mesh.points[t[0]]])
    #     plt.plot(pts[:,0], pts[:,1], 'r',zorder=1);
    # ax.set_aspect('equal')
    return mesh