# -*- coding: utf-8 -*-


from __future__ import division

import numpy as np
import matplotlib.pylab as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import interpolate

plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = ['Arial']


def p3d_plot(fig_dims, time, P, L, pos, suffix='', plot_dir=''):
    fig = plt.figure(figsize=fig_dims)
    ax = fig.gca(projection='3d')
    nt = len(time)
    x = np.linspace(0, L, nt)
    Y, X = np.meshgrid(x, time)
    pt = P.shape[0]
    dz = int(pt/nt)
    Z = P[0:pt:dz,:]
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis,
                  linewidth=0, antialiased=False)
    ax.set_xlabel('t (s)')
    ax.set_ylabel('z (cm)')
    ax.set_zlabel('pressure (mmHg)')
    ax.set_xlim([min(time), max(time)])
    ax.set_ylim([min(x), max(x)])
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if plot_dir == '':
        fig.show()
    else:
        fname = "%s/%s/%s_p3d%d.png" % (plot_dir, suffix, suffix, pos)
        fig.savefig(fname, dpi=600, bbox_inches='tight')
    
    
def q3d_plot(fig_dims, time, U, L, pos, suffix='', plot_dir=''):
    fig = plt.figure(figsize=fig_dims)
    ax = fig.gca(projection='3d')
    nt = len(time)
    x = np.linspace(0, L, nt)
    Y, X = np.meshgrid(x, time)
    pt = U.shape[0]
    dz = int(pt/nt)
    Z = U[0:pt:dz,:]
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis,
                  linewidth=0, antialiased=False)
    ax.set_xlabel('t (s)')
    ax.set_ylabel('z (cm)')
    ax.set_zlabel('flow rate (cm^3/s)')
    ax.set_xlim([min(time), max(time)])
    ax.set_ylim([min(x), max(x)])
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if plot_dir == '':
        fig.show()
    else:
        fname = "%s/%s/%s_q3d%d.png" % (plot_dir, suffix, suffix, pos)
        fig.savefig(fname, dpi=600, bbox_inches='tight')
