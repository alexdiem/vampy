# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import matplotlib.pylab as plt
import sys

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import interpolate

from vampy import vamplot
from vampy import utils

plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = ['Arial']


def spatial_plots(fig_dims, suffix, plot_dir, n):
    rc, qc, Re = self.nondim
    L = self.L * rc
    nt = len(self.U[0,:,0])   
    x = np.linspace(0, L, self.nx)
    skip = int(nt/n)+1
    u = ['a', 'q', 'p']
    l = ['cm^2', 'cm^3/s', 'mmHg']
    positions = range(0,nt-1,skip)
    for i in range(2):
        y = self.U[i,positions,:]
        fname = "%s/%s_%s%d_spatial.png" % (plot_dir, suffix, u[i], self.pos)
        Artery.plot(fig_dims, suffix, plot_dir, x, y, positions, "cm", l[i],
                    fname)
    y = self.P[positions,:] # convert to mmHg    
    fname = "%s/%s_%s%d_spatial.png" % (plot_dir, suffix, u[2], self.pos)
    Artery.plot(fig_dims, suffix, plot_dir, x, y, positions, "cm", l[2],
                    fname)
            
            
def time_plots(fig_dims, suffix, plot_dir, n, time):
    rc, qc, Re = self.nondim
    time = time * rc**3 / qc
    skip = int(self.nx/n)+1
    u = ['a', 'q', 'p']
    l = ['cm^2', 'cm^3/s', 'mmHg']
    positions = range(0,self.nx-1,skip)
    for i in range(2):
        y = self.U[i,:,positions]
        fname = "%s/%s_%s%d_time.png" % (plot_dir, suffix, u[i], self.pos)
        Artery.plot(fig_dims, suffix, plot_dir, time, y, positions, "t", l[i],
                    fname)
    y = np.transpose(self.P[:,positions])   
    fname = "%s/%s_%s%d_time.png" % (plot_dir, suffix, u[2], self.pos)
    Artery.plot(fig_dims, suffix, plot_dir, time, y, positions, "t", l[2],
                    fname)
                        
                        
def pq_plot(fig_dims, suffix, plot_dir):
    L = len(self.P[0,:])-1
    positions = [0, int(L/4), int(L/2), int(3*L/4), L]
    y = np.transpose(self.P[:,positions])
    x = self.U[1,:,positions]
    fname = "%s/%s_%s%d_pq.png" % (plot_dir, suffix, 'pq', self.pos)
    plt.figure(fig_dims)
    labels = ['0', 'L/4', 'L/2', '3L/4', 'L']
    for i in range(len(y[:,0])):
        plt.plot(x[i,:], y[i,:], lw=1, color='k')
    plt.xlabel('flux (cm^3/s)')
    plt.ylabel('pressure (mmHg)')
    plt.savefig(fname, dpi=600, bbox_inches='tight')
            
            
def plot(fig_dims, suffix, plot_dir, x, y, labels, xlabel, ylabel, fname):
    colours = ['#377eb8', '#4daf4a', '#984ea3', '#d95f02']
    plt.figure(fig_dims)
    s = y.shape
    n = min(s)
    for i in range(n):
        plt.plot(x, y[i,:], label="%d" % (labels[i]), lw=2, color=colours[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([min(x), max(x)])
    plt.legend()
    plt.savefig(fname, dpi=600, bbox_inches='tight')


def main(param):
    # read config file
    f, a, s = utils.read_config(param) 
    
    data_dir = f['data_dir']
    plot_dir = f['plot_dir']
    suffix = f['run_id']
    T = s['T']
    tc = s['tc']
    tf = T*tc
    
    pos = 0
    if type(a['Ru']) is float:
        L = a['Ru']*a['lam']
    else:
        L = a['Ru'][pos]*a['lam'][pos]
    
    P = np.loadtxt("%s/%s/p%d_%s.csv" % (data_dir, suffix, pos, suffix), delimiter=',')
    U = np.loadtxt("%s/%s/u%d_%s.csv" % (data_dir, suffix, pos, suffix), delimiter=',')
    t = np.linspace(tf-T, tf, P.shape[1])
    x = np.linspace(0,L,P.shape[0])
    f = interpolate.interp2d(t, x, P, kind='linear')
    g = interpolate.interp2d(t, x, U, kind='linear')
    
    x = np.linspace(0, L, len(t))
    P = f(t, x)
    U = g(t, x)
    
    WIDTH = 510  # the number latex spits out
    FACTOR = 1.0  # the fraction of the width you'd like the figure to occupy
    fig_width_pt  = WIDTH * FACTOR
    inches_per_pt = 1.0 / 72.27
    golden_ratio  = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
    fig_width_in  = fig_width_pt * inches_per_pt  # figure width in inches
    fig_height_in = fig_width_in * golden_ratio   # figure height in inches
    fig_dims    = [fig_width_in, fig_height_in] # fig dims as a list
    
    vamplot.p3d_plot(fig_dims, suffix, plot_dir, t, P, L, pos)
    vamplot.q3d_plot(fig_dims, suffix, plot_dir, t, U, L, pos)
    
    
if __name__ == "__main__":
    script, param = sys.argv
    main(param)
