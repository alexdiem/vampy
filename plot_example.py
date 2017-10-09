# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import matplotlib.pylab as plt
import sys

from scipy import interpolate

from os import makedirs
from os.path import exists

from vampy import vamplot
from vampy import utils

plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = ['Arial']


def main(param):
    # read config file
    f, a, s = utils.read_config(param) 
    
    data_dir = f['data_dir']
    plot_dir = f['plot_dir']
    suffix = f['run_id']
    T = s['T']
    tc = s['tc']
    tf = T*tc
    
    if not exists("%s/%s" % (plot_dir, suffix)):
        makedirs("%s/%s" % (plot_dir, suffix))
    
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
    
    vamplot.p3d_plot(fig_dims, t, P, L, pos, suffix, plot_dir)
    vamplot.q3d_plot(fig_dims, t, U, L, pos, suffix, plot_dir)
    
    
if __name__ == "__main__":
    script, param = sys.argv
    main(param)
