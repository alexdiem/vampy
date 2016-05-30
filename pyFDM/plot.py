# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import matplotlib.pylab as plt


def spatial_plot(fname, pltname, x, n, xlabel, ylabel):
    y = np.savetxt(fname, delimiter=',')
    plt.figure(figsize=(10,6))
    pos = int(len(y[0,:])/n)
    for i in range(n):
        d = int(i*pos)
        plt.plot(x, y[:,d], label="%d" % (d), lw=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(pltname, dpi=600, bbox_inches='tight')
    
    
def time_plot(fname, pltname, x, n, xlabel, ylabel):
    y = np.savetxt(fname, delimiter=',')
    plt.figure(figsize=(10,6))
    pos = int(len(y[:,0])/n)
    for i in range(n):
        d = int(i*pos)
        plt.plot(x, y[d,:], label="%d" % (d), lw=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(pltname, dpi=600, bbox_inches='tight')