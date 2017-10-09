# cython: linetrace=True
# -*- coding: utf-8 -*-
"""
Example file for simulating a single artery using VaMpy.

.. moduleauthor:: Alexandra K. Diem <alexandra.diem@gmail.com>

"""

from __future__ import division

import sys
import numpy as np
from scipy.interpolate import interp1d

from vampy import *
from vampy.artery_network import ArteryNetwork


def inlet(qc, rc, data_dir, f_inlet):
    """
    Function describing the inlet boundary condition. Returns a function.
    """
    Q = np.loadtxt("./%s/%s" % (data_dir, f_inlet), delimiter=',')
    t = [(elem) * qc / rc**3 for elem in Q[:,0]]
    q = [elem / qc for elem in Q[:,1]]
    return interp1d(t, q, kind='linear', bounds_error=False, fill_value=q[0])


def mca_inlet(data_dir, f_inlet, Ru, qc, T):
    Q = np.loadtxt("./%s/%s" % (data_dir, f_inlet), delimiter=',')
    t = [(elem/Q[-1,0]) * T for elem in Q[:,0]]
    q = [elem*100*Ru**2*np.pi/qc for elem in Q[:,1]]
    return interp1d(t, q, kind='linear', bounds_error=False, fill_value=q[0])


def main(param):
    """
    Example main.py for running a VaMpy model of a bifurcation.
    """
    
    # read config file
    f, a, s = utils.read_config(param) 
    
    # nondimensionalisation parameters
    rc = a['rc'] 
    qc = a['qc']  
    rho = a['rho']
    nu = a['nu']
    Re = qc/(nu*rc)
    nondim = [rc, qc, Re]
    
    # assign parameters
    run_id = f['run_id'] # run ID
    f_inlet = f['inlet'] # inlet file
    data_dir = f['data_dir'] # data directory
    T = s['T'] * qc / rc**3 # time of one cycle
    tc = s['tc'] # number of cycles to simulate
    tf = T * tc # total simulation time
    dt = s['dt'] * qc / rc**3 # time step size
    ntr = 50 # number of time steps to be stored
    dx = s['dx'] / rc # spatial step size
    Ru = a['Rd'] / rc # artery radius upstream
    Rd = a['Rd'] / rc # artery radius downstream
    nu = nu*rc/qc # viscosity
    kc = rho*qc**2/rc**4
    k = (a['k1']/kc, a['k2']*rc, a['k3']/kc) # elasticity model parameters (Eh/r)
    out_args = [a['R1']*rc**4/(qc*rho), a['R2']*rc**4/(qc*rho), 
            a['Ct']*rho*qc**2/rc**7] # Windkessel parameters
    out_bc = '3wk'
    p0 = (0 * 1333.22365) * rc**4/(rho*qc**2) # zero transmural pressure
    
    # inlet boundary condition
    t = np.linspace(0,T,200)
    #q_in = mca_inlet(data_dir, f_inlet, Ru[0], qc, T)
    q_in = inlet(qc, rc, data_dir, f_inlet)

    #import matplotlib.pylab as plt
    #plt.plot(t, q_in(t))
    #plt.xlabel("time (s)")
    #plt.ylabel("flow rate (cm^3/s)")
    #plt.show()
    #sys.exit()

    # initialise artery network object
    an = ArteryNetwork(Ru, Rd, a['lam'], k, rho, nu, p0, a['depth'], ntr, Re)
    an.mesh(dx)
    an.set_time(dt, T, tc)
    u0 = q_in(0.0) # initial condition for flux
    an.initial_conditions(0.0)
    
    # run solver
    an.solve(q_in, out_bc, out_args)
    
    # redimensionalise
    an.redimensionalise(rc, qc)
    
    # save results
    an.dump_results(run_id, f['data_dir'])
        
    
if __name__ == "__main__":
    script, param = sys.argv
    main(param)
