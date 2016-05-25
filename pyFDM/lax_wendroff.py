# -*- coding: utf-8 -*-

from __future__ import division

import sys
import numpy as np
from scipy.interpolate import interp1d

from blood_flow import *
import utils


class LaxWendroff(object):
    """
    Class implementing Richtmyer's 2 step Lax-Wendroff method.
    """
    
    
    def __init__(self, nx, dx):
        self._nx = int(nx)
        self._dx = dx
        
        
    def solve(self, U0, U_in, U_out, t, T, cfl_condition, F, S, dt,
              **kwargs):
        # U0: previous timestep, U1 current timestep
        U1 = np.zeros((2,self.nx))
        
        # apply boundary conditions
        U1[:,0] = U_in
        U1[:,-1] = U_out
            
        for j in range(1,self.nx-1):
            U_prev = U0[:,j-1:j+2]
            if len(U_prev[0,:]) == 2:
                U_prev = U0[:,j-1:]
            F_prev = F(U_prev)
            S_prev = S(U_prev)
            U1[:,j] = self.lax_wendroff(U_prev, F_prev, S_prev, F, S, dt)
            if cfl_condition(U1[:,j]) == False:
                raise ValueError(
                    "CFL condition not fulfilled at time %e. Reduce \
time step size." % (t))
                sys.exit(1)
        return U1
                    
                    
    def lax_wendroff(self, U_prev, F_prev, S_prev, F, S, dt, **kwargs):
        # u_prev = [U[m-1], U[m], U[m+1]], F_prev, S_prev analogously
        U_np_mp = (U_prev[:,2]+U_prev[:,1])/2 + dt/2 *\
                    (-(F_prev[:,2]-F_prev[:,1])/self.dx +\
                    (S_prev[:,2]+S_prev[:,1])/2)
        U_np_mm = (U_prev[:,1]+U_prev[:,0])/2 + dt/2 *\
                    (-(F_prev[:,1]-F_prev[:,0])/self.dx +\
                    (S_prev[:,1]+S_prev[:,0])/2)
        F_np_mp = F(U_np_mp)
        F_np_mm = F(U_np_mm)
        S_np_mp = S(U_np_mp)
        S_np_mm = S(U_np_mm)
        
        U_np = U_prev[:,1] - dt/self.dx * (F_np_mp-F_np_mm) + dt/2 *\
                (S_np_mp+S_np_mm)
        return U_np
        
        
    @property   
    def nx(self):
        return self._nx
        
        
    @property   
    def dx(self):
        return self._dx