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
        
    
    def solve(self, U0, U_in, U_out, t, F, S, dt):
        # U0: previous timestep, U1 current timestep
        U1 = np.zeros((2,self.nx))
        
        # apply boundary conditions
        U1[:,0] = U_in
        U1[:,-1] = U_out
        F0 = F(U0)
        S0 = S(U0)
         
        # calculate half step
        U_np_mp = (U0[:,2:]+U0[:,1:-1])/2 + dt/2 * (-(F(U0[:,2:])-F(U0[:,1:-1]))/\
                self.dx + (S(U0[:,2:])+S(U0[:,1:-1]))/2)  
        U_np_mm = (U0[:,1:-1]+U0[:,:-2])/2 + dt/2 * (-(F(U0[:,1:-1])-F(U0[:,:-2]))/\
                self.dx + (S(U0[:,1:-1])+S(U0[:,:-2]))/2) 
        U1[:,1:-1] = U0[:,1:-1] - dt/self.dx * (F(U_np_mp)-F(U_np_mm)) + dt/2 *\
                (S(U_np_mp)+S(U_np_mm))
        return U1
        
        
    @property   
    def nx(self):
        return self._nx
        
        
    @property   
    def dx(self):
        return self._dx