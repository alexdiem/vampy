# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from scipy.interpolate import interp1d


class LaxWendroff(object):
    """
    Class implementing Richtmyer's 2 step Lax-Wendroff method.
    """
    
    
    @property   
    def nt(self):
        return self._nt
        
        
    @property   
    def nx(self):
        return self._nx
        
        
    @property   
    def dt(self):
        return self._dt
        
        
    @property   
    def dx(self):
        return self._dx
        
        
    @property   
    def init_cond(self):
        return self._init_cond
        
        
    @property   
    def U(self):
        return self._U
    
    
    def __init__(self, nt, nx, dt, dx, init_cond):
        self._nt = nt
        self._nx = nx
        self._dt = dt
        self._dx = dx
        self._init_cond = init_cond
        
        
    def initial_conditions(self):
        N = len(init_cond)
        # check that we're dealing with 2 equations
        if N != 2:
            raise ValueError("%d initial conditions supplied, 2 expected."\
                        % (N))
        U = np.zeros((N, nt, nx))
        for i in range(N):
            U[i,0,:] = init_cond[i]
        return U
        
        
    def solve(self, inlet_bc, outlet_bc, cfl_condition, F, S, in_args,
                  out_args, cfl_args, F_args, S_args):
        for i in range(1,nt):
            U_prev = U[:,i-1,:]
            # inlet boundary condition
            U[:,i,0] = inlet_bc(U_prev, in_args)
            # outlet boundary condition
            U[:,i,-1] = outlet_bc(U_prev, out_args)
            
            for j in range(1,nx-1):
                U_prev = U[:,i-1,j-1:j+2]
                if len(U_prev[0,:] == 2):
                    U_prev = U[:,i-1,j-1:]
                F_prev = F(U_prev, F_args)
                S_prev = S(U_prev, S_args)
                U[:,i,j] = lax_wendroff(self, U_prev, F_prev, S_prev, F_args,
                            S_args)
                
                if cfl_condition(U[:,i,j], cfl_args) == False:
                    raise ValueError("CFL condition not fulfilled\nReduce time step size.")
                    sys.exit(1)   
                    
                    
    def lax_wendroff(self, U_prev, F_prev, S_prev, F_args, S_args):
        # u_prev = [U[m-1], U[m], U[m+1]], F_prev, S_prev analogously
        U_np_mp = (U_prev[:,2]+U_prev[:,1])/2 + self.dt/2 *\
                    (-(F_prev[:,2]-F_prev[:,1])/self.dx +\
                    (S_prev[:,2]+S_prev[:,1])/2)
        U_np_mm = (U_prev[:,1]+U_prev[:,0])/2 + self.dt/2 *\
                    (-(F_prev[:,1]-F_prev[:,0])/self.dx +\
                    (S_prev[:,1]+S_prev[:,0])/2)
        F_np_mp = F(U_np_mp, F_args)
        F_np_mm = F(U_np_mm, F_args)
        S_np_mp = S(U_np_mp, S_args)
        S_np_mm = S(U_np_mm, S_args)
        
        U_np = U_prev[:,1] - self.dt/self.dx * (F_np_mp-F_np_mm) + self.dt/2 *\
                (S_np_mp+S_np_mm)
        return U_np
        
        