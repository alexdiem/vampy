# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from lax_wendroff import LaxWendroff

from blood_flow import *

class Artery(object):
    """
    Class representing an artery.
    """
    
    
    @property
    def R(self):
        return self._R
        
        
    @property    
    def A0(self):
        return self._A0
        
        
    @property
    def L(self):
        return self._L
        
        
    @property
    def beta(self):
        return self._beta
        
        
    @property
    def rho(self):
        return self._rho
        
        
    @property
    def mu(self):
        return self._mu
        
        
    @property
    def x(self):
        return self._x
        
        
    @property
    def dx(self):
        return self._dx
        
        
    @property
    def nt(self):
        return self._nt
        
        
    @property
    def dt(self):
        return self._dt
    
    
    def __init__(self, R, lam, k, sigma, rho, mu):
        self._R = R
        self._A0 = np.pi*R**2
        self._L = R*lam
        Ehr = k[0] * np.exp(k[1]*R) + k[2]
        self._beta = np.sqrt(np.pi)*Ehr*R/(self.A0 * (1-sigma)**2)
        self._rho = rho
        self._mu = mu
        
        
    def mesh(self, nx):
        self._x = np.linspace(0.0, self.L, nx)
        self._dx = self.x[1] - self.x[0]
        
        
    def set_time(self, nt, dt):
        self._nt = nt
        self._dt = dt
        
        
    def solve(self, u0, u_in, p_out):
        nx = len(self.x)
        in_args = (u_in, self.beta, self.rho, self.x, self.dt)     
        a_out = (p_out/self.beta + np.sqrt(self.A0))**2
        out_args = (a_out, self.beta, self.rho, self.x, self.dt)
        cfl_args = (self.beta, self.rho, self.dt, self.dx)
        F_args = (self.beta, self.A0, self.rho)
        S_args = (self.mu, self.rho)
        lw = LaxWendroff(self.nt, nx, self.dt, self.dx, (self.A0, u0))
        A, U = lw.solve(inlet_bc, outlet_bc, cfl_condition, F, S, in_args,
                            out_args, cfl_args, F_args, S_args)
        return A, U
            