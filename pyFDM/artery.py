# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import sys

from lax_wendroff import LaxWendroff
import blood_flow
import utils

class Artery(object):
    """
    Class representing an artery.
    """
        
        
    def __init__(self, pos, R, lam, sigma, rho, mu, **kwargs):
        self._pos = pos
        self._R = R
        self._A0 = np.pi*R**2
        self._L = R*lam
        if 'k' in kwargs.keys():
            k = kwargs['k']
            Ehr = k[0] * np.exp(k[1]*R) + k[2]
            self._beta = np.sqrt(np.pi)*Ehr*R/(self.A0 * (1-sigma)**2)
        elif 'beta' in kwargs.keys():
            self._beta = kwargs['beta']  
        else:
            raise ValueError('No elasticity parameter specified')
        self._rho = rho
        self._mu = mu
        self._ntr = kwargs['ntr']
        
        
    def initial_conditions(self, u0):
        if not hasattr(self, '_x'):
            raise AttributeError('Artery not meshed. mesh(self, nx) has to be \
executed first')
        self.U0 = np.zeros((2,len(self.x)))
        self.U0[0,:] = self.A0
        self.U0[1,:] = u0
        self.U = np.zeros((2, self.ntr, self.nx))
        
        
    def mesh(self, nx):
        self._x = np.linspace(0.0, self.L, nx)
        self._dx = self.x[1] - self.x[0]
        
        
    def set_time(self, nt, dt, T=0.0):
        self._nt = nt
        self._dt = dt
        self._dtr = nt*dt/(self.ntr-1)
        self._T = T
        
        
    def get_uprev(self):
        pass
    
    
    def p(self, a):
        return self.beta * (np.sqrt(a)-np.sqrt(self.A0))
        
        
    def wave_speed(self, a):
        return np.sqrt(self.beta*np.sqrt(a)/(2*self.rho))
    
    
    def F(self, U, **kwargs):
        a, u = U
        p = self.p(a)
        f = np.array([a*u, np.power(u,2) + p/self.rho])
        return np.array([a*u, np.power(u,2) + p/self.rho])
        
    
    def S(self, U, **kwargs):
        a, u = U
        return np.array([u*0, -8*np.pi*self.mu/self.rho * u/a])
        
        
    def solve(self, U_in, U_out, u0, p_out, T):
        # solve for current timestep
        U1 = lw.solve(self.U0, U_in, U_out, self.t, self.T,
                      self.cfl_condition, self.F, self.S, self.dt)
        return self.U[0,:,:], self.U[1,:,:]
        
    
    @property
    def pos(self):
        return self._pos

    
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
    def init_cond(self):
        return self._init_cond
        
        
    @property
    def x(self):
        return self._x
        
        
    @property
    def dx(self):
        return self._dx
        
        
    @property
    def U(self):
        return self._U
        
        
    @U.setter
    def U(self, value): 
        self._U = value
        
        
    @property
    def U0(self):
        return self._U0
        
        
    @U0.setter
    def U0(self, value): 
        self._U0 = value