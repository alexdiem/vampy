# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import sys


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
        self._i = 0
        
        
    def initial_conditions(self, u0, ntr):
        if not hasattr(self, '_x'):
            raise AttributeError('Artery not meshed. mesh(self, nx) has to be \
executed first')
        self.U = np.zeros((2, ntr, len(self.x)))
        self.U0 = np.zeros((2,len(self.x)))
        self.U0[0,:] = self.A0
        self.U0[1,:] = u0
        
        
    def mesh(self, nx):
        self._nx = nx
        self._x = np.linspace(0.0, self.L, nx)
        self._dx = self.x[1] - self.x[0]
        
        
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
        
        
    def solve(self, lw, U_in, U_out, t, dt, dtr, cfl_condition):
        # solve for current timestep
        U1 = lw.solve(self.U0, U_in, U_out, t, cfl_condition, self.F,
                        self.S, dt, dx=self.dx, wave_speed=self.wave_speed)
        np.copyto(self.U0, U1)
        if abs(t - dtr*self.i) < dt:
                np.copyto(self.U[:,self.i,:], U1)
                self.i += 1
        return U1
        
    
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
    def nx(self):
        return self._nx
        
        
    @property
    def U0(self):
        return self._U0
        
        
    @U0.setter
    def U0(self, value): 
        self._U0 = value
        
        
    @property
    def U(self):
        return self._U
        
        
    @U.setter
    def U(self, value): 
        self._U = value
        
        
    @property
    def i(self):
        return self._i
        
        
    @i.setter
    def i(self, value): 
        self._i = value