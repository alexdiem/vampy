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
        
        
    def __init__(self, R, lam, sigma, rho, mu, **kwargs):
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
        self._t = 0.0
        self._ntr = kwargs['ntr']
        
        
    def initial_conditions(self, u0):
        if not hasattr(self, '_x'):
            raise AttributeError('Artery not meshed. mesh(self, nx) has to be \
executed first')
        self.U0 = np.zeros((2,len(self.x)))
        self.U0[0,:] = self.A0
        self.U0[1,:] = u0
        
        
    def tf(self):
        return self.dt * self.nt
        
        
    def timestep(self):
        self._t += self.dt
        
        
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
            
    
    def inlet_bc(self, u_prev, u_in):
        c_prev = self.wave_speed(u_prev[0,0])
        w_prev = u_prev[1,0:2] - 4*c_prev
        lam_2 = u_prev[1,0] - c_prev
        x_0 = self.x[0] - lam_2 * self.dt
        w_2 = utils.extrapolate(x_0, self.x[0:2], w_prev)
        a_in = (u_in - w_2)**4 / 64 * (self.rho/self.beta)**2
        return np.array([a_in, u_in])
     
    
    def outlet_bc(self, u_prev, a_out):
        c_prev = self.wave_speed(u_prev[0,-1])
        w_prev = u_prev[1,-2:] + 4*c_prev
        lam_1 = u_prev[1,-1] + c_prev
        x_0 = self.x[-1] - lam_1 * self.dt
        w_1 = utils.extrapolate(x_0, self.x[-2:], w_prev)
        u_out = w_1 - 4*a_out**(1/4) * np.sqrt(self.beta/(2*self.rho))
        return np.array([a_out, u_out])
        
    
    def bifurcation_bc(self, p, d1, d2):
        pass
    
    
    def cfl_condition(self, u, **kwargs):
        c = self.wave_speed(u[0])
        v = (u[1] + c, u[1] - c)
        left = self.dt/self.dx
        right = np.power(np.absolute(v), -1)
        return False if (left > right).any() else True
    
    
    def F(self, U, **kwargs):
        a, u = U
        p = self.p(a)
        f = np.array([a*u, np.power(u,2) + p/self.rho])
        return np.array([a*u, np.power(u,2) + p/self.rho])
        
    
    def S(self, U, **kwargs):
        a, u = U
        return np.array([u*0, -8*np.pi*self.mu/self.rho * u/a])
        
        
    def solve(self, u0, u_in, p_out, T):
        nx = len(self.x)
        lw = LaxWendroff(nx, self.dx)
        i = 0
        self._U = np.zeros((2, self.ntr, nx))
        np.copyto(self.U[:,0,:], self.U0)
        
        while self.t < self.tf():
            # inlet boundary condition
            if self.T > 0:
                in_t = utils.periodic(self.t, self.T)
            U_in = self.inlet_bc(self.U0, u_in(in_t))
            # outlet boundary condition
            U_out = self.outlet_bc(self.U0, self.A0)
            # solve for current timestep
            U1 = lw.solve(self.U0, U_in, U_out, self.t, self.T,
                          self.cfl_condition, self.F, self.S, self.dt)
            
            
            if abs(self.t - self.dtr*i) < self.dt:
                np.copyto(self.U[:,i,:], U1)
                i += 1
            self.timestep()
            np.copyto(self.U0, U1)
        return self.U[0,:,:], self.U[1,:,:]
        
        
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
    def nt(self):
        return self._nt
        
        
    @property
    def dt(self):
        return self._dt
        
        
    @property
    def T(self):
        return self._T
        
        
    @property
    def t(self):
        return self._t
        
        
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
        
        
    @property
    def ntr(self):
        return self._ntr
        
        
    @property
    def dtr(self):
        return self._dtr