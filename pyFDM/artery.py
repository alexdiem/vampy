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
        
        
    def __init__(self, R, lam, k, sigma, rho, mu, inlet_bc, outlet_bc, ntr=100):
        self._R = R
        self._A0 = np.pi*R**2
        self._L = R*lam
        Ehr = k[0] * np.exp(k[1]*R) + k[2]
        self._beta = np.sqrt(np.pi)*Ehr*R/(self.A0 * (1-sigma)**2)
        self._rho = rho
        self._mu = mu
        self._t = 0.0
        self._inlet_bc = inlet_bc
        self._outlet_bc = outlet_bc
        self._ntr = ntr
        
        
    def initial_conditions(self):
        self.U0 = np.zeros((2,len(self.x)))
        self.U0[0,:] = self.A0
        self.U0[1,:] = 0.0
        
        
    def tf(self):
        return self._dt * self._nt
        
        
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
    
    
    def F(U, args):
        a, u = U
        p = p(a)
        return np.array([a*u, np.power(u,2) + p/self.rho])
        
    
    def S(U, args):
        a, u = U
        return np.array([u*0, -8*np.pi*self.mu/self.rho * u/a])
        
        
    def solve(self, u0, u_in, p_out, T):
        nx = len(self.x)
        cfl_args = (self.beta, self.rho, self.dt, self.dx)
        in_args = self.set_in_args()
        F_args = ()
        S_args = ()
        self.initial_conditions()
        lw = LaxWendroff(nx, self.dx)
        i = 0
        self._U = np.zeros((2, self.ntr, nx))
        np.copyto(self.U[:,0,:], self.U0)
        
        while self.t < self.tf:
            # inlet boundary condition
            if self.T > 0:
                in_t = utils.periodic(self.t, self.T)
            U_in = self.inlet_bc(self.U0, self.t, in_args)
            # outlet boundary condition
            U_out = self.outlet_bc(U0, t, out_args)
            # solve for current timestep
            U1 = lw.solve(U0, U_in, U_out, t, T, cfl_condition, F, S, 
                          cfl_args, F_args, S_args)
            print U1
            sys.exit()
            if abs(t - self.dtr*i) < self.dt:
                np.copyto(self.U[:,i,:], U1)
                i += 1
            self.timestep()
            np.copyto(U0, U1)
        
        
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
    def inlet_bc(self):
        return self._inlet_bc
        
    
    @property
    def outlet_bc(self):
        return self._outlet_bc
        
        
    @property
    def ntr(self):
        return self._ntr
        
        
    @property
    def dtr(self):
        return self._dtr