# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import sys
import matplotlib.pylab as plt


class Artery(object):
    """
    Class representing an artery.
    """
        
        
    def __init__(self, pos, R, lam, sigma, rho, mu, **kwargs):
        self._pos = pos
        self._R = R
        self._A0 = np.pi*R**2
        self._L = R*lam
        self._k = kwargs['k']
        self._Ehr = self.k[0] * np.exp(self.k[1]*R) + self.k[2]
        self._rho = rho
        self._mu = mu
        
        
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
        return 4*self.Ehr * (1 - np.sqrt(a *self.A0))/3
        
        
    def wave_speed(self, a):
        return np.sqrt(a/self.rho * 4*self.Ehr/3 *\
                self.A0/(2*np.sqrt(self.A0*a)))
    
    
    #def F(self, U, **kwargs):
    #    a, u = U
    #    p = self.p(a)
    #    return np.array([a*u, np.power(u,2) + p/self.rho])
        
    
    #def S(self, U, **kwargs):
    #    a, u = U
    #    return np.array([u*0, -8*np.pi*self.mu/self.rho * u/a])
        
        
    def F(self, U, **kwargs):
        a, q = U
        f = 4/3 * self.Ehr
        return np.array([q, np.power(q,2)/a + f/self.rho * np.sqrt(self.A0*a)])
        
        
    def S(self, U, **kwargs):
        a, q = U
        delta = np.sqrt(self.mu*kwargs['T']/(self.rho*2*np.pi))
        f = 4/3 * self.Ehr
        df = 4/3 * self.k[0] * self.k[1] * np.exp(self.k[1] * self.R)
        try:
            xgrad = np.gradient(self.R)
        except ValueError:
            xgrad = 0
        return np.array([q*0, -2*np.pi*self.mu*np.sqrt(a/np.pi)*q/(self.rho*delta*a) +\
                1/self.rho * (2*np.sqrt(a) * (np.sqrt(np.pi)*f +\
                np.sqrt(self.A0)*df) - a*df) * xgrad])
        
    
    def solve(self, lw, U_in, U_out, t, dt, save, i, **kwargs):
        # solve for current timestep
        U1 = lw.solve(self.U0, U_in, U_out, t, self.F, self.S, dt, **kwargs)
        np.copyto(self.U0, U1)
        if save:
            np.copyto(self.U[:,i,:], U1)
        return U1
        
        
    def dump_results(self, suffix, data_dir):
        np.savetxt("%s/u%d_%s.csv" % (data_dir, self.pos, suffix),
                   self.U[1,:,:], delimiter=',')
        np.savetxt("%s/a%d_%s.csv" % (data_dir, self.pos, suffix),
                   self.U[0,:,:], delimiter=',')  
        np.savetxt("%s/p%d_%s.csv" % (data_dir, self.pos, suffix),
                   self.p(self.U[0,:,:]), delimiter=',') 
                   
                   
    def spatial_plots(self, suffix, plot_dir, n):
        nt = len(self.U[0,:,0])        
        skip = int(nt/n)
        u = ['a', 'q', 'p']
        l = ['m^2', 'm^3/s', 'Pa']
        positions = range(0,nt-1,skip)
        for i in range(2):
            y = self.U[i,positions,:]
            fname = "%s/%s%d_%s_spatial.png" % (plot_dir, u[i], self.pos, suffix)
            Artery.plot(suffix, plot_dir, self.x, y, positions, "m", l[i],
                        fname)
        y = self.p(self.U[1,positions,:])
        fname = "%s/%s%d_%s_spatial.png" % (plot_dir, u[2], self.pos, suffix)
        Artery.plot(suffix, plot_dir, self.x, y, positions, "m", l[2],
                        fname)
            
            
    def time_plots(self, suffix, plot_dir, n, time):
        nt = len(time)
        skip = int(self.nx/n)
        u = ['a', 'u', 'p']
        l = ['m^2', 'm/s', 'Pa']
        positions = range(0,self.nx-1,skip)
        #positions = range(0,10)
        for i in range(2):
            y = self.U[i,:,positions]
            fname = "%s/%s%d_%s_time.png" % (plot_dir, u[i], self.pos, suffix)
            Artery.plot(suffix, plot_dir, time, y, positions, "t", l[i],
                        fname)
        y = self.p(self.U[1,:,positions])
        fname = "%s/%s%d_%s_time.png" % (plot_dir, u[2], self.pos, suffix)
        Artery.plot(suffix, plot_dir, time, y, positions, "m", l[2],
                        fname)
            
            
    @staticmethod            
    def plot(suffix, plot_dir, x, y, labels, xlabel, ylabel, fname):
        plt.figure(figsize=(10,6))
        s = y.shape
        n = min(s)
        for i in range(n):
            plt.plot(x, y[i,:], label="%d" % (labels[i]), lw=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(fname, dpi=600, bbox_inches='tight')
        
    
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
    def k(self):
        return self._k
        
        
    @property
    def Ehr(self):
        return self._Ehr
        
        
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