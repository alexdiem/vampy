# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import sys
import matplotlib.pylab as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class Artery(object):
    """
    Class representing an artery.
    """
        
        
    def __init__(self, pos, R, lam, sigma, rho, mu, **kwargs):
        self._pos = pos
        self._R = R
        self._A0 = np.pi*R**2
        self._L = R[0]*lam
        self._k = kwargs['k']
        self._Ehr = self.k[0] * np.exp(self.k[1]*R) + self.k[2]
        self._rho = rho
        self._mu = mu
        
        
    def initial_conditions(self, u0, ntr):
        if not hasattr(self, '_x'):
            raise AttributeError('Artery not meshed. mesh(self, nx) has to be \
executed first')
        self.U = np.zeros((2, ntr, len(self.x)))
        self.P = np.zeros((ntr, len(self.x)))
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
        return 4/3 * self.Ehr * (1 - np.sqrt(self.A0 / a))
        
        
    def wave_speed(self, a):
        return np.sqrt(-self.rho * 2/3 * self.Ehr * np.sqrt(self.A0/a))
        
        
    def F(self, U, **kwargs):
        a, q = U
        f = 4/3 * self.Ehr
        out = np.zeros(U.shape)
        out[0] = q
        if U.shape == (2,):
            out[1] = np.power(q,2)/a + f[kwargs['j']]/self.rho * np.sqrt(self.A0[kwargs['j']]*a)
        elif U.shape == (2, self.nx-2):
            out[1] = np.power(q,2)/a + f[1:-1]/self.rho * np.sqrt(self.A0[1:-1]*a)
        return out
        
        
    def S(self, U, **kwargs):
        a, q = U
        #delta = round(np.sqrt(self.mu*kwargs['T']/(self.rho*2*np.pi)), 6)
        delta = 0.000855
        f = 4/3 * self.Ehr
        df = 4/3 * self.k[0] * self.k[1] * np.exp(self.k[1] * self.R)
        xgrad = np.gradient(self.R)
        out = np.zeros(U.shape)
        if U.shape == (2,):
            out[1] = -2*np.pi*self.mu*np.sqrt(a/np.pi)*q/(self.rho*delta*a) +\
                1/self.rho * (2*np.sqrt(a) * (np.sqrt(np.pi)*f[kwargs['j']] +\
                np.sqrt(self.A0[kwargs['j']])*df[kwargs['j']]) - a*df[kwargs['j']]) * xgrad[kwargs['j']]
        elif U.shape == (2, self.nx-2):
            out[1] = -2*np.pi*self.mu*np.sqrt(a/np.pi)*q/(self.rho*delta*a) +\
                1/self.rho * (2*np.sqrt(a) * (np.sqrt(np.pi)*f[1:-1] +\
                np.sqrt(self.A0[1:-1])*df[1:-1]) - a*df[1:-1]) * xgrad[1:-1]
        return out
        

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
                     
        y = self.P[positions,:]    
        fname = "%s/%s%d_%s_spatial.png" % (plot_dir, u[2], self.pos, suffix)
        Artery.plot(suffix, plot_dir, self.x, y, positions, "m", l[2],
                        fname)
            
            
    def time_plots(self, suffix, plot_dir, n, time):
        nt = len(time)
        skip = int(self.nx/n)
        u = ['a', 'q', 'p']
        l = ['m^2', 'm/s', 'Pa']
        positions = range(0,self.nx-1,skip)
        #positions = range(0,1)
        for i in range(2):
            y = self.U[i,:,positions]
            fname = "%s/%s%d_%s_time.png" % (plot_dir, u[i], self.pos, suffix)
            Artery.plot(suffix, plot_dir, time, y, positions, "t", l[i],
                        fname)
                        
        y = np.transpose(self.P[:,positions]) 
        fname = "%s/%s%d_%s_time.png" % (plot_dir, u[2], self.pos, suffix)
        Artery.plot(suffix, plot_dir, time, y, positions, "t", l[2],
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
        
        
    def s3d_plot(self, suffix, plot_dir, time):
        fig = plt.figure(figsize=(10,6))
        ax = fig.gca(projection='3d')
        x = np.linspace(0, self.L, len(time))
        X, Y = np.meshgrid(time, x)
        dz = int(self.nx/len(time))
        Z = self.P[:,0:self.nx+1:dz]
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        fname = "%s/p3d%d_%s_spatial.png" % (plot_dir, self.pos, suffix)
        fig.savefig(fname, dpi=600, bbox_inches='tight')
        
    
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
        
    
    @property
    def P(self):
        return self._P
        
        
    @P.setter
    def P(self, value): 
        self._P = value