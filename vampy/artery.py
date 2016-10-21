# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import sys
import matplotlib.pylab as plt

import utils

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import interp1d


class Artery(object):
    """
    Class representing an artery.
    """
        
        
    def __init__(self, pos, Ru, Rd, lam, k, Re):
        """
        Artery constructor.
        
        :param pos: artery ID
        :param Ru: upstream radius
        :param Rd: downstream radius
        :param lam: length-to-radius (upstream) ratio 
        :param k: iterable containing elasticity parameters k1, k2, k3
        :param Re: Reynold's number
        """
        self._pos = pos
        self._Ru = Ru
        self._Rd = Rd
        self._L = Ru*lam
        self._k = k
        self._Re = Re
        
        
    def initial_conditions(self, u0, ntr):
        if not hasattr(self, '_nx'):
            raise AttributeError('Artery not meshed. Execute mesh(self, dx) \
before setting initial conditions.')
        self.U = np.zeros((2, ntr, self.nx))
        self.P = np.zeros((ntr, self.nx))
        self.U0 = np.zeros((2, self.nx))
        self.U0[0,:] = self.A0.copy()
        self.U0[1,:].fill(0.0)
        
        
    def mesh(self, dx):
        """
        Meshes an artery using spatial step size dx.
        
        :param dx: Spatial step size
        """
        self._dx = dx
        self._nx = int(self.L/dx)+1
        if self.nx-1 != self.L/dx:
            self.L = dx * (self.nx-1)
        X = np.linspace(0.0, self.L, self.nx)#/self.L
        #R = self.Ru * np.power((self.Rd/self.Ru), X)
        R = np.linspace(self.Rd, self.Ru, self.nx)
        self._A0 = R*R*np.pi
        Ehr = self.k[0] * np.exp(self.k[1]*R) + self.k[2]
        #Ehr = np.full_like(R, self.k[0] * np.exp(self.k[1]*R[0]) + self.k[2])
        self._f = 4/3 * Ehr
        self._df = 4/3 * self.k[0] * self.k[1] * np.exp(self.k[1]*R)
        #self._df = np.full_like(R, 4/3 * self.k[0] * self.k[1] * np.exp(self.k[1]*R[0]))
        self._xgrad = np.gradient(R, dx)
        
        
    def boundary_layer_thickness(self, nu, T):
        """
        Calculates the boundary layer thickness of the artery according to
        
        delta = sqrt(nu*T/2*pi).
        
        :param T: Length of one periodic cycle.
        """
        self._delta = np.sqrt(nu*T/(2*np.pi))
        
        
    def p(self, a, **kwargs):
        if 'j' in kwargs:
            j = kwargs['j']
            p = self.f[j] * (1 - np.sqrt(self.A0[j]/a))
        else:
            p = self.f * (1 - np.sqrt(self.A0/a))
        return p
        

    def wave_speed(self, a):
        Ehr = 3/4 * self.f
        return -np.sqrt(2/3 * Ehr * np.sqrt(self.A0/a))
        
        
    def F(self, U, **kwargs):
        a, q = U
        out = np.zeros(U.shape)
        out[0] = q
        if 'j' in kwargs:
            j = kwargs['j']
            a0 = self.A0[j]
            f = self.f[j]
        elif 'k' in kwargs:
            j = kwargs['j']
            k = kwargs['k']
            a0 = self.A0[j:k]
            f = self.f[j:k]
        else:
            raise IndexError("Required to supply at least one index in function F.")
        out[1] = q*q/a + f * np.sqrt(a0*a)
        return out
        
        
    def S(self, U, **kwargs):
        a, q = U
        out = np.zeros(U.shape)
        if 'j' in kwargs:
            j = kwargs['j']
            a0 = self.A0[j]
            xgrad = self.xgrad[j]
            f = self.f[j]
            df = self.df[j]
        elif 'k' in kwargs:
            j = kwargs['j']
            k = kwargs['k']
            a0 = self.A0[j:k]
            xgrad = self.xgrad[j:k]
            f = self.f[j:k]
            df = self.df[j:k]
        else:
            raise IndexError("Required to supply at least one index in function S.")
        R = np.sqrt(a/np.pi)
        out[1] = -2*np.pi*R*q/(self.Re*self.delta*a) +\
                (2*np.sqrt(a) * (np.sqrt(np.pi)*f +\
                np.sqrt(a0)*df) - a*df) * xgrad
        return out
        
        
    def dBdx(self, l, xi):
        if l > self.L:
            x_0 = self.L-self.dx
            x_1 = self.L
            f_l = utils.extrapolate(l, [x_0, x_1], [self.f[-2], self.f[-1]])  
            A0_l = utils.extrapolate(l, [x_0, x_1], [self.A0[-2], self.A0[-1]])  
            df_l = utils.extrapolate(l, [x_0, x_1], [self.df[-2], self.df[-1]])
            xgrad_l = utils.extrapolate(l, [x_0, x_1],
                                        [self.xgrad[-2], self.xgrad[-1]])
        elif l < 0.0:
            x_0 = self.dx
            x_1 = 0.0
            f_l = utils.extrapolate(l, [x_0, x_1], [self.f[1], self.f[0]])  
            A0_l = utils.extrapolate(l, [x_0, x_1], [self.A0[1], self.A0[0]]) 
            df_l = utils.extrapolate(l, [x_0, x_1], [self.df[1], self.df[0]])
            xgrad_l = utils.extrapolate(l, [x_0, x_1],
                                        [self.xgrad[1], self.xgrad[0]])
        elif l == self.L:
            f_l = self.f[-1]
            A0_l = self.A0[-1]
            df_l = self.df[-1]
            xgrad_l = self.xgrad[-1]
        else:
            f_l = self.f[0]
            A0_l = self.A0[0]
            df_l = self.df[0]
            xgrad_l = self.xgrad[0]
        return (2*np.sqrt(xi) * (np.sqrt(np.pi)*f_l + np.sqrt(A0_l)*df_l) -\
                    xi*df_l) * xgrad_l
        
        
    def dBdxi(self, l, xi):
        if l > self.L:
            x_0 = self.L-self.dx
            x_1 = self.L
            f_l = utils.extrapolate(l, [x_0, x_1], [self.f[-2], self.f[-1]])  
            A0_l = utils.extrapolate(l, [x_0, x_1], [self.A0[-2], self.A0[-1]])  
        elif l < 0.0:
            x_0 = self.dx
            x_1 = 0.0
            f_l = utils.extrapolate(l, [x_0, x_1], [self.f[1], self.f[0]])  
            A0_l = utils.extrapolate(l, [x_0, x_1], [self.A0[1], self.A0[0]]) 
        elif l == self.L:
            f_l = self.f[-1]
            A0_l = self.A0[-1]
        else:
            f_l = self.f[0]
            A0_l = self.A0[0]
        return f_l/2 * np.sqrt(A0_l/xi)
        
        
    def dBdxdxi(self, l, xi):
        if l > self.L:
            x_0 = self.L-self.dx
            x_1 = self.L
            f_l = utils.extrapolate(l, [x_0, x_1], [self.f[-2], self.f[-1]])   
            df_l = utils.extrapolate(l, [x_0, x_1], [self.df[-2], self.df[-1]])   
            A0_l = utils.extrapolate(l, [x_0, x_1], [self.A0[-2], self.A0[-1]])  
            xgrad_l = utils.extrapolate(l, [x_0, x_1],
                                        [self.xgrad[-2], self.xgrad[-1]])  
        elif l < 0.0:
            x_0 = self.dx
            x_1 = 0.0
            f_l = utils.extrapolate(l, [x_0, x_1], [self.f[1], self.f[0]])   
            df_l = utils.extrapolate(l, [x_0, x_1], [self.df[1], self.df[0]])   
            A0_l = utils.extrapolate(l, [x_0, x_1], [self.A0[1], self.A0[0]])  
            xgrad_l = utils.extrapolate(l, [x_0, x_1],
                                        [self.xgrad[1], self.xgrad[0]])  
        elif l == self.L:
            f_l = self.f[-1]   
            df_l = self.df[-1]
            A0_l = self.A0[-1]
            xgrad_l = self.xgrad[-1]
        else:
            f_l = self.f[0]   
            df_l = self.df[0]
            A0_l = self.A0[0]
            xgrad_l = self.xgrad[0]
        return (1/(2*np.sqrt(xi)) * (f_l*np.sqrt(np.pi) +\
                                    df_l*np.sqrt(A0_l)) - df_l) * xgrad_l
                                    
                                    
    def dFdxi2(self, l, xi1, xi2):
        if l > self.L:
            x_0 = self.L-self.dx
            x_1 = self.L
            R0_l = utils.extrapolate(l, [x_0, x_1], 
                    [np.sqrt(self.A0[-2]/np.pi), np.sqrt(self.A0[-1]/np.pi)])
        elif l < 0.0:
            x_0 = self.dx
            x_1 = 0.0
            R0_l = utils.extrapolate(l, [x_0, x_1], 
                    [np.sqrt(self.A0[1]/np.pi), np.sqrt(self.A0[0]/np.pi)])
        elif l == self.L:
            R0_l = np.sqrt(self.A0[-1]/np.pi)
        else:
            R0_l = np.sqrt(self.A0[0]/np.pi)
        return 2*np.pi*R0_l*xi1/(self.delta*self.Re*xi2*xi2)
        
        
    def dFdxi1(self, l, xi2):
        if l > self.L:
            x_0 = self.L-self.dx
            x_1 = self.L
            R0_l = utils.extrapolate(l, [x_0, x_1], 
                    [np.sqrt(self.A0[-2]/np.pi), np.sqrt(self.A0[-1]/np.pi)])
        elif l < 0.0:
            x_0 = self.dx
            x_1 = 0.0
            R0_l = utils.extrapolate(l, [x_0, x_1], 
                    [np.sqrt(self.A0[1]/np.pi), np.sqrt(self.A0[0]/np.pi)])
        elif l == self.L:
            R0_l = np.sqrt(self.A0[-1]/np.pi)
        else:
            R0_l = np.sqrt(self.A0[0]/np.pi)
        return -2*np.pi*R0_l/(self.delta*self.Re*xi2)
        
        
    def dpdx(self, l, xi):
        if l > self.L:
            x_0 = self.L-self.dx
            x_1 = self.L
            f_l = utils.extrapolate(l, [x_0, x_1], [self.f[-2], self.f[-1]])   
            A0_l = utils.extrapolate(l, [x_0, x_1], [self.A0[-2], self.A0[-1]])  
        elif l < 0.0:
            x_0 = self.dx
            x_1 = 0.0
            f_l = utils.extrapolate(l, [x_0, x_1], [self.f[1], self.f[0]])   
            A0_l = utils.extrapolate(l, [x_0, x_1], [self.A0[1], self.A0[0]])
        elif l == self.L:
            f_l = self.f[-1]   
            A0_l = self.A0[-1]
        else:
            f_l = self.f[0]   
            A0_l = self.A0[0]
        return f_l/2 * np.sqrt(A0_l/xi**3)
        
        
    def solve(self, lw, U_in, U_out, save, i):
        # solve for current timestep
        U1 = lw.solve(self.U0, U_in, U_out, self.F, self.S)
        if save:
            self.P[i,:] = self.p(self.U0[0,:])
            np.copyto(self.U[:,i,:], self.U0)
        np.copyto(self.U0, U1)
        
        
        
    def dump_results(self, suffix, data_dir):
        np.savetxt("%s/%s/u%d_%s.csv" % (data_dir, suffix, self.pos, suffix),
                   self.U[1,:,:], delimiter=',')
        np.savetxt("%s/%s/a%d_%s.csv" % (data_dir, suffix, self.pos, suffix),
                   self.U[0,:,:], delimiter=',')  
        np.savetxt("%s/%s/p%d_%s.csv" % (data_dir, suffix, self.pos, suffix),
                   self.P, delimiter=',') 
                   

    @property
    def pos(self):
        return self._pos
    
    @property
    def Ru(self):
        return self._Ru
        
    @property
    def Rd(self):
        return self._Rd
        
    @property
    def A0(self):
        return self._A0
        
    @property
    def L(self):
        return self._L
        
    @L.setter
    def L(self, value): 
        self._L = value
        
    @property
    def k(self):
        return self._k
        
    @property
    def Re(self):
        return self._Re
        
    @property
    def f(self):
        return self._f

    @property
    def df(self):
        return self._df
        
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
        
    @property
    def delta(self):
        return self._delta

    @property
    def xgrad(self):
        return self._xgrad