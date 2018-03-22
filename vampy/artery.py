# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import sys
import matplotlib.pylab as plt

import vampy.utils as utils

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import interp1d


class Artery(object):
    """
    Class representing an artery.
    
    :param pos: Artery ID
    :param Ru: Upstream radius
    :param Rd: Downstream radius
    :param lam: Length-to-radius (upstream) ratio 
    :param k: Iterable containing elasticity parameters k1, k2, k3
    :param Re: Reynold's number
    """
        
        
    def __init__(self, pos, Ru, Rd, lam, k, Re, p0):
        """
        Artery constructor.
        """
        self._pos = pos
        self._Ru = Ru
        self._Rd = Rd
        self._L = Ru*lam
        self._k = k
        self._Re = Re
        self._p0 = p0
        
        
    def initial_conditions(self, u0):
        """
        Initialises solution arrays with initial conditions.
        Checks if artery.mesh(dx) has been called first.
        
        :param u0: Initial conditions for solution
        :param ntr: Number of solution time steps to be stored
        :raises: AttributeError
        """
        if not hasattr(self, '_nx'):
            raise AttributeError('Artery not meshed. Execute mesh(self, dx) \
before setting initial conditions.')
        self.U0[0,:] = self.A0.copy()
        self.U0[1,:].fill(u0)
        
        
    def mesh(self, dx, ntr):
        """
        Meshes an artery using spatial step size dx.
        
        :param dx: Spatial step size
        """
        self._dx = dx
        self._nx = int(self.L/dx)+1
        if self.nx-1 != self.L/dx:
            self.L = dx * (self.nx-1)
        X = np.linspace(0.0, self.L, self.nx)
        R = self.Ru * np.power((self.Rd/self.Ru), X/self.L)
        self._A0 = np.power(R, 2)*np.pi
        self._f = 4/3 * (self.k[0] * np.exp(self.k[1]*R) + self.k[2])
        self._df = 4/3 * self.k[0] * self.k[1] * np.exp(self.k[1]*R)
        self._xgrad = (self.Ru * np.log(self.Rd/self.Ru) * np.power((self.Rd/self.Ru), X/self.L))/self.L
        self.U = np.zeros((2, ntr, self.nx))
        self.P = np.zeros((ntr, self.nx))
        self.U0 = np.zeros((2, self.nx))
        
        
    def boundary_layer_thickness(self, nu, T):
        """
        Calculates the boundary layer thickness of the artery according to
        
        delta = sqrt(nu*T/2*pi).
        
        :param nu: Viscosity of blood
        :param T: Length of one periodic cycle.
        """
        self._delta = np.sqrt(nu*T/(2*np.pi))
        
        
    def p(self, a, **kwargs):
        """
        Calculates pressure according to the state equation.
        
        :param a: Area
        :param \**kwargs: See below
        :returns: Pressure 

        :Keyword Arguments:
            * *j* (``int``) -- Index variable
        """
        if 'j' in kwargs:
            j = kwargs['j']
            p = self.f[j] * (1 - np.sqrt(self.A0[j]/a)) + self.p0
        else:
            p = self.f * (1 - np.sqrt(self.A0/a)) + self.p0
        return p
        

    def wave_speed(self, a):
        """
        Calculates the wave speed (required to check CFL condition).
        
        :param a: Area
        :returns: Wave speed
        """
        return -np.sqrt(0.5 * self.f * np.sqrt(self.A0/a))
        
        
    def F(self, U, **kwargs):
        """
        Calculates the flux vector.
        
        :param U: Previous solution
        :param \**kwargs: See below
        :returns: Flux for current solution
        :raises: IndexError
        
        :Keyword Arguments:
            * *j* (``int``) -- Index variable (start)
            * *k* (``int``) -- Index variable (end)
        """
        a, q = U
        out = np.empty_like(U)
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
        out[1] = np.power(q, 2)/a + f * np.sqrt(a0*a)
        return out
        
        
    def S(self, U, **kwargs):
        """
        Calculates the flux vector.
        
        :param U: Previous solution
        :param \**kwargs: See below
        :returns: Flux for current solution
        :raises: IndexError
        
        :Keyword Arguments:
            * *j* (``int``) -- Index variable (start)
            * *k* (``int``) -- Index variable (end)
        """
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
        R = np.sqrt(a0/np.pi)
        out[1] = -(2*np.pi*R/(self.Re*self.delta)) * (q/a) +\
                (2*np.sqrt(a) * (np.sqrt(np.pi)*f +\
                np.sqrt(a0)*df) - a*df) * xgrad
        return out
        
        
    def dBdx(self, l, xi):
        """
        Calculates dB/dx (see [1]).
        
        [1] M. S. Olufsen. Modeling of the Arterial System with Reference to an Anesthesia Simulator. PhD thesis, University of Roskilde, Denmark, 1998.
        
        :param l: Position, either M+1/2 or -1/2.
        :param xi: Area.
        :returns: Solution to dB/dx
        """
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
        """
        Calculates dB/dx_i (see [1]).
        
        [1] M. S. Olufsen. Modeling of the Arterial System with Reference to an Anesthesia Simulator. PhD thesis, University of Roskilde, Denmark, 1998.
        
        :param l: Position, either M+1/2 or -1/2.
        :param xi: Area.
        :returns: Solution to dB/dx_i
        """
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
        """
        Calculates d^2B/dxdx_i (see [1]).
        
        [1] M. S. Olufsen. Modeling of the Arterial System with Reference to an Anesthesia Simulator. PhD thesis, University of Roskilde, Denmark, 1998.
        
        :param l: Position, either M+1/2 or -1/2.
        :param xi: Area.
        :returns: Solution to d^2B/dxdx_i
        """
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
        """
        Calculates dF/dx_2 (see [1]).
        
        [1] M. S. Olufsen. Modeling of the Arterial System with Reference to an Anesthesia Simulator. PhD thesis, University of Roskilde, Denmark, 1998.
        
        :param l: Position, either M+1/2 or -1/2.
        :param xi: Area.
        :returns: Solution to dF/dx_2
        """
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
        return 2*np.pi*R0_l/(self.delta*self.Re) * xi1/(xi2*xi2)
        
        
    def dFdxi1(self, l, xi2):
        """
        Calculates dF/dx_1 (see [1]).
        
        [1] M. S. Olufsen. Modeling of the Arterial System with Reference to an Anesthesia Simulator. PhD thesis, University of Roskilde, Denmark, 1998.
        
        :param l: Position, either M+1/2 or -1/2.
        :param xi: Area.
        :returns: Solution to dF/dx_1
        """
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
        return -2*np.pi*R0_l/(self.delta*self.Re) * 1/xi2
        
        
    def dpdx(self, l, xi):
        """
        Calculates dp/dx (see [1]).
        
        [1] M. S. Olufsen. Modeling of the Arterial System with Reference to an Anesthesia Simulator. PhD thesis, University of Roskilde, Denmark, 1998.
        
        :param l: Position, either M+1/2 or -1/2.
        :param xi: Area.
        :returns: Solution to dp/dx
        """
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
        """
        Solver calling the LaxWendroff solver and storing the new solution in U0.
        Stores new solution in output array U if save is True.
        
        :param lw: LaxWendroff object
        :param U_in: Inlet boundary condition
        :param U_out: Outlet boundary condition
        :param save: True if current time step is to be saved
        :param i: Current time step
        """
        # solve for current timestep
        U1 = lw.solve(self.U0, U_in, U_out, self.F, self.S)
        if save:
            self.P[i,:] = self.p(self.U0[0,:])
            np.copyto(self.U[:,i,:], self.U0)
        np.copyto(self.U0, U1)
        
        
        
    def dump_results(self, suffix, data_dir):
        """
        Outputs solutions U, P to csv files
        
        :param suffix: Simulation identifier
        :param data_dir: Directory data files are stored in
        """
        np.savetxt("%s/%s/u%d_%s.csv" % (data_dir, suffix, self.pos, suffix),
                   self.U[1,:,:], delimiter=',')
        np.savetxt("%s/%s/a%d_%s.csv" % (data_dir, suffix, self.pos, suffix),
                   self.U[0,:,:], delimiter=',')  
        np.savetxt("%s/%s/p%d_%s.csv" % (data_dir, suffix, self.pos, suffix),
                   self.P, delimiter=',') 
                   
                   
    @property
    def L(self):
        """
        Artery length
        """
        return self._L

    @L.setter
    def L(self, value):
        self._L = value
        
    @property
    def nx(self):
        """
        Number of spatial steps
        """
        return self._nx
        
    @property
    def Ru(self):
        """
        Upstream radius
        """
        return self._Ru
        
    @property
    def Rd(self):
        """
        Downstream radius
        """
        return self._Rd
        
    @property
    def k(self):
        """
        Elasticity parameters for relation Eh/r = k1 * exp(k2*r) + k3
        """
        return self._k
        
    @property
    def A0(self):
        """
        Area at rest
        """
        return self._A0
        
    @property
    def dx(self):
        """
        Spatial step size
        """
        return self._dx
    
    @property
    def pos(self):
        """
        Position in ArteryNetwork
        """
        return self._pos
        
    @property
    def f(self):
        """
        f = 4/3 Eh/r
        """
        return self._f
        
    @property
    def xgrad(self):
        """
        dr/dx
        """
        return self._xgrad
        
    @property
    def df(self):
        """
        df/dr
        """        
        return self._df

    @property
    def Re(self):
        """
        Reynold's number
        """
        return self._Re
        
    @property
    def delta(self):
        """
        Boundary layer thickness
        """
        return self._delta

    @property
    def p0(self):
        """
        Reynold's number
        """
        return self._p0