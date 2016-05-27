# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

from artery import Artery


class ArteryNetwork(object):
    """
    Class representing a network of arteries.
    """
    
    
    def __init__(self, R, a, b, lam, sigma, rho, mu, depth, **kwargs):
        self._depth = depth
        self._arteries = []
        self.setup_arteries(R, a, b, lam, sigma, rho, mu, **kwargs)
        self._t = 0.0
        
        
    def setup_arteries(self, R, a, b, lam, sigma, rho, mu, **kwargs):
        pos = 0
        self.arteries.append(Artery(pos, R, lam, sigma, rho, mu, **kwargs)) 
        pos += 1
        radii = [R]
        for i in range(1,self.depth):
            new_radii = []
            for radius in radii:    
                ra = radius * a
                rb = radius * b
                self.arteries.append(Artery(pos, ra, lam, sigma, rho, mu, **kwargs))
                pos += 1
                self.arteries.append(Artery(pos, rb, lam, sigma, rho, mu, **kwargs))
                pos += 1
                new_radii.append(ra)
                new_radii.append(rb)
            radii = new_radii
            
            
    def initial_conditions(self, u0):
        for artery in self.arteries:
            artery.initial_conditions(u0)            
            
            
    def mesh(self, nx):
        for artery in self.arteries:
            artery.mesh(nx)
            
    
    def set_time(self, nt, dt, T=0.0):
        for artery in self.arteries:
            artery.set_time(nt, dt, T)
            
            
    def tf(self):
        return self.dt * self.nt
        
        
    def timestep(self):
        self._t += self.dt
            
    
    @staticmethod        
    def inlet_bc(artery, u_prev, u_in):
        c_prev = artery.wave_speed(u_prev[0,0])
        w_prev = u_prev[1,0:2] - 4*c_prev
        lam_2 = u_prev[1,0] - c_prev
        x_0 = artery.x[0] - lam_2 * artery.dt
        w_2 = utils.extrapolate(x_0, artery.x[0:2], w_prev)
        a_in = (u_in - w_2)**4 / 64 * (artery.rho/artery.beta)**2
        return np.array([a_in, u_in])
     
    
    @staticmethod
    def outlet_bc(artery, u_prev, a_out):
        c_prev = artery.wave_speed(u_prev[0,-1])
        w_prev = u_prev[1,-2:] + 4*c_prev
        lam_1 = u_prev[1,-1] + c_prev
        x_0 = artery.x[-1] - lam_1 * artery.dt
        w_1 = utils.extrapolate(x_0, artery.x[-2:], w_prev)
        u_out = w_1 - 4*a_out**(1/4) * np.sqrt(artery.beta/(2*artery.rho))
        return np.array([a_out, u_out])
        
    
    @staticmethod
    def bifurcation_bc(artery, p, d1, d2):
        pass
    
    
    
    @staticmethod
    def cfl_condition(self, u, **kwargs):
        c = self.wave_speed(u[0])
        v = (u[1] + c, u[1] - c)
        left = self.dt/self.dx
        right = np.power(np.absolute(v), -1)
        return False if (left > right).any() else True
            
            
    def solve(self, u0, u_in, p_out, T):
        An = []
        Un = []
        while self.t < self.tf():
            
            for artery in self.arteries:
                nx = len(artery.x)
                lw = LaxWendroff(nx, artery.dx)
                i = 0
                np.copyto(artery.U[:,0,:], artery.U0)        
                if artery.pos == 0:
                    # inlet boundary condition
                    if artery.T > 0:
                        in_t = utils.periodic(self.t, self.T)
                    U_in = inlet_bc(artery.U0, u_in(in_t))
                else:
                    #todo: bifurcation inlet boundary
                    pass
                if artery.pos >= (len(self.arteries) - 2**(self.depth-1)):
                    # outlet boundary condition
                    U_out = outlet_bc(artery.U0, artery.A0)
                else:
                    #todo: bifurcation outlet condition
                    pass
                A, U = artery.solve(U_in, U_out, u0)
                
            if abs(self.t - self.dtr*i) < self.dt:
                np.copyto(self.U[:,i,:], U1)
            i += 1
            self.timestep()
            np.copyto(self.U0, U1)                
                
                
            An.append(A)
            Un.append(U)
        return An, Un
            
            
    @property
    def depth(self):
        return self._depth
        
        
    @property
    def arteries(self):
        return self._arteries
        
        
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
    def ntr(self):
        return self._ntr
        
        
    @property
    def dtr(self):
        return self._dtr