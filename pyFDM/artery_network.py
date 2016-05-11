# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

from artery import Artery


class ArteryNetwork(object):
    """
    Class representing a network of arteries.
    """
    
    
    def __init__(self, R, alpha, beta, lam, k, sigma, rho, mu, depth):
        self._depth = depth
        self._arteries = []
        self.setup_arteries(R, alpha, beta, lam, k, sigma, rho, mu)
        
        
    def setup_arteries(self, R, alpha, beta, lam, k, sigma, rho, mu):
        if self.depth == 1:
            # trivial case of only one artery
            out_bc = ArteryNetwork.outlet_bc  
        else:
            out_bc = ArteryNetwork.bifurcation_bc
        in_bc = ArteryNetwork.inlet_bc
        self.arteries.append(Artery(R, lam, k, sigma, rho, mu, in_bc, 
                                    out_bc)) 
        radii = [R]
        for i in range(1,self.depth):
            new_radii = []
            for radius in radii:    
                ra = radius * alpha
                rb = radius * beta
                if i == self.depth-1:
                    out_bc = ArteryNetwork.outlet_bc
                else:
                    out_bc = ArteryNetwork.bifurcation_bc
                in_bc = ArteryNetwork.bifurcation_bc
                self.arteries.append(Artery(ra, lam, k, sigma, rho, mu,
                                            in_bc, out_bc))
                self.arteries.append(Artery(rb, lam, k, sigma, rho, mu, 
                                            in_bc, out_bc))
                new_radii.append(ra)
                new_radii.append(rb)
            radii = new_radii
            
            
    def initial_conditions(self):
        for artery in self.arteries:
            artery.initial_conditions()            
            
            
    def mesh(self, nx):
        for artery in self.arteries:
            artery.mesh(nx)
            
    
    def set_time(self, nt, dt):
        for artery in self.arteries:
            artery.set_time(nt, dt)
            
            
    def solve(self, u0, u_in, p_out, T):
        An = []
        Un = []
        for artery in self.arteries:
            A, U = artery.solve(u0, u_in, p_out, T)
            An.append(A)
            Un.append(U)
        return An, Un
        
        
    @staticmethod        
    def calculate_characteristic(a, u_prev, u1, beta, rho, dt, x):
        c_prev = wave_speed(a, beta, rho)
        w_prev = u_prev - 4*c_prev
        lam_2 = u1 - c_prev
        x_0 = x[0] - lam_2 * dt
        return extrapolate(x_0, x, w_prev)
            
    
    @staticmethod        
    def inlet_bc(u_prev, t, (u_in, beta, rho, x, dt)):
        w_2 = calculate_characteristic(u_prev[0,0], u_prev[1,0:2], u_prev[1,0],
                                       beta, rho, dt, x[0:2])
        a_in = (u_in(t) - w_2)**4 / 64 * (rho/beta)**2
        return np.array([a_in, u_in(t)])
     
    
    @staticmethod
    def outlet_bc(u_prev, t, (a_out, beta, rho, x, dt)):
        w_1 = calculate_characteristic(u_prev[0,-1], u_prev[1,-2:], u_prev[1,-1],
                                       beta, rho, dt, x[-2:])
        u_out = w_1 - 4*a_out**(1/4) * np.sqrt(beta/(2*rho))
        return np.array([a_out, u_out])
        
    
    @staticmethod    
    def bifurcation_bc(p, d1, d2):
        uprev_p = p.get_uprev()
        F = bifurcation_system(u_prev)
        return np.array([0, 0])
            
            
    @property
    def depth(self):
        return self._depth
        
        
    @property
    def arteries(self):
        return self._arteries
