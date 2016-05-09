# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

from artery import Artery


class ArteryNetwork(object):
    """
    Class representing a network of arteries.
    """
    
    @property
    def depth(self):
        return self._depth
        
        
    @property
    def arteries(self):
        return self._arteries
    
    
    def __init__(self, prop, depth):
        self._depth = depth
        self._arteries = []
        R = np.zeros(depth)
        R[0] = prop['R']
        alpha = prop['alpha']
        beta = prop['beta']
        lam = prop['lam']
        k = [prop['k1'], prop['k2'], prop['k3']]
        sigma = prop['sigma']        
        rho = prop['rho']
        mu = prop['mu']
        self.setup_arteries(R[0], alpha, beta, lam, k, sigma, rho, mu)
        
        
    def setup_arteries(self, R, alpha, beta, lam, k, sigma, rho, mu):
        self.arteries.append(Artery(R, lam, k, sigma, rho, mu))        
        radii = [R]
        for i in range(1,self.depth):
            new_radii = []
            for radius in radii:    
                ra = radius * alpha
                rb = radius * beta
                self.arteries.append(ra, lam, k, sigma, rho, mu)
                self.arteries.append(rb, lam, k, sigma, rho, mu)
                new_radii.append(ra)
                new_radii.append(rb)
            radii = new_radii
            
            
    def mesh(self, nx):
        for artery in self.arteries:
            artery.mesh(nx)
            
    
    def set_time(self, nt, dt):
        for artery in self.arteries:
            artery.set_time(nt, dt)
            
            
    def solve(self, u0, u_in, p_out):
        An = []
        Un = []
        for artery in self.arteries:
            A, U = artery.solve(u0, u_in, p_out)
            An.append(A)
            Un.append(U)
        return An, Un
            
            
            
