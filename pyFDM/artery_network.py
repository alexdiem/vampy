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
        
        
    def setup_arteries(self, R, a, b, lam, sigma, rho, mu, **kwargs):
        self.arteries.append(Artery(R, lam, sigma, rho, mu, **kwargs)) 
        radii = [R]
        for i in range(1,self.depth):
            new_radii = []
            for radius in radii:    
                ra = radius * a
                rb = radius * b
                self.arteries.append(Artery(ra, lam, sigma, rho, mu, **kwargs))
                self.arteries.append(Artery(rb, lam, sigma, rho, mu, **kwargs))
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
            
            
    def solve(self, u0, u_in, p_out, T):
        An = []
        Un = []
        for artery in self.arteries:
            A, U = artery.solve(u0, u_in, p_out, T)
            An.append(A)
            Un.append(U)
        return An, Un
            
            
    @property
    def depth(self):
        return self._depth
        
        
    @property
    def arteries(self):
        return self._arteries
