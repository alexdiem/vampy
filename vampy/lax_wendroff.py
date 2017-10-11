# -*- coding: utf-8 -*-

from __future__ import division

import sys
import numpy as np
from scipy.interpolate import interp1d

import vampy.utils as utils


class LaxWendroff(object):
    """
    Class implementing Richtmyer's 2 step Lax-Wendroff method.
    """
    
    
    def __init__(self, theta, gamma, nx):
        """
        Constructor for LaxWendroff class.
        
        :param theta: factor for flux vector
        :param gamma: factor for source vector
        :param nx: number of spatial points
        """
        self._theta = theta
        self._gamma = gamma
        self._nx = nx
        

    def solve(self, U0, U_in, U_out, F, S):
        """
        Solver implementing Richtmyer's two-step Lax-Wendroff method [1,2].
        
        [1] R. D. Richtmyer. A Survey of Difference Methods for Non-Steady Fluid Dynamics. NCAR Technical Notes, 63(2), 1963.
        [2] R. J. LeVeque. Numerical Methods for Conservation Laws. Birkhauser Verlag, Basel, Switzerland, 2nd edition, 1992.
        
        :param U0: solution from previous time step
        :param U_in: inlet boundary condition
        :param U_out: outlet boundary condition
        :param F: flux function (see [2])
        :param S: source function (see [2])
        """
        # U0: previous timestep, U1 current timestep
        U1 = np.zeros((2,self.nx))
        # apply boundary conditions
        U1[:,0] = U_in
        U1[:,-1] = U_out
        # calculate half steps
        U_np_mp = (U0[:,2:]+U0[:,1:-1])/2 -\
            self.theta*(F(U0[:,2:], j=2, k=self.nx)-F(U0[:,1:-1], j=1, k=-1))/2 +\
            self.gamma*(S(U0[:,2:], j=2, k=self.nx)+S(U0[:,1:-1], j=1, k=-1))/2
        U_np_mm = (U0[:,1:-1]+U0[:,0:-2])/2 -\
            self.theta*(F(U0[:,1:-1], j=1, k=-1)-F(U0[:,0:-2], j=0, k=-2))/2 +\
            self.gamma*(S(U0[:,1:-1], j=1, k=-1)+S(U0[:,0:-2], j=0, k=-2))/2
        # calculate full step
        U1[:,1:-1] = U0[:,1:-1] -\
            self.theta*(F(U_np_mp, j=1, k=-1)-F(U_np_mm, j=1, k=-1)) +\
            self.gamma*(S(U_np_mp, j=1, k=-1)+S(U_np_mm, j=1, k=-1))
        return U1
        
        
    @property   
    def theta(self):
        """
        dt/dx
        """
        return self._theta
        
    @property   
    def gamma(self):
        """
        dt/2
        """
        return self._gamma
        
    @property   
    def nx(self):
        """
        Number of spatial steps
        """        
        return self._nx