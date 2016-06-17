# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

from artery import Artery
from lax_wendroff import LaxWendroff
import utils

import sys


class ArteryNetwork(object):
    """
    Class representing a network of arteries.
    """
    
    
    def __init__(self, R, a, b, lam, rho, mu, depth, **kwargs):
        self._depth = depth
        self._arteries = []
        self.setup_arteries(R, a, b, lam, rho, mu, **kwargs)
        self._t = 0.0
        self._ntr = kwargs['ntr']
        self._progress = 10
        
        
    def setup_arteries(self, R, a, b, lam, rho, mu, **kwargs):
        pos = 0
        self.arteries.append(Artery(pos, R, lam, rho, mu, **kwargs)) 
        pos += 1
        radii = [R]
        for i in range(1,self.depth):
            new_radii = []
            for radius in radii:    
                ra = radius * a
                rb = radius * b
                self.arteries.append(Artery(pos, ra, lam, rho, mu, **kwargs))
                pos += 1
                self.arteries.append(Artery(pos, rb, lam, rho, mu, **kwargs))
                pos += 1
                new_radii.append(ra)
                new_radii.append(rb)
            radii = new_radii
            
            
    def initial_conditions(self, u0, ntr):
        for artery in self.arteries:
            artery.initial_conditions(u0, self.ntr)            
            
            
    def mesh(self, nx):
        for artery in self.arteries:
            artery.mesh(nx)
            
    
    def set_time(self, nt, dt, T=0.0, tc=1):
        self._nt = nt
        self._dt = dt
        self._tf = nt * dt
        self._dtr = nt*dt/(self.ntr-1)
        self._T = T
        self._tc = tc
            
            
    def timestep(self):
        self._t += self.dt
            
    
    @staticmethod        
    def inlet_bc(artery, q_in, in_t, dt, **kwargs):
        q_0_np = q_in(in_t-dt)
        U_0_n = artery.U0[:,0]
        U_1_n = artery.U0[:,1]
        U_12_np = (U_1_n[1] + U_0_n[1])/2 + dt/2 * (- (artery.F(U_1_n, j=1) -\
                    artery.F(U_0_n, j=1))/artery.dx + (artery.S(U_1_n, j=1) +\
                    artery.S(U_1_n, j=1))/2)
        return np.array([U_0_n[0] - 2*dt/artery.dx * (U_12_np[1] - q_0_np),
                         q_in(in_t)])
     
    
    @staticmethod
    def outlet_bc(artery, dt, **kwargs):
        R1 = 4.1e11
        R2 = 1.9e11
        Ct = 8.7137e-14
        a_n = artery.U0[0,-1]
        q_n = artery.U0[1,-1]
        p_out = p_n = artery.p(a_n)[-1] # initial guess for p_out
        
        U_np_mp = (artery.U0[:,-1] + artery.U0[:,-2])/2 +\
                    dt/2 * (-(artery.F(artery.U0[:,-1], j=-1) -\
                    artery.F(artery.U0[:,-2], j=-2))/artery.dx +\
                    (artery.S(artery.U0[:,-1], j=-1) +\
                    artery.S(artery.U0[:,-2], j=-2))/2)
        U_np_mm = (artery.U0[:,-2] + artery.U0[:,-3])/2 +\
                    dt/2 * (-(artery.F(artery.U0[:,-2], j=-2) -\
                    artery.F(artery.U0[:,-3], j=-3))/artery.dx +\
                    (artery.S(artery.U0[:,-2], j=-2) +\
                    artery.S(artery.U0[:,-3], j=-3))/2)
        U_mm = artery.U0[:,-2] - dt/artery.dx * (artery.F(U_np_mp, j=-2) -\
                artery.F(U_np_mm, j=-2)) + dt/2 * (artery.S(U_np_mp, j=-2) +\
                artery.S(U_np_mm, j=-2))
        
        kmax = 100
        k = 0
        while k < kmax:
            p_old = p_out
            q_out = (p_out - p_n)/R1 + dt/(R1*R2*Ct) * (p_n - q_n*(R1+R2)) + q_n
            a_out = a_n - dt * (q_out - U_mm[1])/artery.dx
            p_out = artery.p(a_out)[-1]
            if abs(p_old - p_out) < 1e-7:
                break
            k += 1
        return np.array([a_out, q_out])
        
    
    @staticmethod
    def bifurcation_bc(artery, p, d1, d2):
        pass
    
    
    @staticmethod
    def cfl_condition(artery, dt):
        a = artery.U0[0]
        c = artery.wave_speed(a)
        u = a / artery.U0[1,1]
        v = (u + c, u - c)
        left = dt/artery.dx
        right = np.power(np.absolute(v), -1)
        return False if (left > right).any() else True
            
    
    def solve(self, u0, q_in, p_out, T):
        tr = np.linspace(self.tf-self.T, self.tf, self.ntr)
        #tr = np.linspace(0, self.tf, self.ntr)
        # variables at 0,0 are set in initial conditions so we do one timestep
        # straight away
        self.timestep()
        i = 1
        
        while self.t < self.tf:
            save = False  

            if (self.t == self.tf) or (i < self.ntr and abs(self.t - tr[i]) < self.dtr):
                save = True
                i += 1
                
            for artery in self.arteries:
                lw = LaxWendroff(artery.nx, artery.dx)
                
                if artery.pos == 0:
                    # inlet boundary condition
                    if self.T > 0:
                        in_t = utils.periodic(self.t, self.T)
                    else:
                        in_t = self.t
                    U_in = ArteryNetwork.inlet_bc(artery, q_in, in_t, self.dt, T=self.T)
                else:
                    #todo: bifurcation inlet boundary
                    pass
                if artery.pos >= (len(self.arteries) - 2**(self.depth-1)):
                    # outlet boundary condition
                    U_out = ArteryNetwork.outlet_bc(artery, self.dt, T=T)
                else:
                    #todo: bifurcation outlet condition
                    pass
                
                artery.solve(lw, U_in, U_out, self.t, self.dt, save, i-1, T=self.T)
                
                if ArteryNetwork.cfl_condition(artery, self.dt) == False:
                    raise ValueError(
                            "CFL condition not fulfilled at time %e. Reduce \
time step size." % (self.t))
                    sys.exit(1)  
            
            self.timestep()
            
            if self.t % (self.tf/10) < self.dt:
                print "Progress {:}%".format(self._progress)
                self._progress += 10
                
        artery.P = artery.p(artery.U[0,:,:])/133 # convert to mmHg
                
            
    def dump_results(self, suffix, data_dir):
        for artery in self.arteries:
            artery.dump_results(suffix, data_dir)
                       
                       
    def spatial_plots(self, suffix, plot_dir, n):
        for artery in self.arteries:
            artery.spatial_plots(suffix, plot_dir, n)
        
        
    def time_plots(self, suffix, plot_dir, n):
        time = np.linspace(self.tf-self.T, self.tf, self.ntr)
        for artery in self.arteries:
            artery.time_plots(suffix, plot_dir, n, time)
            
    
    def s3d_plots(self, suffix, plot_dir):
        time = np.linspace(self.tf-self.T, self.tf, self.ntr)
        for artery in self.arteries:
            artery.p3d_plot(suffix, plot_dir, time)
            artery.q3d_plot(suffix, plot_dir, time)

            
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
    def tf(self):
        return self._tf
        
        
    @property
    def T(self):
        return self._T
        
        
    @property
    def tc(self):
        return self._tc
        
        
    @property
    def t(self):
        return self._t
        
        
    @property
    def ntr(self):
        return self._ntr
        
        
    @property
    def dtr(self):
        return self._dtr