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
    
    
    def __init__(self, R, a, b, lam, rho, nu, delta, depth, **kwargs):
        self._depth = depth
        self._arteries = []
        self.setup_arteries(R, a, b, lam, rho, nu, delta, **kwargs)
        self._t = 0.0
        self._ntr = kwargs['ntr']
        self._progress = 10
        nondim = kwargs['nondim']
        self._rc = nondim[0]
        self._qc = nondim[1]
        self._rho = rho
        self._Re = nondim[2]
        
        
    def setup_arteries(self, R, a, b, lam, rho, nu, delta, **kwargs):
        pos = 0
        self.arteries.append(Artery(pos, R, lam, rho, nu, delta, **kwargs)) 
        pos += 1
        radii = [R]
        for i in range(1,self.depth):
            new_radii = []
            for radius in radii:    
                ra = radius * a
                rb = radius * b
                self.arteries.append(Artery(pos, ra, lam, rho, nu, delta, **kwargs))
                pos += 1
                self.arteries.append(Artery(pos, rb, lam, rho, nu, delta, **kwargs))
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
            
    
    def set_time(self, tf, dt, T=0.0, tc=1):
        self._dt = dt
        self._tf = tf
        self._dtr = tf/self.ntr
        self._T = T
        self._tc = tc
            
            
    def timestep(self):
        self._t += self.dt
            
    
    @staticmethod        
    def inlet_bc(artery, q_in, in_t, dt, **kwargs):
        q_0_np = q_in(in_t+dt/2) # q_0_n+1/2
        q_0_n1 = q_in(in_t) # q_0_n+1
        U_0_n = artery.U0[:,0] # U_0_n
        U_1_n = artery.U0[:,1]
        U_12_np = (U_1_n + U_0_n)/2 - dt*(artery.F(U_1_n, j=1) -\
                artery.F(U_0_n, j=0))/(2*artery.dx) + dt*(artery.S(U_1_n, j=1) +\
                artery.S(U_0_n, j=0))/4 # U_1/2_n+1/2
        q_12_np = U_12_np[1] # q_1/2_n+1/2
        a_0_n1 = U_12_np[0] - 2*dt*(q_12_np - q_0_np)/artery.dx
        return np.array([a_0_n1, q_0_n1])
     
    
    @staticmethod
    def outlet_bc(artery, dt, rc, qc, rho, **kwargs):
        R1 = 4100*rc**4/(qc*rho)
        R2 = 1900*rc**4/(qc*rho)
        Ct = 8.7137e-6*rho*qc**2/rc**7
        a_n = artery.U0[0,-1]
        q_n = artery.U0[1,-1]
        p_out = p_o = artery.p(a_n)[-1] # initial guess for p_out
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
        U_mm = artery.U0[:,-2] - dt/artery.dx * (artery.F(U_np_mm, j=-1) -\
                artery.F(U_np_mp, j=-1)) + dt/2 * (artery.S(U_np_mm, j=-1) +\
                artery.S(U_np_mp, j=-1))
        k = 0
        while k < 1000:
            p_old = p_o
            q_out = q_n + (p_o-p_out)/R1 + dt*(p_out/(R2*Ct) -\
                    q_n*(R1+R2)/(R2*Ct))/R1
            a_out = a_n - dt * (q_out - U_mm[1])/artery.dx
            p_o = artery.p(a_out)[-1]
            if abs(p_old - p_o) < 1e-7:
                break
            k += 1
        return np.array([a_out, q_out])
        
    
    @staticmethod
    def bifurcation_bc(artery, p, d1, d2):
        pass
    
    
    @staticmethod
    def cfl_condition(artery, dt):
        a = artery.U0[0,1]
        c = artery.wave_speed(a)
        u = artery.U0[1,1] / a
        v = [u + c, u - c]
        left = dt/artery.dx
        right = np.power(np.absolute(v), -1)
        return False if (left > right).any() else True
            
    
    def solve(self, q_in, p_out, T):
        tr = np.linspace(self.tf-self.T, self.tf, self.ntr)
        #tr = np.linspace(0, self.tf, self.ntr)
        i = 0
        
        self.timestep()
        
        
        
        while self.t < self.tf:
            save = False  
            
            if i < self.ntr and (abs(tr[i]-self.t) < self.dtr or self.t >= self.tf-self.dt):
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
                    U_in = self.inlet_bc(artery, q_in, in_t, self.dt, T=self.T)
                else:
                    #todo: bifurcation inlet boundary
                    pass
                if artery.pos >= (len(self.arteries) - 2**(self.depth-1)):
                    # outlet boundary condition
                    U_out = ArteryNetwork.outlet_bc(artery, self.dt, self.rc,
                                                    self.qc, self.rho, T=T)
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
                
        
        # redimensionalise
        for artery in self.arteries:
            artery.P = 80 + artery.P*self.rho*self.qc**2*760 / (1.01325*10**6*self.rc**4)
            artery.U[0,:,:] = artery.U[0,:,:] * self.rc**2  
            artery.U[1,:,:] = artery.U[1,:,:] * self.qc
                
            
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
        
    @property
    def rc(self):
        return self._rc
        
    @property
    def qc(self):
        return self._qc
        
    @property
    def rho(self):
        return self._rho
        
    @property
    def Re(self):
        return self._Re