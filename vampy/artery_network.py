# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

from scipy import linalg

from artery import Artery
from lax_wendroff import LaxWendroff
import utils

import sys


class ArteryNetwork(object):
    """
    Class representing a network of arteries.
    """
    
    
    def __init__(self, Ru, Rd, a, b, lam, rho, nu, delta, depth, **kwargs):
        self._depth = depth
        self._arteries = []
        self.setup_arteries(Ru, Rd, a, b, lam, rho, nu, delta, **kwargs)
        self._t = 0.0
        self._ntr = kwargs['ntr']
        self._progress = 10
        nondim = kwargs['nondim']
        self._rc = nondim[0]
        self._qc = nondim[1]
        self._rho = rho
        self._Re = nondim[2]
        
        
    def setup_arteries(self, Ru, Rd, a, b, lam, rho, nu, delta, **kwargs):
        pos = 0
        self.arteries.append(Artery(pos, Ru, Rd, lam, rho, nu, delta, depth=0,
                                    **kwargs)) 
        pos += 1
        radii_u = [Ru]
        radii_d = [Rd]
        for i in range(1,self.depth):
            new_radii_u = []
            new_radii_d = []
            for i in range(len(radii_u)):    
                ra_u = radii_u[i] * a
                rb_u = radii_u[i] * b
                ra_d = radii_d[i] * a
                rb_d = radii_d[i] * b
                self.arteries.append(Artery(pos, ra_u, ra_d, lam, rho, nu, delta,
                                            depth=i, **kwargs))
                pos += 1
                self.arteries.append(Artery(pos, rb_u, ra_d, lam, rho, nu, delta,
                                            depth=i, **kwargs))
                pos += 1
                new_radii_u.append(ra_u)
                new_radii_u.append(rb_u)
                new_radii_d.append(ra_d)
                new_radii_d.append(rb_d)
            radii_u = new_radii_u
            radii_d = new_radii_d
            
            
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
    def inlet_bc(artery, q_in, in_t, dt):
        q_0_np = q_in(in_t-dt/2) # q_0_n+1/2
        q_0_n1 = q_in(in_t) # q_0_n+1
        U_0_n = artery.U0[:,0] # U_0_n
        U_1_n = artery.U0[:,1]
        U_12_np = (U_1_n+U_0_n)/2 -\
                    dt*(artery.F(U_1_n, j=1)-artery.F(U_0_n, j=0))/(2*artery.dx) +\
                    dt*(artery.S(U_1_n, j=1)+artery.S(U_0_n, j=0))/4 # U_1/2_n+1/2
        q_12_np = U_12_np[1] # q_1/2_n+1/2
        a_0_n1 = U_0_n[0] - 2*dt*(q_12_np - q_0_np)/artery.dx
        return np.array([a_0_n1, q_0_n1])
     
    
    @staticmethod
    def outlet_bc(artery, dt, rc, qc, rho):
        R1 = 13900*rc**4/(qc*rho) # olufsen 4100
        R2 = 25300*rc**4/(qc*rho) # olufsen 1900
        Ct = 1.3384e-6*rho*qc**2/rc**7 # 8.7137e-6
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
    def jacobian(x, parent, d1, d2, theta, gamma):
        M12 = parent.L + parent.dx/2
        D1_12 = -d1.dx/2
        D2_12 = -d2.dx/2
        zeta7 = -parent.dpdx(parent.L, x[10])
        zeta10 = -parent.dpdx(parent.L, x[9])
        Dfr = np.zeros((18, 18)) # Jacobian
        Dfr[0,0] = Dfr[1,3] = Dfr[2,6] = Dfr[3,9] = Dfr[4,12] = Dfr[5,15] = -1
        Dfr[6,1] = Dfr[7,4] = Dfr[8,7] = Dfr[9,10] = Dfr[10,13] = Dfr[11,16] = -1
        Dfr[12,1] = Dfr[13,0] = -1
        Dfr[6,2] = Dfr[7,5] = Dfr[8,8] = Dfr[9,11] = Dfr[10,14] = Dfr[11,17] = 0.5
        Dfr[12,4] = Dfr[12,7] = Dfr[13,3] = Dfr[13,6] = 1.0
        Dfr[3,2] = -theta
        Dfr[4,5] = Dfr[5,8] = theta
        Dfr[0,2] = -2*theta*x[2]/x[11] + gamma*parent.dFdxi1(M12, x[11])
        Dfr[0,11] = theta * ((x[2]/x[11])**2 - parent.dBdxi(M12,x[11])) +\
                    gamma * (parent.dFdxi2(M12, x[2], x[11]) +\
                            parent.dBdxdxi(M12, x[11]))
                            
                            
        Dfr[1,5] = 2*theta*x[5]/x[14] + gamma*d1.dFdxi1(D1_12, x[14])
        Dfr[1,14] = theta * (-(x[5]/x[14])**2 + d1.dBdxi(D1_12,x[14])) +\
                    gamma * (d1.dFdxi2(D1_12, x[5], x[14]) +\
                            d1.dBdxdxi(D1_12, x[14]))
        Dfr[2,8] = 2*theta*x[8]/x[17] + gamma*d2.dFdxi1(D2_12, x[17])
        Dfr[2,17] = theta * (-(x[8]/x[17])**2 + d2.dBdxi(D2_12,x[17])) +\
                    gamma * (d2.dFdxi2(D2_12, x[8], x[17]) +\
                            d2.dBdxdxi(D2_12, x[17]))
        Dfr[14,10] = zeta7
        Dfr[14,13] = d1.dpdx(0.0, x[13])
        Dfr[15,10] = zeta7
        Dfr[15,16] = d2.dpdx(0.0, x[16])
        Dfr[16,9] = zeta10
        Dfr[16,12] = d1.dpdx(0.0, x[12])
        Dfr[17,9] = zeta10
        Dfr[17,15] = d2.dpdx(0.0, x[15])
        return Dfr
        

    @staticmethod
    def residuals(x, parent, d1, d2, theta, gamma):
        U_p_np = (parent.U0[:,-1] + parent.U0[:,-2])/2 +\
                gamma * (-(parent.F(parent.U0[:,-1], j=-1) -\
                parent.F(parent.U0[:,-2], j=-2))/parent.dx +\
                (parent.S(parent.U0[:,-1], j=-1) +\
                parent.S(parent.U0[:,-2], j=-2))/2)
        U_d1_np = (d1.U0[:,1] + d1.U0[:,0])/2 +\
                gamma * (-(d1.F(d1.U0[:,1], j=1) -\
                d1.F(d1.U0[:,0], j=0))/d1.dx + (d1.S(d1.U0[:,1], j=1) +\
                d1.S(d1.U0[:,0], j=0))/2)
        U_d2_np = (d2.U0[:,1] + d2.U0[:,0])/2 +\
                gamma * (-(d2.F(d2.U0[:,1], j=1) -\
                d2.F(d2.U0[:,0], j=0))/d2.dx + (d2.S(d2.U0[:,1], j=1) +\
                d2.S(d2.U0[:,0], j=0))/2)
        f_p_mp = utils.extrapolate(parent.L+parent.dx/2,
                [parent.L-parent.dx, parent.L], [parent.f[-2], parent.f[-1]])
        f_d1_mp = utils.extrapolate(-d1.dx/2, [d1.dx, 0.0],
                                    [d1.f[1], d1.f[0]])
        f_d2_mp = utils.extrapolate(-d2.dx/2, [d2.dx, 0.0],
                                    [d2.f[1], d2.f[0]])
        A0_p_mp = utils.extrapolate(parent.L+parent.dx/2,
                [parent.L-parent.dx, parent.L], [parent.A0[-2], parent.A0[-1]])
        A0_d1_mp = utils.extrapolate(-d1.dx/2, [d1.dx, 0.0],
                                     [d1.A0[1], d1.A0[0]])
        A0_d2_mp = utils.extrapolate(-d2.dx/2, [d2.dx, 0.0],
                                     [d2.A0[1], d2.A0[0]])
        R0_p_mp = np.sqrt(A0_p_mp/np.pi)
        R0_d1_mp = np.sqrt(A0_d1_mp/np.pi)
        R0_d2_mp = np.sqrt(A0_d2_mp/np.pi)
        B_p_mp = f_p_mp * np.sqrt(x[11]*A0_p_mp)
        B_d1_mp = f_d1_mp * np.sqrt(x[14]*A0_d1_mp)
        B_d2_mp = f_d2_mp * np.sqrt(x[17]*A0_d2_mp)
        k1 = parent.U0[1,-1] + theta * (parent.F(U_p_np, j=-1)[1]) +\
                gamma * (parent.S(U_p_np, j=-1)[1])
        k2 = d1.U0[1,0] - theta * (d1.F(U_d1_np, j=0)[1]) +\
                gamma * (d1.S(U_d1_np, j=0)[1])
        k3 = d2.U0[1,0] - theta * (d2.F(U_d2_np, j=0)[1]) +\
                gamma * (d2.S(U_d2_np, j=0)[1])
        k4 = parent.U0[0,-1] + theta*parent.F(U_p_np, j=-1)[0]
        k5 = d1.U0[0,0] - theta*d1.F(U_d1_np, j=0)[0]
        k6 = d2.U0[0,0] - theta*d2.F(U_d2_np, j=0)[0]
        k7 = U_p_np[1]/2
        k8 = U_d1_np[1]/2
        k9 = U_d2_np[1]/2
        k10 = U_p_np[0]/2
        k11 = U_d1_np[0]/2
        k12 = U_d2_np[0]/2
        k15a = -parent.f[-1] + d1.f[0]
        k15b = d1.f[0] * np.sqrt(d1.A0[0])
        k16a = -parent.f[-1] + d2.f[0]
        k16b = d2.f[0] * np.sqrt(d2.A0[0])
        k156 = parent.f[-1] * np.sqrt(parent.A0[-1])
        fr1 = k1 - x[0] - theta*(x[2]**2/x[11] + B_p_mp) +\
                gamma*(-2*np.pi*R0_p_mp*x[2]/(parent.delta*parent.Re*x[11]) +\
                parent.dBdx(parent.L+parent.dx/2, x[11]))
        fr2 = k2 - x[3] + theta*(x[5]**2/x[14] + B_d1_mp) +\
                gamma*(-2*np.pi*R0_d1_mp*x[5]/(d1.delta*d1.Re*x[14]) +\
                d1.dBdx(-d1.dx/2, x[14]))
        fr3 = k3 - x[6] + theta*(x[8]**2/x[17] + B_d2_mp) +\
                gamma*(-2*np.pi*R0_d2_mp*x[8]/(d2.delta*d2.Re*x[17]) +\
                d2.dBdx(-d2.dx/2, x[17]))
        fr4 = -x[9] - theta*x[2] + k4
        fr5 = -x[12] + theta*x[5] + k5
        fr6 = -x[15] + theta*x[8] + k6
        fr7 = -x[1] + x[2]/2 + k7
        fr8 = -x[4] + x[5]/2 + k8
        fr9 = -x[7] + x[8]/2 + k9
        fr10 = -x[10] + x[11]/2 + k10
        fr11 = -x[13] + x[14]/2 + k11
        fr12 = -x[16] + x[17]/2 + k12
        fr13 = -x[1] + x[4] + x[7]
        fr14 = -x[0] + x[3] + x[6]
        fr15 = k156/np.sqrt(x[10]) - k15b/np.sqrt(x[13]) + k15a
        fr16 = k156/np.sqrt(x[10]) - k16b/np.sqrt(x[16]) + k16a
        fr17 = k156/np.sqrt(x[9]) - k15b/np.sqrt(x[12]) + k15a
        fr18 = k156/np.sqrt(x[9]) - k16b/np.sqrt(x[15]) + k16a
        return np.array([fr1, fr2, fr3, fr4, fr5, fr6, fr7, fr8, fr9, fr10,
                         fr11, fr12, fr13, fr14, fr15, fr16, fr17, fr18])
        
    
    @staticmethod
    def bifurcation(parent, d1, d2, dt):
        theta = dt/parent.dx
        gamma = dt/2
        qm_p = parent.U0[1,-1]
        qm2_p = parent.U0[1,-2]
        am_p = parent.U0[0,-1]
        am2_p = parent.U0[0,-2]
        q0_d1 = d1.U0[1,0]
        q02_d1 = d1.U0[1,1]
        a0_d1 = d1.U0[0,0]
        a02_d1 = d1.U0[0,1]
        q0_d2 = d2.U0[1,0]
        q02_d2 = d2.U0[1,1]
        a0_d2 = d2.U0[0,0]
        a02_d2 = d2.U0[0,1]
        x0 = np.array([qm_p, (qm_p+qm2_p)/2, qm_p, q0_d1, (q0_d1+q02_d1)/2,
                        q0_d1, q0_d2, (q0_d2+q02_d2)/2, q0_d2, am_p,
                        (am_p+am2_p)/2, am_p, a0_d1, (a0_d1+a02_d1)/2, a0_d1,
                        a0_d2, (a0_d2+a02_d2)/2, a0_d2])
        k = 0
        while k < 1000:
            Dfr = ArteryNetwork.jacobian(x0, parent, d1, d2, theta, gamma)
            Dfr_inv = linalg.inv(Dfr)
            fr = ArteryNetwork.residuals(x0, parent, d1, d2, theta, gamma)
            x1 = x0 - np.dot(Dfr_inv, fr)
            if (abs(x1 - x0) < 1e-5).all():
                break
            k += 1
            np.copyto(x0, x1)
        return x1
                
    
    @staticmethod
    def cfl_condition(artery, dt):
        a = artery.U0[0,1]
        c = artery.wave_speed(a)
        u = artery.U0[1,1] / a
        v = [u + c, u - c]
        left = dt/artery.dx
        right = 1/np.absolute(v)
        return False if (left > right).any() else True
        
        
    def get_daughters(self, parent):
        p = parent.pos
        return self.arteries[p+1], self.arteries[p+2]
            
    
    def solve(self, q_in, p_out, T):
        tr = np.linspace(self.tf-self.T, self.tf, self.ntr)
        i = 0
        self.timestep()
        bc_in = np.zeros((len(self.arteries), 2))
        while self.t < self.tf:
            save = False  
            
            if i < self.ntr and (abs(tr[i]-self.t) < self.dtr or self.t >= self.tf-self.dt):
                save = True
                i += 1
                
            for artery in self.arteries:
                lw = LaxWendroff(artery.nx, artery.dx)
                
                if self.depth > 1 and artery.pos < 2**self.depth-1 - 2:
                    # need to sort out how the inlet is going to be applied
                    d1, d2 = self.get_daughters(artery)
                    x_out = ArteryNetwork.bifurcation(artery, d1, d2, self.dt)
                    U_out = np.array([x_out[9], x_out[0]])
                    bc_in[d1.pos] = np.array([x_out[12], x_out[3]])
                    bc_in[d2.pos] = np.array([x_out[15], x_out[6]])
                
                if artery.pos == 0:
                    # inlet boundary condition
                    if self.T > 0:
                        in_t = utils.periodic(self.t, self.T)
                    else:
                        in_t = self.t
                    U_in = ArteryNetwork.inlet_bc(artery, q_in, in_t, self.dt)
                else:
                    U_in = bc_in[artery.pos]
                    
                if artery.pos >= (len(self.arteries) - 2**(self.depth-1)):
                    # outlet boundary condition
                    U_out = ArteryNetwork.outlet_bc(artery, self.dt, self.rc,
                                                    self.qc, self.rho)
                
                artery.solve(lw, U_in, U_out, self.t, self.dt, save, i-1)
                
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
            artery.P = 85 + artery.P*self.rho*self.qc**2*760 / (1.01325*10**6*self.rc**4)
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