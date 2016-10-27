# -*- coding: utf-8 -*-

from vampy.artery import *
from vampy.lax_wendroff import *
from scipy.interpolate import interp1d


def parameter():
    # read config file
    f, a, s = utils.read_config("../bifurcation.cfg") 
    
    # nondimensionalisation parameters
    rc = a['rc'] 
    qc = a['qc']  
    nu = a['nu']
    rho = a['rho']
    Re = qc/(nu*rc)
    nondim = [rc, qc, Re]
    
    # assign parameters
    nu = nu*rc/qc
    T = s['T'] * qc / rc**3
    Ru = a['Rd'][0] / rc # artery radius upstream
    Rd = a['Rd'][0] / rc # artery radius downstream
    kc = rho*qc**2/rc**4
    k = (a['k1']/kc, a['k2']*rc, a['k3']/kc) # elasticity model parameters (Eh/r)
    pos = 0
    lam = a['lam'][0]
    return pos, Ru, Rd, lam, k, Re, nu, T


def test_artery_init():
    pos, Ru, Rd, lam, k, Re, nu, T = parameter()
    artery = Artery(pos, Ru, Rd, lam, k, Re)
    assert artery.pos == pos
    assert artery.Ru == Ru
    assert artery.Rd == Rd
    assert artery.L == Ru*lam
    assert artery.k == k
    assert artery.Re == Re
    
    
def test_initial_conditions():
    pos, Ru, Rd, lam, k, Re, nu, T = parameter()
    artery = Artery(pos, Ru, Rd, lam, k, Re)
    artery.mesh(10, 10)
    u0 = 0.5
    artery.initial_conditions(u0)
    assert (artery.U0[1,:] == u0).all()
    assert (artery.U0[0,:] == artery.A0).all()
    
    
def test_mesh():
    pos, Ru, Rd, lam, k, Re, nu, T = parameter()
    artery = Artery(pos, Ru, Rd, lam, k, Re)
    nx = 10    
    l = np.linspace(0, Ru*lam, nx)
    dx = l[1] - l[0]
    ntr = 20
    artery.mesh(dx, ntr)
    assert len(artery.A0) == nx
    assert len(artery.f) == nx
    assert len(artery.df) == nx
    assert len(artery.xgrad) == nx
    assert artery.U0.shape == (2, nx)
    assert artery.U.shape == (2, ntr, nx)
    assert artery.P.shape == (ntr, nx)
    
    
def test_boundary_layer_thickness():
    pos, Ru, Rd, lam, k, Re, nu, T = parameter()
    artery = Artery(pos, Ru, Rd, lam, k, Re)
    blt = np.sqrt(nu*T/(2*np.pi))
    artery.boundary_layer_thickness(nu, T)
    assert blt == artery.delta
    
    
def test_p():
    pos, Ru, Rd, lam, k, Re, nu, T = parameter()
    artery = Artery(pos, Ru, Rd, lam, k, Re)
    artery.mesh(0.1, 20)
    p0 = artery.f[0] * (1 - np.sqrt(artery.A0[0]/artery.A0[0]))
    p = artery.f * (1 - np.sqrt(artery.A0/artery.A0))
    assert artery.p(artery.A0[0], j=0) == p0
    assert (artery.p(artery.A0) == p).all()
                
                
def test_wave_speed():
    pos, Ru, Rd, lam, k, Re, nu, T = parameter()
    artery = Artery(pos, Ru, Rd, lam, k, Re)
    artery.mesh(0.1, 20)
    c = -np.sqrt(0.5 * artery.f * np.sqrt(artery.A0/artery.A0))
    assert (c == artery.wave_speed(artery.A0)).all()
                
                
def test_F():
    pos, Ru, Rd, lam, k, Re, nu, T = parameter()
    artery = Artery(pos, Ru, Rd, lam, k, Re)
    artery.mesh(0.1, 20)
    artery.initial_conditions(0.43)
    j = 0
    k = 5
    f = artery.F(artery.U0[:,j:k], j=j, k=k)
    a, q = artery.U0[:,j:k]
    assert (f == np.array([q, q*q/a + artery.f[j:k] * np.sqrt(artery.A0[j:k]*a)])).all()
    
    
def test_S():
    pos, Ru, Rd, lam, k, Re, nu, T = parameter()
    artery = Artery(pos, Ru, Rd, lam, k, Re)
    artery.mesh(0.1, 20)
    artery.initial_conditions(0.43)
    artery.boundary_layer_thickness(nu, T)
    j = 0
    k = 5
    s = artery.S(artery.U0[:,j:k], j=j, k=k)
    a, q = artery.U0[:,j:k]
    R = np.sqrt(a/np.pi)
    s0 = np.zeros((2,k))
    s0[0,:].fill(0.0)
    s0[1,:] =-2*np.pi*R*q/(artery.Re*artery.delta*a) +\
                (2*np.sqrt(a) * (np.sqrt(np.pi)*artery.f[j:k] +\
                np.sqrt(artery.A0[j:k])*artery.df[j:k]) -\
                a*artery.df[j:k]) * artery.xgrad[j:k]
    assert (s == s0).all()
    
    
def test_dBdx():
    pos, Ru, Rd, lam, k, Re, nu, T = parameter()
    artery = Artery(pos, Ru, Rd, lam, k, Re)
    artery.mesh(0.1, 20)
    artery.initial_conditions(0.43)
    artery.boundary_layer_thickness(nu, T)
    xi = artery.A0[-1]
    dBdx = artery.dBdx(artery.L, xi)
    assert dBdx == (2*np.sqrt(xi) * (np.sqrt(np.pi)*artery.f[-1] +\
                    np.sqrt(artery.A0[-1])*artery.df[-1]) -\
                    xi*artery.df[-1]) * artery.xgrad[-1]
                    
                    
def test_dBdxi():
    pos, Ru, Rd, lam, k, Re, nu, T = parameter()
    artery = Artery(pos, Ru, Rd, lam, k, Re)
    artery.mesh(0.1, 20)
    artery.initial_conditions(0.43)
    artery.boundary_layer_thickness(nu, T)
    xi = artery.A0[-1]
    dBdxi = artery.dBdxi(artery.L, xi)
    assert dBdxi == artery.f[-1]/2 * np.sqrt(artery.A0[-1]/xi)
    
    
def test_dBdxdxi():
    pos, Ru, Rd, lam, k, Re, nu, T = parameter()
    artery = Artery(pos, Ru, Rd, lam, k, Re)
    artery.mesh(0.1, 20)
    artery.initial_conditions(0.43)
    artery.boundary_layer_thickness(nu, T)
    xi = artery.A0[-1]
    dBdxdxi = artery.dBdxdxi(artery.L, xi)
    assert dBdxdxi == (1/(2*np.sqrt(xi)) * (artery.f[-1]*np.sqrt(np.pi) +\
                    artery.df[-1]*np.sqrt(artery.A0[-1])) -\
                    artery.df[-1]) * artery.xgrad[-1]
                    
                    
def test_dFdxi2():
    pos, Ru, Rd, lam, k, Re, nu, T = parameter()
    artery = Artery(pos, Ru, Rd, lam, k, Re)
    artery.mesh(0.1, 20)
    artery.initial_conditions(0.43)
    artery.boundary_layer_thickness(nu, T)
    xi1 = xi2 = artery.A0[-1]
    R0 = np.sqrt(xi1/np.pi)
    dFdxi2 = artery.dFdxi2(artery.L, xi1, xi2)
    assert dFdxi2 == 2*np.pi*R0*xi1/(artery.delta*artery.Re*xi2*xi2)
    
    
def test_dFdxi1():
    pos, Ru, Rd, lam, k, Re, nu, T = parameter()
    artery = Artery(pos, Ru, Rd, lam, k, Re)
    artery.mesh(0.1, 20)
    artery.initial_conditions(0.43)
    artery.boundary_layer_thickness(nu, T)
    xi = artery.A0[-1]
    R0 = np.sqrt(xi/np.pi)
    dFdxi1 = artery.dFdxi1(artery.L, xi)
    assert dFdxi1 == -2*np.pi*R0/(artery.delta*artery.Re*xi)
    
    
def test_dpdx():
    pos, Ru, Rd, lam, k, Re, nu, T = parameter()
    artery = Artery(pos, Ru, Rd, lam, k, Re)
    artery.mesh(0.1, 20)
    artery.initial_conditions(0.43)
    artery.boundary_layer_thickness(nu, T)
    xi = artery.A0[-1]
    dpdx = artery.dpdx(artery.L, xi)
    assert dpdx == artery.f[-1]/2 * np.sqrt(artery.A0[-1]/xi**3)
    

def test_solve():
    pos, Ru, Rd, lam, k, Re, nu, T = parameter()
    artery = Artery(pos, Ru, Rd, lam, k, Re)
    dt = 0.0001
    dx = 0.1
    artery.mesh(dx, 20)
    artery.initial_conditions(0.43)
    artery.boundary_layer_thickness(nu, T)
    U0 = artery.U0
    lw = LaxWendroff(dt/dx, dt/2, artery.nx)
    U_in = np.array([artery.A0[0], 0.43])
    U_out = np.array([artery.A0[-1], 0.43])
    artery.solve(lw, U_in, U_out, False, 0)
    assert U0.shape == artery.U0.shape