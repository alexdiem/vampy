# -*- coding: utf-8 -*-

from vampy.artery import *
from scipy.interpolate import interp1d


eps = 1e-5


def equal(a, b):
    return True if abs(a-b) < eps else False


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
    Ru = a['Rd'][0] / rc # artery radius upstream
    Rd = a['Rd'][0] / rc # artery radius downstream
    kc = rho*qc**2/rc**4
    k = (a['k1']/kc, a['k2']*rc, a['k3']/kc) # elasticity model parameters (Eh/r)
    pos = 0
    lam = a['lam'][0]
    return pos, Ru, Rd, lam, k, Re


def test_artery_init():
    pos, Ru, Rd, lam, k, Re = parameter()
    artery = Artery(pos, Ru, Rd, lam, k, Re)
    assert artery.pos == pos
    assert artery.Ru == Ru
    assert artery.Rd == Rd
    assert artery.L == Ru*lam
    assert artery.k == k
    assert artery.Re == Re
    
    
def test_initial_conditions():
    pos, Ru, Rd, lam, k, Re = parameter()
    artery = Artery(pos, Ru, Rd, lam, k, Re)
    artery.mesh(10, 10)
    u0 = 0.5
    artery.initial_conditions(u0)
    assert (artery.U0[1,:] == u0).all()
    assert (artery.U0[0,:] == artery.A0).all()
    
    
def test_mesh():
    pos, Ru, Rd, lam, k, Re = parameter()
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
    
    
#def test_set_time():
#    R, lam, sigma, rho, mu, beta, ntr = parameter()
#    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
#    nt = 100
#    dt = 0.0001
#    artery.set_time(nt, dt)
#    assert artery.nt == nt
#    assert artery.dt == dt
#    assert artery.T == 0.0
#    
#    
#def test_p():
#    R, lam, sigma, rho, mu, beta, ntr = parameter()
#    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
#    assert artery.p(artery.A0) == artery.beta * (np.sqrt(artery.A0) -\
#                np.sqrt(artery.A0))
#    assert artery.p(100*artery.A0) == artery.beta * (np.sqrt(100*artery.A0) -\
#                np.sqrt(artery.A0))
#                
#                
#def test_wave_speed():
#    R, lam, sigma, rho, mu, beta, ntr = parameter()
#    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
#    assert artery.wave_speed(artery.A0) == np.sqrt(artery.beta *\
#                np.sqrt(artery.A0)/(2*artery.rho))
#    assert artery.wave_speed(100*artery.A0) == np.sqrt(artery.beta *\
#                np.sqrt(100*artery.A0)/(2*artery.rho))
#                
#                
#def test_inlet_bc():
#    R, lam, sigma, rho, mu, beta, ntr = parameter()
#    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
#    artery.mesh(10) 
#    u0 = 0.43
#    artery.initial_conditions(u0)
#    artery.set_time(100, 0.1)
#    u = 0.45
#    a_in, u_in = artery.inlet_bc(artery.U0, u)
#    assert u_in == u
#    w2 = u0 - 4*artery.wave_speed(artery.A0)
#    assert a_in == (u - w2)**4/64 * (artery.rho/artery.beta)**2
#    
#    
#def test_outlet_bc():
#    R, lam, sigma, rho, mu, beta, ntr = parameter()
#    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
#    artery.mesh(10) 
#    u0 = 0.43
#    artery.initial_conditions(u0)
#    artery.set_time(100, 0.1)
#    a = artery.A0
#    a_out, u_out = artery.outlet_bc(artery.U0, a)
#    assert a_out == a
#    w1 = u0 + 4*artery.wave_speed(artery.A0)
#    assert u_out == w1 - 4*a**(1/4.) * np.sqrt(artery.beta/(2*artery.rho))
#    
#    
#def test_cfl_condition():
#    R, lam, sigma, rho, mu, beta, ntr = parameter()
#    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
#    c = artery.wave_speed(artery.A0)
#    artery.mesh(10)
#    u0 = 0.43
#    artery.initial_conditions(u0)
#    dt = 1e-3
#    nt = 100
#    artery.set_time(nt, dt)
#    cfl = artery.cfl_condition(artery.U0[:,0])
#    v = (artery.U0[1,0] + c, artery.U0[1,0] - c)
#    left = artery.dt/artery.dx
#    right = np.power(np.absolute(v), -1)
#    assert cfl != (left > right).any()
#    
#    
#def test_F():
#    R, lam, sigma, rho, mu, beta, ntr = parameter()
#    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
#    artery.mesh(10)
#    u0 = 0.43
#    artery.initial_conditions(u0)
#    f = artery.F(artery.U0)
#    a, u = artery.U0
#    assert (f == np.array([a*u, np.power(u,2) + artery.p(a)/artery.rho])).all()
#    
#    
#def test_S():
#    R, lam, sigma, rho, mu, beta, ntr = parameter()
#    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
#    artery.mesh(10)
#    u0 = 0.43
#    artery.initial_conditions(u0)
#    s = artery.S(artery.U0)
#    a, u = artery.U0
#    assert (s == np.array([u*0, -8*np.pi*artery.mu/artery.rho * u/a])).all()
#    
#    
#def sine_inlet(t):
#    return interp1d(t, np.sin(t))
#    
#    
#def test_solve():
#    R, lam, sigma, rho, mu, beta, ntr = parameter()
#    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
#    nx = 10    
#    artery.mesh(nx)
#    u0 = 0.43
#    artery.initial_conditions(u0)
#    T = 0.85
#    nt = 100
#    artery.set_time(nt, 0.001, T)
#    time = np.linspace(0,T,1000)
#    u_in = sine_inlet(time)
#    a, u = artery.solve(u0, u_in, 0.0, T)
#    assert a.shape == (nt, nx)
#    assert u.shape == (nt, nx)