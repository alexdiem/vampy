# -*- coding: utf-8 -*-

from vampy.artery import *
from scipy.interpolate import interp1d


eps = 1e-5


def equal(a, b):
    return True if abs(a-b) < eps else False


def parameter():
    R = 1.0
    lam = 20
    sigma = 0.5
    rho = 1.06e6
    mu = 4.88
    beta = 22967.4
    ntr = 100
    return R, lam, sigma, rho, mu, beta, ntr


def test_artery_init():
    R, lam, sigma, rho, mu, beta, ntr = parameter()
    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
    assert artery.R == R
    assert artery.L == R*lam
    
    
def test_initial_conditions():
    R, lam, sigma, rho, mu, beta, ntr = parameter()
    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
    artery.mesh(10)
    u0 = 0.5
    artery.initial_conditions(u0)
    assert (artery.U0[1,:] == u0).all()
    assert (artery.U0[0,:] == artery.R**2 * np.pi).all()
    
    
def test_tf():
    R, lam, sigma, rho, mu, beta, ntr = parameter()
    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
    nt = 100
    dt = 0.001
    artery.set_time(nt, dt)
    assert artery.tf() == nt*dt
    
    
def test_timestep():
    R, lam, sigma, rho, mu, beta, ntr = parameter()
    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
    dt = 0.01
    artery.set_time(100, dt)
    for i in range(10):
        assert equal(artery.t, i*dt)
        artery.timestep()
        
    
def test_mesh():
    R, lam, sigma, rho, mu, beta, ntr = parameter()
    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
    nx = 10    
    artery.mesh(nx)
    assert len(artery.x) == nx
    
    
def test_set_time():
    R, lam, sigma, rho, mu, beta, ntr = parameter()
    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
    nt = 100
    dt = 0.0001
    artery.set_time(nt, dt)
    assert artery.nt == nt
    assert artery.dt == dt
    assert artery.T == 0.0
    
    
def test_p():
    R, lam, sigma, rho, mu, beta, ntr = parameter()
    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
    assert artery.p(artery.A0) == artery.beta * (np.sqrt(artery.A0) -\
                np.sqrt(artery.A0))
    assert artery.p(100*artery.A0) == artery.beta * (np.sqrt(100*artery.A0) -\
                np.sqrt(artery.A0))
                
                
def test_wave_speed():
    R, lam, sigma, rho, mu, beta, ntr = parameter()
    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
    assert artery.wave_speed(artery.A0) == np.sqrt(artery.beta *\
                np.sqrt(artery.A0)/(2*artery.rho))
    assert artery.wave_speed(100*artery.A0) == np.sqrt(artery.beta *\
                np.sqrt(100*artery.A0)/(2*artery.rho))
                
                
def test_inlet_bc():
    R, lam, sigma, rho, mu, beta, ntr = parameter()
    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
    artery.mesh(10) 
    u0 = 0.43
    artery.initial_conditions(u0)
    artery.set_time(100, 0.1)
    u = 0.45
    a_in, u_in = artery.inlet_bc(artery.U0, u)
    assert u_in == u
    w2 = u0 - 4*artery.wave_speed(artery.A0)
    assert a_in == (u - w2)**4/64 * (artery.rho/artery.beta)**2
    
    
def test_outlet_bc():
    R, lam, sigma, rho, mu, beta, ntr = parameter()
    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
    artery.mesh(10) 
    u0 = 0.43
    artery.initial_conditions(u0)
    artery.set_time(100, 0.1)
    a = artery.A0
    a_out, u_out = artery.outlet_bc(artery.U0, a)
    assert a_out == a
    w1 = u0 + 4*artery.wave_speed(artery.A0)
    assert u_out == w1 - 4*a**(1/4.) * np.sqrt(artery.beta/(2*artery.rho))
    
    
def test_cfl_condition():
    R, lam, sigma, rho, mu, beta, ntr = parameter()
    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
    c = artery.wave_speed(artery.A0)
    artery.mesh(10)
    u0 = 0.43
    artery.initial_conditions(u0)
    dt = 1e-3
    nt = 100
    artery.set_time(nt, dt)
    cfl = artery.cfl_condition(artery.U0[:,0])
    v = (artery.U0[1,0] + c, artery.U0[1,0] - c)
    left = artery.dt/artery.dx
    right = np.power(np.absolute(v), -1)
    assert cfl != (left > right).any()
    
    
def test_F():
    R, lam, sigma, rho, mu, beta, ntr = parameter()
    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
    artery.mesh(10)
    u0 = 0.43
    artery.initial_conditions(u0)
    f = artery.F(artery.U0)
    a, u = artery.U0
    assert (f == np.array([a*u, np.power(u,2) + artery.p(a)/artery.rho])).all()
    
    
def test_S():
    R, lam, sigma, rho, mu, beta, ntr = parameter()
    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
    artery.mesh(10)
    u0 = 0.43
    artery.initial_conditions(u0)
    s = artery.S(artery.U0)
    a, u = artery.U0
    assert (s == np.array([u*0, -8*np.pi*artery.mu/artery.rho * u/a])).all()
    
    
def sine_inlet(t):
    return interp1d(t, np.sin(t))
    
    
def test_solve():
    R, lam, sigma, rho, mu, beta, ntr = parameter()
    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
    nx = 10    
    artery.mesh(nx)
    u0 = 0.43
    artery.initial_conditions(u0)
    T = 0.85
    nt = 100
    artery.set_time(nt, 0.001, T)
    time = np.linspace(0,T,1000)
    u_in = sine_inlet(time)
    a, u = artery.solve(u0, u_in, 0.0, T)
    assert a.shape == (nt, nx)
    assert u.shape == (nt, nx)