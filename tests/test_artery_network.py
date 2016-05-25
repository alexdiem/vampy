# -*- coding: utf-8 -*-

from pyFDM.artery_network import *
from scipy.interpolate import interp1d


def parameter():
    R = 1.0
    a = 0.9
    b = 0.5
    lam = 20
    sigma = 0.5
    rho = 1.06e6
    mu = 4.88
    depth = 3
    beta = 22967.4
    ntr = 100
    return R, a, b, lam, sigma, rho, mu, depth, beta, ntr
    

def test_an_init():
    R, a, b, lam, sigma, rho, mu, depth, beta, ntr = parameter()
    an = ArteryNetwork(R, a, b, lam, sigma, rho, mu, depth, beta=beta, ntr=100)
    assert an.depth == depth
    assert len(an.arteries) == 2**depth - 1
    depth = 5
    an = ArteryNetwork(R, a, b, lam, sigma, rho, mu, depth, beta=beta, ntr=100)
    assert an.depth == depth
    assert len(an.arteries) == 2**depth - 1
    
    
def test_setup_arteries():
    R, a, b, lam, sigma, rho, mu, depth, beta, ntr = parameter()
    an = ArteryNetwork(R, a, b, lam, sigma, rho, mu, depth, beta=beta, ntr=100)
    assert an.arteries[0].R == R
    assert an.arteries[1].R == a*R
    assert an.arteries[2].R == b*R
    assert an.arteries[0].L == R*lam
    assert an.arteries[1].L == a*R*lam
    assert an.arteries[2].L == b*R*lam
    
    
def test_initial_conditions():
    R, a, b, lam, sigma, rho, mu, depth, beta, ntr = parameter()
    an = ArteryNetwork(R, a, b, lam, sigma, rho, mu, depth, beta=beta, ntr=100)
    an.mesh(10)
    u0 = 0.34
    an.initial_conditions(u0)
    for artery in an.arteries:
        assert (artery.U0[1,:] == u0).all()
        assert (artery.U0[0,:] == artery.R**2 * np.pi).all()
        
        
def test_mesh():
    R, a, b, lam, sigma, rho, mu, depth, beta, ntr = parameter()
    an = ArteryNetwork(R, a, b, lam, sigma, rho, mu, depth, beta=beta, ntr=100)
    nx = 10    
    an.mesh(nx)
    an.initial_conditions(0.3)
    for artery in an.arteries:
        assert len(artery.U0[1,:]) == nx
        
        
def test_set_time():
    R, a, b, lam, sigma, rho, mu, depth, beta, ntr = parameter()
    an = ArteryNetwork(R, a, b, lam, sigma, rho, mu, depth, beta=beta, ntr=100)
    nt = 100
    dt = 0.001
    an.set_time(nt, dt)
    for artery in an.arteries:
        assert artery.nt == nt
        assert artery.dt == dt
    T = 0.3
    an.set_time(nt, dt, T)
    for artery in an.arteries:
        assert artery.T == T
        
        
def sine_inlet(t):
    return interp1d(t, np.sin(t))
        
        
def test_solve():
    R, a, b, lam, sigma, rho, mu, depth, beta, ntr = parameter()
    an = ArteryNetwork(R, a, b, lam, sigma, rho, mu, depth, beta=beta, ntr=100)
    nx = 10
    an.mesh(nx)
    u0 = 0.3
    an.initial_conditions(u0)
    T = 0.85
    an.set_time(100, 0.001, T)
    time = np.linspace(0,0.85,1000)
    u_in = sine_inlet(time)
    An, Un = an.solve(u0, u_in, 0.0, T)
    assert len(An) == 2**depth-1
    assert len(Un) == 2**depth-1
    for i in range(len(An)):    
        assert len(An[i]) == ntr
        assert len(Un[i]) == ntr
    