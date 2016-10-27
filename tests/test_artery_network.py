# -*- coding: utf-8 -*-

from vampy.artery_network import *
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
    Ru = a['Rd'] / rc # artery radius upstream
    Rd = a['Rd'] / rc # artery radius downstream
    kc = rho*qc**2/rc**4
    k = (a['k1']/kc, a['k2']*rc, a['k3']/kc) # elasticity model parameters (Eh/r)
    pos = 0
    lam = a['lam']
    p0 = 0
    depth = a['depth']
    ntr = 200
    return Ru, Rd, lam, k, rho, nu, p0, depth, ntr, Re
    

def test_an_init():
    Ru, Rd, lam, k, rho, nu, p0, depth, ntr, Re = parameter()
    an = ArteryNetwork(Ru, Rd, lam, k, rho, nu, p0, depth, ntr, Re)
    assert an.depth == depth
    assert len(an.arteries) == 2**depth - 1
    depth = 5
    an = ArteryNetwork(Ru, Rd, lam, k, rho, nu, p0, depth, ntr, Re)
    assert an.depth == depth
    assert len(an.arteries) == 2**depth - 1
    
    
def test_setup_arteries():
    # test setup_arteries separately
    Ru, Rd, lam, k, rho, nu, p0, depth, ntr, Re = parameter()
    an = ArteryNetwork(Ru, Rd, lam, k, rho, nu, p0, depth, ntr, Re)
    assert len(an.arteries) == len(Ru)
    
    
def test_setup_arteries_ab():
    # test setup_arteries separately
    Ru, Rd, lam, k, rho, nu, p0, depth, ntr, Re = parameter()
    a = b = 0.9
    an = ArteryNetwork(Ru[0], Rd[0], lam[0], k, rho, nu, p0, depth, ntr, Re, a=a, b=b)
    Ru = Ru[0]
    lam = lam[0]
    assert an.arteries[0].Ru == Ru
    assert an.arteries[1].Ru == a*Ru
    assert an.arteries[2].Ru == b*Ru
    assert an.arteries[0].L == Ru*lam
    assert an.arteries[1].L == a*Ru*lam
    assert an.arteries[2].L == b*Ru*lam
    
    
def test_initial_conditions():
    Ru, Rd, lam, k, rho, nu, p0, depth, ntr, Re = parameter()
    an = ArteryNetwork(Ru, Rd, lam, k, rho, nu, p0, depth, ntr, Re)
    an.mesh(0.1)
    u0 = 0.34
    an.initial_conditions(u0)
    for artery in an.arteries:
        assert (artery.U0[1,:] == u0).all()
        assert (artery.U0[0,:] == artery.A0).all()
        
        
def test_mesh():
    Ru, Rd, lam, k, rho, nu, p0, depth, ntr, Re = parameter()
    an = ArteryNetwork(Ru, Rd, lam, k, rho, nu, p0, depth, ntr, Re)
    dx = 0.1   
    an.mesh(dx)
    an.initial_conditions(0.3)
    for artery in an.arteries:
        assert hasattr(artery.A0, 'shape')
        assert hasattr(artery.f, 'shape')
        assert hasattr(artery.df, 'shape')
        assert hasattr(artery.xgrad, 'shape')
        assert hasattr(artery.U, 'shape')
        assert hasattr(artery.P, 'shape')
        assert hasattr(artery.U0, 'shape')
        
        
def test_set_time():
    Ru, Rd, lam, k, rho, nu, p0, depth, ntr, Re = parameter()
    an = ArteryNetwork(Ru, Rd, lam, k, rho, nu, p0, depth, ntr, Re)
    dt = 0.001
    T = 0.85
    an.set_time(dt, T)
    for artery in an.arteries:
        assert artery.delta > 0
        
        
#def test_solve():
#    R, a, b, lam, sigma, rho, mu, depth, beta, ntr = parameter()
#    an = ArteryNetwork(R, a, b, lam, sigma, rho, mu, depth, beta=beta, ntr=100)
#    nx = 10
#    an.mesh(nx)
#    u0 = 0.3
#    an.initial_conditions(u0)
#    T = 0.85
#    an.set_time(100, 0.001, T)
#    time = np.linspace(0,0.85,1000)
#    u_in = sine_inlet(time)
#    An, Un = an.solve(u0, u_in, 0.0, T)
#    assert len(An) == 2**depth-1
#    assert len(Un) == 2**depth-1
#    for i in range(len(An)):    
#        assert len(An[i]) == ntr
#        assert len(Un[i]) == ntr
#    