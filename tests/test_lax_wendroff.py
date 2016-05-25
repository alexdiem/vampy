# -*- coding: utf-8 -*-


from pyFDM.lax_wendroff import *
from pyFDM.artery import *
import numpy as np


def test_init():
    nx = 10
    dx = 0.1
    lw = LaxWendroff(nx, dx)
    assert lw.nx == nx
    assert lw.dx == dx
    
    
def parameter():
    R = 1.0
    lam = 20
    sigma = 0.5
    rho = 1.06e6
    mu = 4.88
    beta = 22967.4
    ntr = 100
    return R, lam, sigma, rho, mu, beta, ntr
    
    
def test_solve():
    nx = 10
    dx = 0.1
    lw = LaxWendroff(nx, dx)
    U0 = np.zeros((2,nx))
    u0 = 0.3
    U_in = np.array([np.pi, u0])
    U_out = np.array([np.pi, u0])
    t = 0.0
    T = 0.85
    nt = 100
    dt = 1e-2
    R, lam, sigma, rho, mu, beta, ntr = parameter()
    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
    artery.mesh(nx)
    artery.initial_conditions(u0)
    artery.set_time(nt, dt, T)
    U1 = lw.solve(U0, U_in, U_out, t, T, artery.cfl_condition, artery.F,
                  artery.S, dt)
    assert U1.shape == U0.shape
    
    
def test_lax_wendroff():
    U_prev = np.array([[3.3, 3.4, 3.35], [0.1, 0.3, 0.2]])
    R, lam, sigma, rho, mu, beta, ntr = parameter()
    artery = Artery(R, lam, sigma, rho, mu, beta=beta, ntr=ntr)
    F_prev = artery.F(U_prev)
    S_prev = artery.S(U_prev)
    nx = 10
    dx = 0.1
    lw = LaxWendroff(nx, dx)
    U_np = lw.lax_wendroff(U_prev, F_prev, S_prev, artery.F, artery.S, 1e-4)
