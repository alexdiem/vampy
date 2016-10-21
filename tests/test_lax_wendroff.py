# -*- coding: utf-8 -*-


from vampy.lax_wendroff import *
from vampy.artery import *
import numpy as np


def create_lw():
    theta = 0.1
    gamma = 0.1
    nx = 10
    return LaxWendroff(theta, gamma, nx)
    
    
def F(U, **kwargs):
    return np.zeros_like(U)
    

def S(U, **kwargs):
    return np.zeros_like(U)


def test_init():
    theta = 0.1
    gamma = 0.1
    nx = 10
    lw = LaxWendroff(theta, gamma, nx)
    assert lw.theta == theta
    assert lw.gamma == gamma
    assert lw.nx == nx
    
    
def test_solve():
    lw = create_lw()
    U0 = np.zeros((2,lw.nx))
    U_in = U_out = np.array([0,0])
    U1 = lw.solve(U0, U_in, U_out, F, S)
    assert (U1 == U0).all()
    