# -*- coding: utf-8 -*-

import numpy as np


def extrapolate(x0, x, y):
    return y[0] + (y[1]-y[0]) * (x0 - x[0])/(x[1] - x[0])


def wave_speed(a, beta, rho):
    return np.sqrt(beta*np.sqrt(a)/(2*rho))
              

def inlet_bc(u_prev, t, (u_in, beta, rho, x, dt)):
    c_prev = wave_speed(u_prev[0,0], beta, rho)
    w_prev = u_prev[1,0:2] - 4*c_prev
    lam_2 = u_prev[1,0] - c_prev
    x_0 = x[0] - lam_2 * dt
    w_2 = extrapolate(x_0, x[0:2], w_prev)
    a_in = (u_in(t) - w_2)**4 / 64 * (rho/beta)**2
    return np.array([a_in, u_in(t)])
     
    
def outlet_bc(u_prev, t, (a_out, beta, rho, x, dt)):
    c_prev = wave_speed(u_prev[0,-1], beta, rho)
    w_prev = u_prev[1,-2:] + 4*c_prev
    lam_1 = u_prev[1,-1] + c_prev
    x_l = x[-1] - lam_1 * dt
    w_1 = extrapolate(x_l, x[-2:], w_prev)
    u_out = w_1 - 4*a_out**(1/4) * np.sqrt(beta/(2*rho))
    return np.array([a_out, u_out])
        
    
def cfl_condition(u, (beta, rho, dt, dx)):
    c = wave_speed(u[0], beta, rho)
    v = (u[1] + c, u[1] - c)
    left = dt/dx
    right = np.power(np.absolute(v), -1)
    #print left, right
    return False if (left > right).any() else True
        
    
def F(U, (beta, A0, rho)):
    a, u = U
    p = beta * (np.sqrt(a)-np.sqrt(A0))
    return np.array([a*u, np.power(u,2) + p/rho])
        
    
def S(U, (mu, rho)):
    a, u = U
    return np.array([u*0, -8*np.pi*mu/rho * u/a])