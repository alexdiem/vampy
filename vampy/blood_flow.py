# -*- coding: utf-8 -*-

import numpy as np


def extrapolate(x0, x, y):
    return y[0] + (y[1]-y[0]) * (x0 - x[0])/(x[1] - x[0])


def wave_speed(a, beta, rho):
    return np.sqrt(beta*np.sqrt(a)/(2*rho))
    
    
def bifurcation_system(u_prev):
    F = np.zeros(6)
    F[0] = calculate_characteristic(u_prev[0,0], u_prev[1,0:2], u_prev[1,0],
                                   beta, rho, dt, x[0:2])
    
    
def jacobian(A_p, u_p, A_1, u_1, A_2, u_2):
    J = np.zeros((6,6))
    J[0,:] = [-1, -A_p**(-3/4) * np.sqrt(beta_p/2/rho), 0, 0, 0, 0]
    J[1,:] = [0, 0, -1, A_1**(-3/4) * np.sqrt(beta_1/2/rho), 0, 0]
    J[2,:] = [0, 0, 0, 0, -1, A_2**(-3/4) * np.sqrt(beta_2/2/rho)]
    J[3,:] = [A_p, u_p, -A_1, -u_1, -A_2, -u_2]
    J[4,:] = [rho*u_p, beta_p/2*A_p**(-1/2), -rho*u_1, -beta_1/2*A_1**(-1/2),\
                0, 0]
    J[5,:] = [rho*u_p, beta_p/2*A_p**(-1/2), 0, 0, -rho*u_2,\
                -beta_2/2*A_2**(-1/2)]
    return J
    

def inlet_bc(u_prev, t, (u_in, beta, rho, x, dt)):
    w_2 = calculate_characteristic(u_prev[0,0], u_prev[1,0:2], u_prev[1,0],
                                   beta, rho, dt, x[0:2])
    a_in = (u_in(t) - w_2)**4 / 64 * (rho/beta)**2
    return np.array([a_in, u_in(t)])
     
    
def outlet_bc(u_prev, t, (a_out, beta, rho, x, dt)):
    w_1 = calculate_characteristic(u_prev[0,-1], u_prev[1,-2:], u_prev[1,-1],
                                   beta, rho, dt, x[-2:])
    u_out = w_1 - 4*a_out**(1/4) * np.sqrt(beta/(2*rho))
    return np.array([a_out, u_out])
    
    
def bifurcation_bc(p, d1, d2):
    uprev_p = p.get_uprev()
    F = bifurcation_system(u_prev)
    return 
        
    
def cfl_condition(u, (beta, rho, dt, dx)):
    c = wave_speed(u[0], beta, rho)
    v = (u[1] + c, u[1] - c)
    left = dt/dx
    right = np.power(np.absolute(v), -1)
    #print left, right
    return False if (left > right).any() else True