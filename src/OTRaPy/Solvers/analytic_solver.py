
import numpy as np 
from scipy.special import i0, k0 
from scipy import integrate 

def integrand_T(y, r, k, g, r0, h): 
    """
    Integrand in eqtn 1
    """
    x = np.sqrt(g/(k*h))
    I0_r = i0(x*r)
    I0_y = i0(x*y)
    K0_r = k0(x*r)
    K0_y = k0(x*y)
    if y <= r: 
        F = K0_r * I0_y
    else: 
        F = K0_y * I0_r
    
    return F * np.exp(-y**2 /r0**2) * y

def integrand_T_solve(y, r, ratio, r0, h): 
    """
    Integrand in eqtn 1
    """
    x = np.sqrt(ratio/(h))
    I0_r = i0(x*r)
    I0_y = i0(x*y)
    K0_r = k0(x*r)
    K0_y = k0(x*y)
    if y <= r: 
        F = K0_r * I0_y
    else: 
        F = K0_y * I0_r
    
    return F * np.exp(-y**2 /r0**2) * y


def integrand_Tm_solve(r, ratio,r0,h):
    const = 1
    T = const * integrate.quad(integrand_T_solve, 0, 10e-6, args=(r, ratio, r0, h))[0]
    return T * r * np.exp(-r**2/(r0**2))