# Author : Amalya Cox Johnson
# email : amalyaj@stanford.edu

import numpy as np
from scipy.special import i0, k0
from scipy import integrate

# Helper functions for analytic solving of thermal transport equations


def integrand_T(y, r, k, g, r0, h):
    """
    Integrand for computing analytic solution to cylindrical heating.
    From: https://pubs.acs.org/doi/10.1021/nl9041966

    Parameters
    ----------
    y : float
        The variable of integration.
    r : float
        the spatial coordinate over which to integrate
    k : float
        thermal conductivity of the material (W/mK)
    g : float
        thermal conductance per unit volume (thermal boundary conductance) (W/m2K)
    r0 : float
        spot size of laser (m)
    h : float
        thickness of material (m)

    Returns
    -------
    F : float
        The value of the integrand at y.
    """
    x = np.sqrt(g / (k * h))
    I0_r = i0(x * r)
    I0_y = i0(x * y)
    K0_r = k0(x * r)
    K0_y = k0(x * y)
    if y <= r:
        F = K0_r * I0_y
    else:
        F = K0_y * I0_r

    return F * np.exp(-(y**2) / r0**2) * y


def integrand_T_solve(y, r, ratio, r0, h):
    """
    Integrand for solving the analytic solution to cylindrical heating, but as a function of k/g ratio for solving.
    From: https://pubs.acs.org/doi/10.1021/nl9041966

    Parameters
    ----------
    y : float
        The variable of integration.
    r : float
        the spatial coordinate over which to integrate
    ratio : float
        ratio of thermal conductivity to thermal boundary conductance (k/g)
    r0 : float
        spot size of laser (m)
    h : float
        thickness of material (m)

    Returns
    -------
    F : float
        The value of the integrand at y.
    """
    x = np.sqrt(ratio / (h))
    I0_r = i0(x * r)
    I0_y = i0(x * y)
    K0_r = k0(x * r)
    K0_y = k0(x * y)
    if y <= r:
        F = K0_r * I0_y
    else:
        F = K0_y * I0_r

    return F * np.exp(-(y**2) / r0**2) * y


def integrand_Tm_solve(r, ratio, r0, h):
    """
    Weighted integral of integrand_T_solve for calculating T measured by raman laser
    From: https://pubs.acs.org/doi/10.1021/nl9041966

    Parameters
    ----------
    r : float
        The variable of integration.
    ratio : float
        ratio of thermal conductivity to thermal boundary conductance (k/g)
    r0 : float
        spot size of laser (m)
    h : float
        thickness of material (m)

    Returns
    -------
    F : float
        The value of the integrand at y.
    """
    const = 1
    T = const * integrate.quad(integrand_T_solve, 0, 10e-6, args=(r, ratio, r0, h))[0]
    return T * r * np.exp(-(r**2) / (r0**2))
