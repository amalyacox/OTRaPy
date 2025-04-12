# Author : Amalya Cox Johnson
# email : amalyaj@stanford.edu


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as curve_fit
import os
import h5py as h5


def linear(x, m, b):
    """
    Linear function for fitting

    Parameters
    ----------
    x : array like
        x axis
    m : float or int
        slope
    b : float or int
        y-intercept

    Returns
    -------
    y : array like
        y axis
    """
    return x * m + b


def angle_to_OD(angle, max_angle=270):
    """
    Converts angle to optical density (OD) using a linear fit, based on thorlabs round continuous variable ND filter.
    https://www.thorlabs.com/thorproduct.cfm?partnumber=NDC-25C-2

    Parameters
    ----------
    angle : float or int
        angle in degrees   
    max_angle : float or int, optional
        angle that is the max power of the filter, default 270
    
    Returns
    -------
    OD : float or int
        optical density (OD) of the filter at the given angle
     
    """
    b = -2
    m = (2 - 0.04) / max_angle
    return m * angle + b


def angle_to_power_fit(angle, Tmax, max_angle=270):
    """
    Fitting function for power vs angle. 

    Parameters
    ----------
    angle : float or int
        angle in degrees
    Tmax : float or int
        maximum power of the laser (W)
    max_angle : float or int, optional
        angle that is the max power of the filter, default 270
    
    Returns
    -------
    power : float or int
        power at the given angle (W)
    """
    # From previous isotropic raman measurements
    # Tmax_dict = {'20x': 4.7e-3 / (10**-0.04), '10x':4.7e-3 / (10**-0.04)}
    # Tmax = Tmax_dict[Tmax]
    OD = angle_to_OD(angle, max_angle)
    attenuation = 10 ** (OD)
    power = Tmax * attenuation
    return power

def curve_fit_power(angle, m, b, Tmax):
    """
    Fitting function for scipy.optimize.curve_fit for fitting power vs angle.

    Parameters
    ----------
    angle : float or int
        angle in degrees
    m : float or int
        slope of the linear fit
    b : float or int
        y-intercept of the linear fit
    Tmax : float or int
        maximum power of the laser (W)

    Returns
    -------
    power : float or int
        power at the given angle (W)
    """
    OD = (m / 270) * angle + b
    attenuation = 10 ** (OD)
    power = Tmax * attenuation
    return power

def domega_to_dT(warr, dwdT):
    """
    Given an array of raman peak positions, converts to temperature using the slope of the peak position vs temperature curve (dwdT).

    Parameters
    ----------
    warr : array like
        array of raman peak positions (cm^-1)
    dwdT : float or int
        slope of the peak position vs temperature curve (cm^-1/K)
    
    Returns
    -------
    dT : array like
        array of temperature changes (K)
    """
    w0 = warr[0]
    dw = warr - w0
    dT = dw / dwdT
    return dT


def crop_spot_pic(pic_array, padding=None):
    """
    Cuts out and returns a 2*padding X 2*padding sized window from `pic_array`, centered on the
    brightest pixel in `pic_array` (assumed to be located near the center of the laser spot).

    Parameters
    ----------
    pic_array : array like
        array of the laser spot picture (2D array)
    padding : int, optional
        padding for the size of the window to be cut out, default is 100 pixels

    Returns
    -------
    pic_array : array like
        array of the cropped laser spot picture (2D array)
    """
    if padding == None:
        padding = 100  # use default value if nothing specific provided
    row, col = np.indices(
        pic_array.shape
    )  # Matrix of indices for pixels in `pic_array`
    brightest_px_mask = (
        pic_array == pic_array.max()
    )  # create mask to locate brightest pixel (near spot center)
    n_px, m_px = (
        row[brightest_px_mask][0],
        col[brightest_px_mask][0],
    )  # x, y coordinates for brightest pixel
    return pic_array[
        n_px - padding : n_px + padding, m_px - padding : m_px + padding
    ]  # , n_px, m_px


def Gaussian_2D_curvefit(xandy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    Function for arbitrarily oriented 2D Gaussian function.  Used to fit laser spot pictures.
    
    Parameters
    ----------
    xandy : tuple
        x and y coordinates of the pixels in the laser spot picture (tuple of arrays)
    amplitude : float or int
        amplitude of the Gaussian function (peak intensity)
    xo : float or int
        x coordinate of the center of the Gaussian function (center of the laser spot)
    yo : float or int
        y coordinate of the center of the Gaussian function (center of the laser spot)
    sigma_x : float or int
        standard deviation of the Gaussian function in the x direction (width of the laser spot)
    sigma_y : float or int
        standard deviation of the Gaussian function in the y direction (width of the laser spot)
    theta : float or int
        angle of rotation of the Gaussian function (orientation of the laser spot)
    offset : float or int
        offset of the Gaussian function (background intensity)

    Returns
    -------
    g : array like
        array of the Gaussian function evaluated at the x and y coordinates (1D array, flattened)
    """
    (x, y) = xandy
    xo = float(xo)  # make sure not int type (will not divide properly)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    g = offset + amplitude * np.exp(
        -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
    )
    return g.ravel()


def rgb2gray(rgb):
    """Convert RGB image to grayscale using the luminosity method.
    
    Parameters
    ----------
    rgb : array like
        RGB image (3D array)
    
    Returns
    -------
    gray : array like
        Grayscale image (2D array)
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def r_from_zl(r0, zl, NA):
    """
    Calculate the radius of the laser spot at a given distance from the focus.

    Parameters
    ----------
    r0 : float
        radius of the laser spot at the focus
    zl : float
        distance from the focus
    NA : float
        numerical aperture of the lens
    
    Returns
    -------
    r : float
        radius of the laser spot at the given distance from the focus

    """
    return np.sqrt(r0**2 + (zl * NA) ** 2)

# Calibrations from HiP Microscope at Molecular Foundry, 2024 
px_to_um_10 = 1024 / 650
px_to_um_20 = 1024 / 320
px_to_um_100 = 1024 / 65


