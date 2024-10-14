

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
    """
    return x*m + b 

def angle_to_OD(angle, max_angle=270):
    b = -2
    m = (2-0.04)/max_angle
    return m*angle + b

def angle_to_power_fit(angle, Tmax, max_angle=270):
    # From previous isotropic raman measurements 
    # Tmax_dict = {'20x': 4.7e-3 / (10**-0.04), '10x':4.7e-3 / (10**-0.04)}
    # Tmax = Tmax_dict[Tmax]
    OD = angle_to_OD(angle, max_angle)
    attenuation = 10**(OD)
    power = Tmax * attenuation
    return power

def domega_to_dT(warr, dwdT): 
    w0 = warr[0]
    dw = warr - w0
    dT = dw / dwdT 
    return dT 
    

def crop_spot_pic(pic_array, padding=None):
    """Cuts out and returns a 2*padding X 2*padding sized window from `pic_array`, centered on the
    brightest pixel in `pic_array` (assumed to be located near the center of the laser spot)."""
    if padding==None:
        padding = 100 # use default value if nothing specific provided
    row, col = np.indices(pic_array.shape) # Matrix of indices for pixels in `pic_array`
    brightest_px_mask = pic_array==pic_array.max() # create mask to locate brightest pixel (near spot center)
    n_px, m_px = row[brightest_px_mask][0], col[brightest_px_mask][0] # x, y coordinates for brightest pixel
    return pic_array[n_px-padding:n_px+padding, m_px-padding:m_px+padding]#, n_px, m_px

def Gaussian_2D_curvefit(xandy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """Function for arbitrarily oriented 2D Gaussian function.  Used to fit laser spot pictures."""
    (x,y) = xandy
    xo = float(xo) # make sure not int type (will not divide properly)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

px_to_um_10 = 1024 / 650
px_to_um_20 = 1024 / 320
px_to_um_100 = 1024 / 65


def create_sample_dict(directory, settings_i_want=[], settings_loc=[]): 
    sorted_files = os.listdir(directory)
    sorted_files = [f for f in sorted_files if '.tif' in f or '.h5' in f]
    sorted_files.sort()
    sample_dict = {}

    for f in sorted_files: 
        fn = f'{directory}/{f}' 
        time = fn.split('/')[-1].split('_')[1]
        # date = fn.split('/')[-1].split('_')[0]
        measurement = '_'.join(fn.replace('.h5', '').split('/')[-1].split('_')[2:])
        if 'h5' in f: 
            try: 
                file = h5.File(fn)
                name = dict(file['app/settings'].attrs.items())['sample']
                if name not in sample_dict.keys(): 
                    sample_dict[name] = {fn:{'time':time, 'measurement':measurement, 'settings' : extract_h5_settings_HiP(file, settings_i_want, settings_loc)}}
                else: 
                    sample_dict[name][fn] = {'time':time, 'measurement':measurement, 'settings' : extract_h5_settings_HiP(file, settings_i_want, settings_loc)}
            except OSError: 
                pass 
    return sample_dict

def extract_h5_settings_HiP(file, settings_i_want, settings_loc): 
    """
    Extract desired settings from an h5 file 
    """
    settings = {}

    # all_settings = dict(file[f'{preamble}/settings'].attrs.items())

    for i_want,loc in zip(settings_i_want, settings_loc): 
        settings[i_want] = dict(file[f'{loc}'].attrs.items())[i_want]
    return settings



    

