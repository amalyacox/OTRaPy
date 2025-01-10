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
    return x * m + b


def angle_to_OD(angle, max_angle=270):
    b = -2
    m = (2 - 0.04) / max_angle
    return m * angle + b


def angle_to_power_fit(angle, Tmax, max_angle=270):
    # From previous isotropic raman measurements
    # Tmax_dict = {'20x': 4.7e-3 / (10**-0.04), '10x':4.7e-3 / (10**-0.04)}
    # Tmax = Tmax_dict[Tmax]
    OD = angle_to_OD(angle, max_angle)
    attenuation = 10 ** (OD)
    power = Tmax * attenuation
    return power


def domega_to_dT(warr, dwdT):
    w0 = warr[0]
    dw = warr - w0
    dT = dw / dwdT
    return dT


def crop_spot_pic(pic_array, padding=None):
    """Cuts out and returns a 2*padding X 2*padding sized window from `pic_array`, centered on the
    brightest pixel in `pic_array` (assumed to be located near the center of the laser spot).
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
    """Function for arbitrarily oriented 2D Gaussian function.  Used to fit laser spot pictures."""
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
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


px_to_um_10 = 1024 / 650
px_to_um_20 = 1024 / 320
px_to_um_100 = 1024 / 65


def create_sample_dict(directory, settings_i_want=[], settings_loc=[]):
    sorted_files = os.listdir(directory)
    sorted_files = [f for f in sorted_files if ".tif" in f or ".h5" in f]
    sorted_files.sort()
    sample_dict = {}

    for f in sorted_files:
        fn = f"{directory}/{f}"
        time = fn.split("/")[-1].split("_")[1]
        # date = fn.split('/')[-1].split('_')[0]
        measurement = "_".join(fn.replace(".h5", "").split("/")[-1].split("_")[2:])
        if "h5" in f:
            try:
                file = h5.File(fn)
                name = dict(file["app/settings"].attrs.items())["sample"]
                if name not in sample_dict.keys():
                    sample_dict[name] = {
                        fn: {
                            "time": time,
                            "measurement": measurement,
                            "settings": extract_h5_settings_HiP(
                                file, settings_i_want, settings_loc
                            ),
                        }
                    }
                else:
                    sample_dict[name][fn] = {
                        "time": time,
                        "measurement": measurement,
                        "settings": extract_h5_settings_HiP(
                            file, settings_i_want, settings_loc
                        ),
                    }
            except OSError:
                pass
    return sample_dict


def extract_h5_settings_HiP(file, settings_i_want, settings_loc):
    """
    Extract desired settings from an h5 file
    """
    settings = {}

    # all_settings = dict(file[f'{preamble}/settings'].attrs.items())

    for i_want, loc in zip(settings_i_want, settings_loc):
        settings[i_want] = dict(file[f"{loc}"].attrs.items())[i_want]
    return settings


def get_name_h5(fn):
    name = dict(h5.File(fn)["/app/settings"].attrs.items())["sample"]
    return name


def solve_T(solver, Q, kx, ky, g, Ta, threshold=1E-6): 
    Qmin, Qmax = Q.min(), Q.max()
    T1 = solver.Txy(kx, ky, g, Qmin, Ta=Ta, threshold=threshold)
    Tav1 = solver.weighted_average(T1)

    T2 = solver.Txy(kx, ky, g, Qmax, Ta=Ta, threshold=threshold)
    Tav2 = solver.weighted_average(T2)
    return np.linspace(Tav1, Tav2, len(Q)), T2

def plot_fits(solver, kx, ky, g, alpha, h, 
              dTdQ_x, Qx, w0_dTdQx, l0_dTdQx, 
              dTdQ_y, Qy, w0_dTdQy, l0_dTdQy,
              dTdQ_g, Qg, w0_dTdQg, l0_dTdQg):
    solver.alpha=alpha
    solver.h = h
    f, ((ax1,ax2,ax3), (ax4, ax5, ax6)) = plt.subplots(2,3,figsize=(15,10))
    dTdQ_g_arr = linear(Qg,dTdQ_g,300)
    ax1.scatter(Qg, dTdQ_g_arr)

    solver.update_w0(w0_dTdQg)
    solver.update_l0(l0_dTdQg)

    Tav, T = solve_T(solver,Qg, kx, ky, g, 300)
    ax1.plot(Qg, Tav, label='model')
    ax1.set_title('iso')
    ax1.legend()

    imiso = ax4.pcolormesh(solver.X[0], solver.Y[0],T[0])
    ax4.set_title('iso laser')
    plt.colorbar(imiso)

    dTdQ_x_arr = linear(Qx,dTdQ_x,300)
    ax2.scatter(Qx, dTdQ_x_arr)

    solver.update_w0(w0_dTdQx)
    solver.update_l0(l0_dTdQx)
    Tav, T = solve_T(solver,Qx, kx, ky, g, 300)
    ax2.plot(Qx, Tav, label='model')
    ax2.set_title('laser y')

    imly = ax5.pcolormesh(solver.X[0], solver.Y[0],T[0])
    ax5.set_title('laser y')
    plt.colorbar(imly)


    solver.update_w0(w0_dTdQy)
    solver.update_l0(l0_dTdQy)
    dTdQ_y_arr = linear(Qy,dTdQ_y,300)
    ax3.scatter(Qy, dTdQ_y_arr)
    Tav, T = solve_T(solver,Qy, kx, ky, g, 300)
    ax3.plot(Qy, Tav, label='model')
    ax3.set_title('laser x')

    imlx = ax6.pcolormesh(solver.X[0], solver.Y[0],T[0])
    ax6.set_title('laser x')
    plt.colorbar(imlx)

    f.suptitle(f'kx={round(kx,2)}W/mK, ky={round(ky,2)}W/mK, g={round(g/1e6, 2)}MW/m2K')
    return dTdQ_x_arr, dTdQ_y_arr, dTdQ_g_arr

def dict_to_plot(my_dict):
    pwr = my_dict['pwr']
    A = my_dict['A']
    Aerr = my_dict['Aerr']
    dwdt = my_dict['dwdT']
    dT = (np.array(A) - A[0]) * 1/dwdt
    dT_err = np.array(Aerr) * -1/dwdt
    return np.array(pwr), dT, dT_err, np.array(A), np.array(Aerr), dwdt