# Author : Amalya Cox Johnson 
# email : amalyaj@stanford.edu


import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit as curve_fit
from lmfit.models import LinearModel, GaussianModel, VoigtModel, LorentzianModel, ExponentialModel
import lmfit as lm 
from OTRaPy.utils import * 
import h5py as h5

# Functions for fitting data and stuff

def wv_range(energies, wv1, wv2):
    """
    In self.energies, find index closest to specific start/stop points

    Args:
        wv1 (float or int) : first energy in range
        wv2 (float or int) : last energy in range

    Returns:
        start, stop (int): indices in self.energies to get nearest to wv1, wv2
    """
    start = np.argmin(np.abs(energies - wv1))
    stop = np.argmin(np.abs(energies - wv2))
    return start, stop

def custom_function(x:np.ndarray, y:np.ndarray, npeaks:int, peakfunction:lm.models, backgroundfunction:lm.models, centers:list | np.ndarray, peaktol:float | int=100, diffpeaks:bool=False):
        """"
        Function to build custom lmfit model for an arbitrary spectra given spectra data, number of peaks to fit,
        function to fit background, function to fit peaks (currently all peaks fit with same function)

        Args:
            x: xdata / energies
            y: ydata / spectra to fit
            npeaks (int): Number of peaks to fit
            peakfunction (lmfit.models): Model to fit all peaks, typically LorenzianModel, GaussianModel, VoigtModel
            backgroundfuction (lmfit.models): Model to fit background, typically ExponentialModel, LinearModel
            centers (list): Initial guess for peak centers
            peaktol (int or float): Min/max range for peak center in fitting
            diffpeaks (bool): If you want to fit each peak to specific lorentzian/gaussian/voigt model

        Returns:
            out (lmfit.model.ModelResult): Model result
        """

        bg_pre_dir = {ExponentialModel:'bgexp_', LinearModel:'bglin_'}
        model = backgroundfunction(prefix=bg_pre_dir[backgroundfunction])
        pars = model.guess(y, x=x)

        if diffpeaks == False:
            pre_dir = {ExponentialModel:'exp', GaussianModel:'g', LorentzianModel:'l', VoigtModel:'v'}
            pre = pre_dir[peakfunction]

            for n in np.arange(npeaks):
                mod = peakfunction(prefix=f'{pre}{n}_')
                init_center = centers[n]

                pars += mod.guess(y, x=x, center=init_center)
                pars[f'{pre}{n}_amplitude'].min = 0
                pars[f'{pre}{n}_center'].min = init_center - peaktol
                pars[f'{pre}{n}_center'].max = init_center + peaktol
                # other constraints
                model += mod

        out = model.fit(y, pars, x=x)
        return out


def fit_E_A(wl, spec, wv1=350, wv2=450,peakfunction=GaussianModel, backgroundfunction=ExponentialModel, return_height=False):
      """
      Function to fit E' and A' modes in raman spectra of MoS2
      Fit on a custom range
      The out result from custom func actually contains all the fitting parameters, here just looking at the peak locations, but may be worthwile to store
      all the pars if needed later, since they are available?

      Args:
          wv1: starting cm-1 of fit
          wv2: ending cm-1 of fit
          peakfunction (lmfit.models): Model to fit all peaks, typically LorenzianModel, GaussianModel, VoigtModel
          backgroundfuction (lmfit.models): Model to fit background, typically ExponentialModel, LinearModel
      Returns:
          peak1: E' fitted peak location
          peak1_err: Error on E'
          peak2: A' fitted peak location
          peak2_err: Error on A'
          x, y : selected raw data x and y to fit
          out.best_fit: best fit

      """
      start, stop = wv_range(wl, wv1,wv2)
      x = wl[start:stop]
      y = spec[start:stop]

      out = custom_function(x, y, 2, peakfunction, backgroundfunction, centers=[390,410], peaktol=5)
      
      # just selecting the parameters of interest
      pre_dir = {ExponentialModel:'exp', GaussianModel:'g', LorentzianModel:'l', VoigtModel:'v'}
      pre = pre_dir[peakfunction]
      peak1 = out.params[f'{pre}0_center'].value
      peak2 = out.params[f'{pre}1_center'].value
      peak1_err = out.params[f'{pre}0_center'].stderr
      peak2_err = out.params[f'{pre}1_center'].stderr
      if peak1_err == None:
          peak1_err = 0

      if peak2_err == None:
          peak2_err = 0

      if return_height:
        peak1height = out.params[f'{pre}0_height'].value
        peak2height = out.params[f'{pre}1_height'].value
        peak1height_err = out.params[f'{pre}0_height'].stderr
        peak2height_err = out.params[f'{pre}1_height'].stderr
        return peak1, peak1_err, peak1height, peak1height_err, peak2, peak2_err, peak2height, peak2height_err, x, y, out.best_fit
      return peak1, peak1_err, peak2, peak2_err, x, y, out.best_fit

def fit_multiple(files, measurement='piezo_hyperspec',plot=False, xaxis=None, custom_kwargs={}):
    """
    Fit multiple h5 maps. *not sweepmaps* 
    """
    Aarr = np.zeros(len(files))
    Aerrs = np.zeros(len(files))

    Earr = np.zeros(len(files))
    Eerrs = np.zeros(len(files)) 

    if plot: 
       fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))

    for i, f in enumerate(files): 
        file = h5.File(f)
        name = dict(file['/app/settings'].attrs.items())['sample']
        if measurement == 'hyperspec_picam_mcl': 
           wl_name = 'raman_shifts'
        elif measurement == 'piezo_hyperspec':
           wl_name = 'wls'
        wls = np.array(file[f'/measurement/{measurement}/{wl_name}'])
        spec_map = np.array(file[f'/measurement/{measurement}/spec_map'])
        if measurement == 'hyperspec_picam_mcl':
          pz, px, py , pwv = spec_map.shape
        elif measurement == 'piezo_hyperspec':
          pz, px, py ,_,  pwv = spec_map.shape
        spec_map = spec_map.reshape(px*py, pwv)
        av = spec_map.mean(axis=0)
        E, Eerr, A, Aerr, x, y, out = fit_E_A(wls, av, **custom_kwargs)
        Aarr[i] = A 
        Aerrs[i] = Aerr
        Earr[i] = E
        Eerrs[i] = Eerr

        if plot: 
           ax1.plot(x,y)
           ax1.plot(x,out)
    if plot: 
      if xaxis is None: 
        xaxis = np.arange(len(Aarr))

      ax2.errorbar(xaxis, Aarr, Aerrs, marker='o', linestyle='')
      fig.suptitle(name)
    return Aarr, Aerrs, Earr, Eerrs 


def fit_multiple_sweepmap(fn_list, dp, xaxis, remove_spikes=False, size=10, order=2, delta=2, plot=False):

  if plot: 
    f, (ax1, ax2) = plt.subplots(1,2)
  A_arr = np.zeros(dp)
  Aerr_arr = np.zeros(dp)

  for i in np.arange(0,dp):
    all_specs = {}
    for j, fn in enumerate(fn_list):
      file = h5.File(fn)
      spec_map = np.array(file['/measurement/hyperspec_picam_mcl_sweep/spec_map'])
      wls = np.array(file['/measurement/hyperspec_picam_mcl_sweep/raman_shifts'])
      scan_index_array = np.array(file['/measurement/hyperspec_picam_mcl_sweep/scan_index_array'])
      name = dict(file['/app/settings'].attrs.items())['sample']
      dz, dx, dy, dp, dwv = spec_map.shape
      temp = spec_map[0][:,:,i].reshape(dy*dx,dwv)
      all_specs[j] = temp
    this_pwr = np.concatenate([all_specs[0], all_specs[1]])#, all_specs[2]])
    av = this_pwr.mean(axis=0)
    if remove_spikes: 
      av = despike(av, size, order)
    _, _, A, Aerr, x, y, out = fit_E_A(wls, av, peakfunction=GaussianModel)
    A_arr[i] = A
    Aerr_arr[i] = Aerr
    if plot: 
      ax1.plot(wls, av + i*10)
      ax1.plot(x,out + i*10)
      ax2.errorbar(xaxis[i], A, Aerr, marker='o', color='blue')
  if plot: 
    ax1.set_xlim(350,450)
    ax1.set_ylim(1300,1800)
    mask = A_arr > 0
    popt, pcov = curve_fit(linear, xaxis[mask], A_arr[mask])
    ax2.plot(xaxis[mask], linear(xaxis[mask], *popt), label=round(popt[0],4))
    ax2.legend()
  return A_arr, Aerr_arr

def get_spot_size(image, name,initial_guess=[100,100,100,10,10,1,0.0], scale=px_to_um_100):
    f, (ax1, ax2) = plt.subplots(1,2)
    cropped_img = crop_spot_pic(rgb2gray(image), padding=100)
    xx, yy = np.indices(cropped_img.shape)
    ax1.pcolormesh(xx,yy,cropped_img)
    zz = cropped_img.ravel()
    popt, pcov = curve_fit(Gaussian_2D_curvefit, (xx,yy),zz, p0=initial_guess)
    fit = Gaussian_2D_curvefit((xx,yy), *popt).reshape(xx.shape)
    sigx = popt[3]
    sigy = popt[4]
    sigx /= scale
    sigy /= scale 
    perr = np.sqrt(np.diag(pcov))
    sigx_err = perr[3]
    sigy_err = perr[4]
    sigx_err /= scale
    sigy_err /= scale
    ax2.pcolormesh(xx,yy,fit)
    ax2.set_title(rf'$\sigma_x=${round(sigx,2)} $\mu$m, $\sigma_y=${round(sigy,2)} $\mu$m')
    f.suptitle(name)
    return sigx, sigy, sigx_err, sigy_err


def fit_dwdp(pwr, peak, peak_err=None, fit_err=False):
    if fit_err: 
        popt, pcov = curve_fit(linear, pwr, peak, sigma=peak_err, absolute_sigma=True)
    else: 
        popt, pcov = curve_fit(linear, pwr, peak)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def fit_sweepmap(fn, xaxis, do_dwdp=True,remove_spikes=False, wls=None,plot=True):
  file = h5.File(fn)
  name = dict(file['/app/settings'].attrs.items())['sample']
  named_position = dict(file['/hardware/nd_wheel/settings'].attrs.items())['named_position']
  angle = dict(file['/hardware/power_wheel/settings'].attrs.items())['position']
  if wls is None: 
     wls = np.array(file['/measurement/hyperspec_picam_mcl_sweep/raman_shifts'])

  spec_map = np.array(file['/measurement/hyperspec_picam_mcl_sweep/spec_map'])
  scan_index_array = np.array(file['/measurement/hyperspec_picam_mcl_sweep/scan_index_array'])
  exposure = dict(file['/hardware/picam/settings'].attrs.items())['ExposureTime']
  dz, dx, dy, dp, dwv = spec_map.shape
  if plot: 
    f, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
  Aarr = []
  Aarr_err = []
  skip_index = []
  for i in range(dp): 
      temp = spec_map[0][:,:,i].reshape(dy*dx,dwv)
      av = temp.mean(axis=0)
      _, _, A, Aerr, x, y, out = fit_E_A(wls, av, peakfunction=GaussianModel)
      if plot: 
        ax1.plot(x, y + i*5)
        ax1.plot(x,out + i*5)
        ax1.set_xlim(350,450)
        # ax1.set_ylim(1300,np.max(out)+100)
        ax2.errorbar(xaxis[i], A, Aerr, marker='o', color='blue')
      if Aerr == 0 or Aerr == np.NaN:
        if do_dwdp:
          skip_index.append(i)
        else:
          Aarr.append(A)      
          Aarr_err.append(Aerr)

      else:
        Aarr.append(A)      
        Aarr_err.append(Aerr)

  Aarr = np.array(Aarr)
  Aarr_err = np.array(Aarr_err)
  xindex = np.arange(dp)
  xindex = list(set(xindex).difference(set(skip_index)))
  
  xaxis = [xaxis[i] for i in xindex]
  xaxis = np.array(xaxis)
  if do_dwdp: 
    popt,perr = fit_dwdp(xaxis, Aarr, Aarr_err)
    if plot: 
      ax2.plot(xaxis, linear(xaxis, *popt), label=str(round(popt[0], 4)) + '+/-' + str(round(perr[0],4)))
      ax2.legend()
      f.suptitle(name + '|' + str(exposure/1e3) + 's')
    return Aarr, Aarr_err, popt, perr, xaxis
  else:
    return Aarr, Aarr_err, xaxis