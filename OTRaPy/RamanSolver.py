
# Author : Amalya Cox Johnson 
# email : amalyaj@stanford.edu

import numpy as np 
from scipy.interpolate import RectBivariateSpline 
from scipy.optimize import curve_fit
from scipy.optimize import minimize 

from OTRaPy.utils import * 

class RamanSolver: 
    """
    Solver for 2D Diffusion in Raman Thermometry Measurements
    Default materials parameters come from Taube, et. al, MoS2 on SiO2/Si 
    """
    def __init__(self, lx:float=1.5E-6, ly:float=1.5E-6, nx:int=50, ny:int=50, h:float=0.7e-9, alpha:float=0.07, w0:float=0.4e-6, l0:float=0.4e-6, iso_w0:float=0.4e-6, iso_l0:float=0.4e-6) -> None:
        # define the grid

        self.lx = lx 
        self.ly = ly 
        self.nx = nx
        self.ny = ny
        # things that won't be changing throughout fitting 
        # but will ultimately probably need error propagation? 
        
        self.h = h #thickness (m)
        self.w0 = w0 #gaussian beam width in x-direction (m)
        self.l0 = l0 #gaussian beam width in y-direction (m)
        self.iso_w0 = w0 #gaussian beam width for isotropic measurement
        self.iso_l0 = l0 #gaussian beam width for isotropic measurement
        self.alpha = alpha #material % absorption at laser wavelength (%)
    
        dx = 2*self.lx / (self.nx-1)
        dy = 2*self.ly / (self.nx-1)
        assert dx == dy, 'hm, dx and dy are different, please reconsider'

        self.delta = dx 
        self.x = np.linspace(-self.lx, self.lx, self.nx)
        self.y = np.linspace(-self.ly, self.ly, self.ny)

        self.X, self.Y = np.meshgrid(self.x, np.flipud(self.y))

    def generate_qdot(self, Q:float=1e-3):
        """ 
        Generate initial power flux from laser in W/m2. Default w0=l0 gives gaussian beam. 
        w0!=l0 gives elliptical beam. 
        
        Parameters 
        ----------
        Q : float 
            Input laser power, default 1e-3 (W)
        
        """
        coef = (Q * self.alpha) / (np.pi * self.w0 * self.l0) #W/m2
        qdot = coef * np.exp(-(self.X**2/self.w0**2 + self.Y**2/self.l0**2))
        self.qdot = qdot
    
    def Txy(self, kx:float=62.2, ky:float=62.2, g:float=1.94E6, Q:float=1e-3, threshold:float=1.E-5, Ta:float=300., verbose:bool=False):
        """
        Solve the steady-state 2D temperature distribution in a material with laser heating. 

        Parameters 
        ----------
        Q : float 
            Input laser power, default 1e-3(W)
        kx : float 
            Thermal conductivity in the x-direction (W/mK), default 62.2
        ky : float
            Thermal conductivity in the y-direction (W/mK), default 62.2
        g : float 
            Interfacial thermal conductance between material and substrate (for supported sample) (W/m2K), default 1.94E6
        threshold : float 
            Threshold for error in calculation, default 1E-5
        Ta : float 
            Temperature of substrate (K), default 300
        verbose : bool 
            Whether or not to print error, default False

        """
        error = 1 
        count = 1
    
        self.generate_qdot(Q)

        # giving it a better initial guess to start 
        # if not hasattr(self, 'Tg'): 
        T = np.ones((self.nx,self.ny))*Ta
        Tg = T.copy()
        # else: 
            # T = self.Tg.cop/y()
            # Tg = self.Tg
        d2h = self.delta**2/self.h

        while error > threshold: 
            for i in range(1, self.nx-1): 
                for j in range(1, self.ny-1): 
                    T[i,j]=(kx * (T[i,j-1] + T[i,j+1]) + ky * (T[i-1,j] + T[i+1,j]) + d2h * (self.qdot[i,j] + g * Ta)) / (d2h * g + 2*kx + 2*ky) 

            error = np.sqrt(np.sum(np.abs(T-Tg)**2))
            if verbose: 
                if count % 100 == 0 :
                    print(error)
            count +=1
            Tg = T.copy()

        # self.Tg = T
        return T
    
    def weighted_average(self, T):
        """
        Weighted average of temperature (or any) distribution over laser. 
        Integral evaluated using scipy.interpolate.RectBivariateSpline

        Parameters 
        ----------
        T : array like
            Array to perform average over 

        """
        exp_factor = np.exp(-(self.X**2/self.w0**2 + self.Y**2/self.l0**2))
        rbs = RectBivariateSpline(self.x, self.y, T * exp_factor)
        integrated = rbs.integral(-self.lx, self.lx, -self.ly, self.ly)
        Tav = integrated / (np.pi * self.w0 * self.l0)
        return Tav 
    
    def dTdQ(self, kx:float, ky:float, g:float, Qarr, quick:bool=True):
        """
        Calculate dTdQ for a given Q range and thermal properties (kx, ky, kg)

        Parameters 
        ----------
        kx : float 
            Thermal conductivity in x-direction 
        ky : float 
            Thermal conductivity in y-direction
        g : float 
            Interfacial thermal conductance 
        Qarr : array-like
            Range of input power values 
        quick : bool 
            Whether or not to calculate the slope as just y2-y1 / x2-x1 (first and last points) or fit a line to the data, default True 

        """
        if quick: 
            T1 =  self.Txy(kx=kx,ky=ky, g=g, Q=Qarr[0])
            Tav1 = self.weighted_average(T1)
            T2 = self.Txy(kx=kx,ky=ky, g=g, Q=Qarr[-1])
            Tav2 = self.weighted_average(T2)

            slope = (Tav2 - Tav1) / (Qarr[-1] - Qarr[0])
            return slope 
        else: 
            Tarr = []
            for Q in Qarr: 
                T = self.Txy(kx=kx,ky=ky, g=g, Q=Q)
                Tav = self.weighted_average(T)
                Tarr.append(Tav)
            Tarr = np.array(Tarr)

            popt, pcov = curve_fit(linear, Qarr, Tarr)
            slope = popt[0]
            return slope, Tarr 

    def solve_kx(self, kx, dTdQ_x:float, Qarr, ky:float, g:float, verbose:bool=False):
        """
        Helper function for scipy.optimize.minimize to solve for kx independently

        Parameters 
        ----------
        kx : float 
            Independent variable for scipy.optimize.minimize. What we are solving for 
        dTdQ_x : float 
            Experimental value to fit model to and solve for kx
        Qarr : array-like 
            Range of input power values 
        ky : float
            Value for ky at which we are solving for kx 
        g : float 
            Value for g at which we are solving for kx
        verbose : bool 
            print progress, default False 
        """
        scaled = dTdQ_x * 1

        guess = self.dTdQ(kx=kx, ky=ky, g=g, Qarr=Qarr)*1
        if verbose: 
            print('solving kx:',np.sqrt((guess - scaled)**2))
        return np.sqrt((guess - scaled)**2)

    def solve_ky(self, ky, dTdQ_y:float, Qarr, kx:float, g:float, verbose:bool=False):
        """
        Helper function for scipy.optimize.minimize to solve for ky independently

        Parameters 
        ----------
        ky : float 
            Independent variable for scipy.optimize.minimize. What we are solving for 
        dTdQ_y : float 
            Experimental value to fit model to and solve for ky
        Qarr : array-like 
            Range of input power values 
        kx : float
            Value for ky at which we are solving for ky
        g : float 
            Value for g at which we are solving for ky
        verbose : bool 
            print progress, default False 
        """
        scaled = dTdQ_y * 1
        guess = self.dTdQ(kx=kx, ky=ky, g=g, Qarr=Qarr)*1
        if verbose: 
            print('solving ky:', np.sqrt((guess - scaled)**2))
        return np.sqrt((guess - scaled)**2)

    def solve_g(self, g, dTdQ_g:float, Qarr, kx:float, ky:float, verbose:bool=False):
        """
        Helper function for scipy.optimize.minimize to solve for g independently
        TODO: This will probably actually be minimizing to TDTR data 

        Parameters 
        ----------
        g : float 
            Independent variable for scipy.optimize.minimize. What we are solving for 
        dTdQ_g : float 
            Experimental value to fit model to and solve for g
        Qarr : array-like 
            Range of input power values 
        kx : float
            Value for ky at which we are solving for g 
        ky : float 
            Value for g at which we are solving for g
        verbose : bool 
            print progress, default False 
        """
        scaled = dTdQ_g * 1   
        guess = self.dTdQ(kx=kx, ky=ky, g=g, Qarr=Qarr)*1

        if verbose: 
            print('solving g:', np.sqrt((guess - scaled)**2))
        return np.sqrt((guess - scaled)**2)

    def iter_solve(self, kxg:float, kyg:float, gg:float, dTdQ_x:float, dTdQ_y:float, dTdQ_g:float, Qarr, threshold=1, verbose=False, 
                   tolerance=1E-6, method='Nelder-Mead', 
                   kx_bounds = ((0.,500),), ky_bounds=((0.,500.),), g_bounds=((0.5E6,10E6),), 
                   Qy=None, Qiso=None, wy=None, ly=None, wx = None, lx=None): 
        """
        Solve for kx, ky, g iteratively 
        
        Parameters 
        ----------
        kxg : float 
            Initial guess for thermal conductivity in x-direction 
        kyg : float 
            Initial guess for thermal conductivity in y-direction 
        gg : float 
            Initial guess for interfacial thermal conductance             
        dTdQ_x : float 
            dTdQ from anisotropic experiment for determining kx 
        dTdQ_y : float 
            dTdQ from anisotropic experiment experiment for determining ky 
        dTdQ_g : float 
            dTdQ from isotropic experiment for determining g 
        Qarr : array-like
            Range of input power values 
        threshold : float or int 
            Threshold for convergence of kx, ky, g 
        verbose : bool 
            print progress, default False 
        method : string 
            method for scipy.optimize.minimize, default = 'Nelder-Mead' 
        kx_bounds : tuple
            bounds for kx, default ((0., 500.),)
        kx_bounds : tuple
            bounds for ky, default ((0., 500.),)
        g_bounds : tuple 
            bounds for g, default ((0.5E6, 10E6),)
        Qy : array-like
            Optional if need to update Q when solving for ky, default None
        Qiso : array-like
            Optional if need to update Q when solving for g, default None
        """
        error_kx = error_ky = error_g = 10
        count = 0
        # We need to remember to flip the orientation of the laser depending on what we are solving for 
        # we may also need to update the Qarr if slightly different Q for different measurements
        
        if Qy is None: 
            Qy = Qarr
        if Qiso is None: 
            Qiso = Qarr 

        if wy is None and wx is None: 
            widths = [self.w0, self.l0]
            widths.sort()
            width_short, width_long = widths 
            wx = width_short # for solving kx, "lasery"
            lx = width_long # for solving kx, "lasery"

            wy = width_long # for solving ky, "laserx"
            ly = width_short # for solving ky, "laserx"

        # I don't think I want the kx and ky that are conver
        while error_kx > threshold or error_ky > threshold: # or error_g > threshold: 
            # first, assume kyg (and gg always for now) are correct, solve for kx
            print('----------- assuming ky, solving kx ---------------')
            
            # when solving for kx, we are using dTdQ from when the laser is aligned with the y axis 
            self.w0 = wx # width_short
            self.l0 = lx # width_long 
        
            sol = minimize(self.solve_kx, kxg, args=(dTdQ_x, Qarr, kyg, gg, verbose), tol=tolerance, method=method, bounds=kx_bounds)
            kx_solved = sol.x[0] 
            dTdQ_x_solved = self.dTdQ(kx_solved, kyg, gg, Qarr)
            dTdQ_x_err = np.sqrt((dTdQ_x - dTdQ_x_solved)**2)

            error_kx = np.sqrt((kxg - kx_solved)**2)
            kxg = kx_solved

            # assume kxg solution is correct, and solve for ky
            print(f'----------- kx updated, kx = {kxg}, dTdQ_x_err = {dTdQ_x_err}, solving ky ---------------')

            # when solving for ky, we are using dTdQ from when the laser is aligned with the x axis 
            self.w0 = wy #width_long
            self.l0 = ly #width_short

            sol = minimize(self.solve_ky, kyg, args=(dTdQ_y, Qy, kxg, gg, verbose), tol=tolerance, method=method, bounds=ky_bounds)
            ky_solved = sol.x[0] 
            dTdQ_y_solved = self.dTdQ(kxg, ky_solved, gg, Qy)
            dTdQ_y_err = np.sqrt((dTdQ_y - dTdQ_y_solved)**2)


            error_ky = np.sqrt((kyg - ky_solved)**2)
            
            kyg = ky_solved

            print(f'----------- ky updated, ky = {kyg}, dTdQ_y_err = {dTdQ_y_err}, solving g ---------------')

            # to solve for g, we use dTdQ from the isotropic raman measurements

            # self.w0 = self.iso_w0
            # self.l0 = self.iso_l0

            # sol = minimize(self.solve_g, gg, args=(dTdQ_g, Qiso, kxg, kyg, verbose), tol=tolerance, method=method, bounds=g_bounds)
            # g_solved = sol.x[0]
            # error_g = np.sqrt((gg - g_solved)**2)
            # gg = g_solved
            # print(f'----------- g updated, g = {gg} ---------------')

            count +=1 
            print(f'----------- iter {count}, kx = {kxg}, ky = {kyg}, g = {gg}, error_kx = {error_kx}, error_ky = {error_ky}, dTdQ_x_err = {dTdQ_x_err}, dTdQ_y_err = {dTdQ_y_err}, error_g = {error_g} -----------')

        print(f'converged at iter {count}, kx = {kxg}, ky = {kyg}, g = {gg}')

        self.kx_solved = kxg 
        self.ky_solved = kyg 
        self.g_solved = gg







    
    
    

    

        







        
    


    