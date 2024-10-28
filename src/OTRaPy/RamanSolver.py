# Author : Amalya Cox Johnson
# email : amalyaj@stanford.edu

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import curve_fit, minimize, least_squares
import warnings
from OTRaPy.utils import *
from scipy.interpolate import interp1d


class RamanSolver:
    """
    Solver for 2D Diffusion in Raman Thermometry Measurements
    Default materials parameters come from Taube, et. al, MoS2 on SiO2/Si
    """

    def __init__(
        self,
        lx: float = 1.5e-6,
        ly: float = 1.5e-6,
        nx: int = 50,
        ny: int = 50,
        h: float = 0.7e-9,
        alpha: float = 0.07,
        w0: float = 0.4e-6,
        l0: float = 0.4e-6,
        w0_dTdQg = None,
        l0_dTdQg = None,
        w0_dTdQx = None, 
        l0_dTdQx = None, 
        w0_dTdQy = None, 
        l0_dTdQy = None
    ) -> None:
        # define the grid

        self.lx = lx
        self.ly = ly
        self.nx = nx
        self.ny = ny
        # things that won't be changing throughout fitting
        # but will ultimately probably need error propagation?

        self.h = h  # thickness (m)

        self.update_l0(l0) # gaussian beam width in y-direction (m)
        self.update_w0(w0) # gaussian beam width in x-direction (m)

        self.alpha = alpha  # material % absorption at laser wavelength (%)
        
        self.w0_dTdQg, self.l0_dTdQg = w0_dTdQg, l0_dTdQg  # gaussian beam width for isotropic measurement
        self.w0_dTdQx, self.l0_dTdQx, self.w0_dTdQy, self.l0_dTdQy = w0_dTdQx, l0_dTdQx, w0_dTdQy, l0_dTdQy # gaussian beam width for anisotropic measurements

        dx = 2 * self.lx / (self.nx - 1)
        dy = 2 * self.ly / (self.nx - 1)
        assert dx == dy, "hm, dx and dy are different, please reconsider"

        self.delta = dx
        self.x = np.linspace(-self.lx, self.lx, self.nx)
        self.y = np.linspace(-self.ly, self.ly, self.ny)

        self.X, self.Y = np.meshgrid(self.x, np.flipud(self.y))
        self.X = np.broadcast_to(self.X, (len(self.w0), self.nx, self.nx))
        self.Y = np.broadcast_to(self.Y, (len(self.l0), self.ny, self.ny))
    
    def update_w0(self, new_w0): 
        if type(new_w0) == float or type(new_w0) == np.float64: 
            self.w0 = np.array([new_w0])
        else: 
            self.w0 = new_w0
        self.w0 = np.broadcast_to(self.w0, (self.nx, self.ny, len(self.w0))).T

    def update_l0(self, new_l0):
        if type(new_l0) == float or type(new_l0) == np.float64: 
            self.l0 = np.array([new_l0])
        else: 
            self.l0 = new_l0
        self.l0 = np.broadcast_to(self.l0, (self.nx, self.ny, len(self.l0))).T


    def generate_qdot(self, Q: float = 1e-3):
        """
        Generate initial power flux from laser in W/m2. Default w0=l0 gives gaussian beam.
        w0!=l0 gives elliptical beam.

        Parameters
        ----------
        Q : float
            Input laser power, default 1e-3 (W)

        """
        coef = (Q * self.alpha) / (np.pi * self.w0 * self.l0)  # W/m2
        qdot = coef * np.exp(-(self.X**2 / self.w0**2 + self.Y**2 / self.l0**2))
        self.qdot = qdot

    
    def Txy(
        self,
        kx: float = 62.2,
        ky: float = 62.2,
        g: float = 1.94e6,
        Q: float = 1e-3,
        threshold: float = 1.0e-5,
        Ta: float = 300.0,
        d2h=None,
        verbose: bool = False,
    ):
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

        self.generate_qdot(Q)

        T = np.ones(self.X.shape) * Ta
        Tg = T.copy()
        

        if d2h is None:
            d2h = self.delta**2 / self.h

        while error > threshold:
            ### Might be producing some error with edges?
            # padded_T = np.pad(T, pad_width=1, mode='constant', constant_values=Ta)

            Tx = np.roll(T, -1, axis=2) + np.roll(T, 1, axis=2)

            Ty = np.roll(T, -1, axis=1) + np.roll(T, 1, axis=1)

            T = (kx * (Tx) + ky * (Ty) + d2h * (self.qdot + g * Ta)) / (
                d2h * g + 2 * kx + 2 * ky
            )
            T[:,:, 0] = Ta
            T[:, -1] = Ta
            T[:,-1, :] = Ta
            T[:,0, :] = Ta
            error = np.sqrt(np.sum(np.abs(T - Tg) ** 2))

            Tg = T.copy()
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
        Tav = []
        exp_factor = np.exp(-(self.X**2 / self.w0**2 + self.Y**2 / self.l0**2))
        for i,Ti in enumerate(T*exp_factor): 
            rbs = RectBivariateSpline(self.x, self.y, Ti)
            integrated = rbs.integral(-self.lx, self.lx, -self.ly, self.ly)
            Tav.append((integrated / (np.pi * self.w0[i]* self.l0[i]))[0,0])
        Tav = np.array(Tav)
        return Tav

    def dTdQ(self, kx: float, ky: float, g: float, Q, quick: bool = True, **kwargs):
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
        Q : array-like
            Range of input power values
        quick : bool
            Whether or not to calculate the slope as just y2-y1 / x2-x1 (first and last points) or fit a line to the data, default True

        """
        T1 = self.Txy(kx=kx, ky=ky, g=g, Q=Q[0], **kwargs)
        Tav1 = self.weighted_average(T1)
        T2 = self.Txy(kx=kx, ky=ky, g=g, Q=Q[-1], **kwargs)
        Tav2 = self.weighted_average(T2)

        slope = (Tav2 - Tav1) / (Q[-1] - Q[0])
        
        return slope
        
    def dTdQ_sim(self, kx, ky, g, Q,w0,l0,Ta):
        self.update_l0(l0)
        self.update_w0(w0)
        dTdQ = self.dTdQ(kx,ky,g,Q)
        sim = linear(Q, dTdQ, Ta)
        return sim
    
    def resid_dTdQ(self, kx, ky, g, Q, dTdQ_arr, w0,l0,Ta): 
        sim = self.dTdQ_sim(kx, ky, g, Q, w0,l0,Ta)
        # plt.plot(Q, dTdQ_arr)
        # plt.plot(Q, sim)
        return (dTdQ_arr - sim)**2

    def resid_full(self, p, dTdQ_x_arr, Qx, w0_dTdQx, l0_dTdQx, 
                                dTdQ_y_arr, Qy, w0_dTdQy, l0_dTdQy,
                                dTdQ_g_arr, Qg, w0_dTdQg, l0_dTdQg,Ta): 
        kx, ky, g = p
        eq1 = self.resid_dTdQ(kx, ky, g, Qx, dTdQ_x_arr, w0_dTdQx, l0_dTdQx, Ta)
        eq2 = self.resid_dTdQ(kx, ky, g, Qy, dTdQ_y_arr, w0_dTdQy, l0_dTdQy, Ta)
        eq3 = self.resid_dTdQ(kx, ky, g, Qg, dTdQ_g_arr, w0_dTdQg, l0_dTdQg, Ta)
        eqtns = np.concatenate([eq1, eq2, eq3])
        print(np.sum(eqtns), kx, ky, g)
        return eqtns

    def minimize_resid_full(self, x0, dTdQ_x, Qx, w0_dTdQx, l0_dTdQx, 
                                dTdQ_y, Qy, w0_dTdQy, l0_dTdQy,
                                dTdQ_g, Qg, w0_dTdQg, l0_dTdQg,
                                Ta, **ls_kwargs):
        
        root = least_squares(self.resid_full, x0=x0, args=(dTdQ_x, Qx, w0_dTdQx, l0_dTdQx, 
                                dTdQ_y, Qy, w0_dTdQy, l0_dTdQy,
                                dTdQ_g, Qg, w0_dTdQg, l0_dTdQg,Ta),**ls_kwargs)
        return root
    
    def resid_full_varyall(self, p, dTdQ_x_arr, Qx,dTdQ_y_arr, Qy, dTdQ_g_arr, Qg, Ta): 
        kx, ky, g, h, alpha, w0_dTdQx, l0_dTdQx, w0_dTdQg = p #w0_dTdQy, l0_dTdQy,
        
        w0_dTdQy = l0_dTdQx # enforcing anisotropic beam to be the same for both measurements 
        l0_dTdQy = w0_dTdQx
        l0_dTdQg = w0_dTdQg # enforcing isotropic beam to be perfectly isotropic
        
        self.h = h 
        self.alpha = alpha

        eq1 = self.resid_dTdQ(kx, ky, g, Qx, dTdQ_x_arr, w0_dTdQx, l0_dTdQx, Ta)
        eq2 = self.resid_dTdQ(kx, ky, g, Qy, dTdQ_y_arr, w0_dTdQy, l0_dTdQy, Ta)
        eq3 = self.resid_dTdQ(kx, ky, g, Qg, dTdQ_g_arr, w0_dTdQg, l0_dTdQg, Ta)
        eqtns = np.concatenate([eq1, eq2, eq3])
        print(np.sum(eqtns), kx, ky, g)
        return eqtns
         
    def minimize_resid_full_varyall(self, x0, dTdQ_x, Qx,dTdQ_y, Qy, dTdQ_g, Qg, Ta, **ls_kwargs):
        
        root = least_squares(self.resid_full_varyall, x0=x0, args=(dTdQ_x, Qx,dTdQ_y, Qy, dTdQ_g, Qg, Ta),**ls_kwargs)
        
        return root
         


    def solve_kx(
        self, kx, dTdQ_x: float, Qx, ky: float, g: float, w0_dTdQx : float = 0.4E-6, l0_dTdQx : float = 0.4E-6, verbose: bool = False
    ):
        """
        Helper function for scipy.optimize.minimize to solve for kx independently

        Parameters
        ----------
        kx : float
            Independent variable for scipy.optimize.minimize. What we are solving for
        dTdQ_x : float
            Experimental value to fit model to and solve for kx
        Qx : array-like
            Range of input power values
        ky : float
            Value for ky at which we are solving for kx
        g : float
            Value for g at which we are solving for kx
        w0_dTdQx : float, default None 
            Value for w0 for this experiment 
        l0_dTdQx : float, default None 
            Value for l0 for this experiment 
        verbose : bool
            print progress, default False
        """
        if self.l0_dTdQx is not None: 
            l0_dTdQx = self.l0_dTdQx
        if self.w0_dTdQx is not None: 
            w0_dTdQx = self.w0_dTdQx

        # assert l0_dTdQx > w0_dTdQx , 'kx is solved when the laser is long in the y-direction, short in the x-direction ("lasery")'

        self.update_w0(w0_dTdQx)
        self.update_l0(l0_dTdQx)

        experimental_value = dTdQ_x

        guess = self.dTdQ(kx=kx, ky=ky, g=g, Q=Qx)
        if verbose:
            print("solving kx:", np.sqrt((guess - experimental_value) ** 2))
        return np.sqrt((guess - experimental_value) ** 2)

    def solve_ky(
        self, ky, dTdQ_y: float, Qy, kx: float, g: float, w0_dTdQy : float = 0.4E-6, l0_dTdQy : float = 0.4E-6, verbose: bool = False
    ):
        """
        Helper function for scipy.optimize.minimize to solve for ky independently

        Parameters
        ----------
        ky : float
            Independent variable for scipy.optimize.minimize. What we are solving for
        dTdQ_y : float
            Experimental value to fit model to and solve for ky
        Qy : array-like
            Range of input power values
        kx : float
            Value for ky at which we are solving for ky
        g : float
            Value for g at which we are solving for ky
        w0_dTdQy : float 
            Value for w0 for this experiment 
        l0_dTdQy : float 
            Value for l0 for this experiment 
        verbose : bool
            print progress, default False
        """
        if self.l0_dTdQy is not None: 
            l0_dTdQy = self.l0_dTdQy
        if self.w0_dTdQy is not None: 
            w0_dTdQy = self.w0_dTdQy

        # assert l0_dTdQy < w0_dTdQy , 'ky is solved when the laser is long in the x-direction, short in the y-direction ("laserx")'

        self.update_w0(w0_dTdQy)
        self.update_l0(l0_dTdQy)

        experimental_value = dTdQ_y
        guess = self.dTdQ(kx=kx, ky=ky, g=g, Q=Qy)
        if verbose:
            print("solving ky:", np.sqrt((guess - experimental_value) ** 2))
        return np.sqrt((guess - experimental_value) ** 2)

    def solve_g(
        self, g, dTdQ_g: float, Qg, kx: float, ky: float, w0_dTdQg : float  = 0.4E-6, l0_dTdQg : float  = 0.4E-6, verbose: bool = False
    ):
        """
        Helper function for scipy.optimize.minimize to solve for g independently
        TODO: This will probably actually be minimizing to TDTR data

        Parameters
        ----------
        g : float
            Independent variable for scipy.optimize.minimize. What we are solving for
        dTdQ_g : float
            Experimental value to fit model to and solve for g
        Qg : array-like
            Range of input power values
        kx : float
            Value for ky at which we are solving for g
        ky : float
            Value for g at which we are solving for g
        w0_dTdQg : float 
            Value for w0 for this experiment 
        l0_dTdQg : float 
            Value for l0 for this experiment 
        verbose : bool
            print progress, default False
        """
        if self.l0_dTdQg is not None: 
            l0_dTdQg = self.l0_dTdQg
        if self.w0_dTdQg is not None: 
            w0_dTdQg = self.w0_dTdQg

        self.update_w0(w0_dTdQg)
        self.update_l0(l0_dTdQg)

        experimental_value = dTdQ_g
        guess = self.dTdQ(kx=kx, ky=ky, g=g, Q=Qg)

        if verbose:
            print("solving g:", np.sqrt((guess - experimental_value) ** 2))
        return np.sqrt((guess - experimental_value) ** 2)
    
    def solve_k(
        self, k, dTdQ: float, Q, g: float, w0_dTdQg : float = 0.4E-6, l0_dTdQg : float = 0.4E-6, verbose: bool = False
    ):
        """
        Helper function for scipy.optimize.minimize to solve for kx independently

        Parameters
        ----------
        k : float
            Independent variable for scipy.optimize.minimize. What we are solving for
        dTdQ : float
            Experimental value to fit model to and solve for kx
        Q : array-like
            Range of input power values

        g : float
            Value for g at which we are solving for kx
        w0_dTdQg : float 
            Value for w0 for this experiment 
        l0_dTdQg : float 
            Value for l0 for this experiment  
        verbose : bool
            print progress, default False
        """
        if self.l0_dTdQg is not None: 
            l0_dTdQg = self.l0_dTdQg
        if self.w0_dTdQg is not None: 
            w0_dTdQg = self.w0_dTdQg

        self.update_w0(w0_dTdQg)
        self.update_l0(l0_dTdQg)

        experimental_value = dTdQ
        guess = self.dTdQ(kx=k, ky=k, g=g, Q=Q)

        if verbose:
            print("solving kx:", np.sqrt((guess - experimental_value) ** 2))
        return np.sqrt((guess - experimental_value) ** 2)
    
    def solve_anisotropic_raman(self, p, dTdQ_x, Qx, w0_dTdQx, l0_dTdQx, 
                                dTdQ_y, Qy, w0_dTdQy, l0_dTdQy,
                                dTdQ_g, Qg, w0_dTdQg, l0_dTdQg): 
        kx, ky, g = p 

        eq1 = self.solve_kx(kx, dTdQ_x, Qx, ky, g, w0_dTdQx, l0_dTdQx)
        eq2 = self.solve_ky(ky, dTdQ_y, Qy, kx, g, w0_dTdQy, l0_dTdQy)
        eq3 = self.solve_g(g, dTdQ_g, Qg, kx, ky, w0_dTdQg, l0_dTdQg)
        eqtns = np.concatenate([eq1, eq2, eq3])#is this generalizable?
        print(np.sum(eqtns), kx, ky, g/1E6)
        return eqtns
    
    def get_kx_ky_g(self, x0, dTdQ_x, Qx, w0_dTdQx, l0_dTdQx, 
                    dTdQ_y, Qy, w0_dTdQy, l0_dTdQy,
                    dTdQ_g, Qg, w0_dTdQg, l0_dTdQg, **ls_kwargs):
        
        root = least_squares(self.solve_anisotropic_raman, x0=x0, 
                      args=(dTdQ_x, Qx, w0_dTdQx, l0_dTdQx, 
                    dTdQ_y, Qy, w0_dTdQy, l0_dTdQy,
                    dTdQ_g, Qg, w0_dTdQg, l0_dTdQg), **ls_kwargs)
        return root 

    def solve_anisotropic_raman_varyall(self, p, dTdQ_x, Qx,dTdQ_y, Qy, dTdQ_g, Qg): 

        kx, ky, g, h, alpha, w0_dTdQx, l0_dTdQx, w0_dTdQg = p #w0_dTdQy, l0_dTdQy,
        w0_dTdQy = l0_dTdQx # enforcing anisotropic beam to be the same for both measurements 
        l0_dTdQy = w0_dTdQx
        l0_dTdQg = w0_dTdQg # enforcing isotropic beam to be perfectly isotropic
        
        self.h = h 
        self.alpha = alpha

        eq1 = self.solve_kx(kx, dTdQ_x, Qx, ky, g, w0_dTdQx, l0_dTdQx)
        eq2 = self.solve_ky(ky, dTdQ_y, Qy, kx, g, w0_dTdQy, l0_dTdQy)
        eq3 = self.solve_g(g, dTdQ_g, Qg, kx, ky, w0_dTdQg, l0_dTdQg)
        eqtns = np.concatenate([eq1, eq2, eq3])#is this generalizable?
        print(np.sum(eqtns), kx, ky, g/1E6)
        return eqtns

    def get_kx_ky_g_varyall(self, x0, dTdQ_x, Qx, dTdQ_y, Qy, dTdQ_g, Qg, **ls_kwargs):
        
        root = least_squares(self.solve_anisotropic_raman_varyall, x0=x0, 
                      args=(dTdQ_x, Qx, dTdQ_y, Qy, dTdQ_g, Qg, ), **ls_kwargs)
        return root  
    
    def solve_anisotropic_raman_kxky_varyall(self, p, dTdQ_x, Qx,dTdQ_y, Qy,g): 
        # hold g fixed

        kx, ky, h, alpha, w0_dTdQx, l0_dTdQx = p #w0_dTdQy, l0_dTdQy,
        w0_dTdQy = l0_dTdQx # enforcing anisotropic beam to be the same for both measurements 
        l0_dTdQy = w0_dTdQx
        
        self.h = h 
        self.alpha = alpha

        eq1 = self.solve_kx(kx, dTdQ_x, Qx, ky, g, w0_dTdQx, l0_dTdQx)
        eq2 = self.solve_ky(ky, dTdQ_y, Qy, kx, g, w0_dTdQy, l0_dTdQy)
        eqtns = np.concatenate([eq1, eq2])#is this generalizable?
        print(np.sum(eqtns), kx, ky)
        return eqtns

    def get_kx_ky_varyall(self, x0, dTdQ_x, Qx, dTdQ_y, Qy, g, **ls_kwargs):
        
        root = least_squares(self.solve_anisotropic_raman_kxky_varyall, x0=x0, 
                      args=(dTdQ_x, Qx, dTdQ_y, Qy, g), **ls_kwargs)
        return root  
    
    
    def solve_isotropic_raman(self, p, dTdQ_array, Q, r_array):
        k, g = p 
        eqns = []
        for r, dTdQ in zip(r_array, dTdQ_array): 
            # eqns.append(self.solve_k(k, dTdQ, Q, k, g, r, r))
            eqns.append(self.solve_g(g, dTdQ, Q, k, g, r, r))
        return tuple(eqns) 
    
    def get_k_g(self, x0, dTdQ_array, Q, r_array, **ls_kwargs): 
        root = least_squares(self.solve_isotropic_raman, x0=x0, args=(dTdQ_array, Q, r_array), **ls_kwargs)
        return root 



    def iter_solve(
        self,
        kxg: float,
        kyg: float,
        gg: float,
        dTdQ_x: float,
        dTdQ_y: float,
        dTdQ_g: float,
        Qx,
        threshold=1,
        verbose=False,
        tolerance=1e-6,
        method="Nelder-Mead",
        kx_bounds=((0.0, 500),),
        ky_bounds=((0.0, 500.0),),
        g_bounds=((0.5e6, 30e6),),
        Qy=None,
        Qg=None,
        wy=None,
        ly=None,
        wx=None,
        lx=None,
    ):
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
        Qx : array-like
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
        Qg : array-like
            Optional if need to update Q when solving for g, default None
        """
        warnings.warn("iter_solve is deprecated, use new_function instead", DeprecationWarning, stacklevel=2)

        error_kx = error_ky = 10
        error_g = 1e3

        dTdQ_x_perc = 10
        dTdQ_y_perc = 10
        dTdQ_g_perc = 10

        count = 0
        # We need to remember to flip the orientation of the laser depending on what we are solving for
        # we may also need to update the Qarr if slightly different Q for different measurements

        if Qy is None:
            Qy = Qx
        if Qg is None:
            Qg = Qx

        if wy is None and wx is None:
            widths = [self.w0, self.l0]
            widths.sort()
            width_short, width_long = widths
            wx = width_short  # for solving kx, "lasery"
            lx = width_long  # for solving kx, "lasery"

            wy = width_long  # for solving ky, "laserx"
            ly = width_short  # for solving ky, "laserx"
        print(wx,lx, wy, ly)
        # I don't think I want the kx and ky that are conver
        while (
            dTdQ_x_perc > threshold
            or dTdQ_y_perc > threshold
            or dTdQ_g_perc > threshold
        ) or (error_kx > 5 or error_ky > 5 or error_g > 1e4):
            # first, assume kyg (and gg always for now) are correct, solve for kx
            print("----------- assuming ky, solving kx ---------------")

            # when solving for kx, we are using dTdQ from when the laser is aligned with the y axis

            sol = minimize(
                self.solve_kx,
                kxg,
                args=(dTdQ_x, Qx, kyg, gg, wx, lx, verbose),
                tol=tolerance,
                method=method,
                bounds=kx_bounds,
            )
            kx_solved = sol.x[0]
            dTdQ_x_solved = self.dTdQ(kx_solved, kyg, gg, Qx)
            dTdQ_x_err = np.sqrt((dTdQ_x - dTdQ_x_solved) ** 2)
            dTdQ_x_perc = np.abs(dTdQ_x - dTdQ_x_solved) / dTdQ_x

            error_kx = np.sqrt((kxg - kx_solved) ** 2)
            kxg = kx_solved

            # assume kxg solution is correct, and solve for ky
            print(
                f"----------- kx updated, kx = {kxg}, kx_error = {error_kx}, dTdQ_x % error = {dTdQ_x_perc}, solving ky ---------------"
            )

            # when solving for ky, we are using dTdQ from when the laser is aligned with the x axis

            sol = minimize(
                self.solve_ky,
                kyg,
                args=(dTdQ_y, Qy, kxg, gg, wy, ly, verbose),
                tol=tolerance,
                method=method,
                bounds=ky_bounds,
            )
            ky_solved = sol.x[0]
            dTdQ_y_solved = self.dTdQ(kxg, ky_solved, gg, Qy)
            dTdQ_y_err = np.sqrt((dTdQ_y - dTdQ_y_solved) ** 2)
            dTdQ_y_perc = np.abs(dTdQ_y - dTdQ_y_solved) / dTdQ_y

            error_ky = np.sqrt((kyg - ky_solved) ** 2)

            kyg = ky_solved

            print(
                f"----------- ky updated, ky = {kyg}, ky_error = {error_ky}, dTdQ_y % error = {dTdQ_y_perc}, solving g ---------------"
            )

            # to solve for g, we use dTdQ from the isotropic raman measurements

            sol = minimize(
                self.solve_g,
                gg,
                args=(dTdQ_g, Qg, kxg, kyg, self.w0_dTdQg, self.l0_dTdQg, verbose),
                tol=tolerance,
                method="Nelder-Mead",
                bounds=g_bounds,
            )
            g_solved = sol.x[0]
            dTdQ_g_solved = self.dTdQ(kxg, kyg, g_solved, Qg)
            dTdQ_g_err = np.sqrt((dTdQ_g - dTdQ_g_solved) ** 2)
            dTdQ_g_perc = np.abs(dTdQ_g - dTdQ_g_solved) / dTdQ_g

            error_g = np.sqrt((gg - g_solved) ** 2)
            gg = g_solved
            print(
                f"----------- g updated, g = {gg}, g_error = {error_g}, dTdQ_g % error = {dTdQ_g_perc} ---------------"
            )

            count += 1
            print(
                f"----------- iter {count}, kx = {kxg}, ky = {kyg}, g = {gg}, error_kx = {error_kx}, error_g = {error_g} error_ky = {error_ky}, dTdQ_x % err = {dTdQ_x_perc}, dTdQ_y % err = {dTdQ_y_perc}, dTdQ_g % err = {dTdQ_g_perc}------"
            )

        print(f"converged at iter {count}, kx = {kxg}, ky = {kyg}, g = {gg}")

        self.kx_solved = kxg
        self.ky_solved = kyg
        self.g_solved = gg
