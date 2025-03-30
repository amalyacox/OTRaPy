# Author : Amalya Cox Johnson
# email : amalyaj@stanford.edu

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import curve_fit, minimize, least_squares
import warnings
from OTRaPy.utils import *
from scipy.interpolate import interp1d
from OTRaPy.Solvers.analytic_solver import * 


class RamanSolver:
    """
    Solver for 2D Diffusion in Raman Thermometry Measurements
    Default materials parameters come from Taube, et. al, MoS2 on SiO2/Si
    
    Attributes 
    ----------
    lx : float 
        x-dimension of grid in meters 
    ly : float 
        y-dimension of grid in meters 
    nx : int 
        number of points along x 
    ny : int 
        number of points along y
    h : float 
        thickness of sample 
    alpha : float 
        absorption coefficient of sample 
    w0 : float 
        gaussian beam width along x, able to take arrays for z-sweep measurements 
    l0 : float 
        gaussian beam width along y, able to take arrays for z-sweep measurements 
    w0_dTdQg : float, default None 
        for anisotropic measurement, isotropic gaussian beam width along x
    l0_dTdQg : float, default None
        for anisotropic measurement, isotropic gaussian beam width along y
    w0_dTdQx : float, default None 
        for anisotropic measurement, for solving kx, anisotropic gaussian beam width along x (w0 when laser parallel to y, "lasery")
    l0_dTdQx : float, default None 
        for anisotropic measurement, for solving kx, anisotropic gaussian beam width along y (l0 when laser parallel to y, "lasery")
    w0_dTdQy : float, default None
        for anisotropic measurement, for solving ky, anisotropic gaussian beam width along x (w0 when laser parallel to x, "laserx")
    l0_dTdQy : float, default None 
        for anisotropic measurement, for solving ky, anisotropic gaussian beam width along y (l0 when laser parallel to x, "laserx")


    Methods 
    -------
    dTdQ
    dTdQ_sim
    generate_qdot
    get_g_fixk
    get_k_g
    get_k_g_varyall
    get_kx_ky_fixg
    get_kx_ky_g
    get_kx_ky_g_curve
    get_kx_ky_g_varyall
    get_kx_ky_varyall
    iter_solve
    mc_sim_resid_full
    mc_sim_resid_full_varyall
    minimize_resid_fixg
    minimize_resid_fixk
    minimize_resid_full
    minimize_resid_full_isotropic
    minimize_resid_full_isotropic_varyall
    minimize_resid_full_varyall
    resid_dTdQ
    resid_fixg
    resid_fixk
    resid_full
    resid_full_isotropic
    resid_full_isotropic_varyall
    resid_full_varyall
    solve_anisotropic_raman
    solve_anisotropic_raman_curve
    solve_anisotropic_raman_fixg
    solve_anisotropic_raman_kxky_varyall
    solve_anisotropic_raman_varyall',
    solve_g
    solve_isotropic_raman
    solve_isotropic_raman_fixk
    solve_isotropic_raman_varyall
    solve_k
    solve_kx
    solve_ky
    update_alpha
    update_l0
    update_w0
    weighted_average


    """

    def __init__(
        self,
        lx: float = 10e-6,
        ly: float = 10e-6,
        nx: int = 200,
        ny: int = 200,
        h: float = 0.7e-9,
        alpha : float = 0.04, 
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
        self.alpha = alpha  # material % absorption at laser wavelength (%) with laser orientation 

        
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
        """
        Update w0 to new parameter, matching shapes for proper implementation 

        Parameters 
        ----------
        new_w0 : float or np.ndarray 
            new value for w0  
        """
        if type(new_w0) == float or type(new_w0) == np.float64: 
            self.w0 = np.array([new_w0])
        else: 
            self.w0 = new_w0
        self.w0 = np.broadcast_to(self.w0, (self.nx, self.ny, len(self.w0))).T

    def update_l0(self, new_l0):
        """
        Update l0 to new parameter, matching shapes for proper implementation 

        Parameters 
        ----------
        new_l0 : float or np.ndarray 
            new value for l0
        """
        if type(new_l0) == float or type(new_l0) == np.float64: 
            self.l0 = np.array([new_l0])
        else: 
            self.l0 = new_l0
        self.l0 = np.broadcast_to(self.l0, (self.nx, self.ny, len(self.l0))).T

    def update_alpha(self, new_alpha : float): 
        """
        Update alpha to new parameter

        Parameters 
        ----------
        new_alpha : float 
            new value for alpha 
        """
        if new_alpha is None: 
            pass
        else: 
            self.alpha = new_alpha 

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
        d2h=None
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

    def analytic_Txy(self, kx=62.2,ky=62.2,g=1.94E6,Q=1e-3,Ta=300,upper=10e-6):
        """
        Analytically solve for T(r) using solution to cylindrical laser heating equation 
        """
        T = []
        rr = np.arange(0, self.lx, self.delta)

        r0 = self.w0[0,0,0]
        const = (Q * self.alpha) / (kx * self.h * np.pi * r0**2)
        # Turn this into a meshgrid 
        for r in rr: 
        # We are integrating over all space (y) for each value of r. Limits of integration found to not break the code
            sol = const * integrate.quad(integrand_T, 0, upper, args=(r,kx, g, r0, self.h))[0]
            T.append(sol)
        T = np.array(T)

        R  = np.sqrt(self.X**2 + self.Y**2)
        r_values = np.arange(0,self.lx,self.delta)
        T = np.interp(R.flatten(), r_values, T)
        T = T.reshape(self.X.shape) + Ta

        return T

    def weighted_average_analytic(self, k=62.2,g=1.94E6,Q=1e-3,ratio=None,upper=10e-6): 
        r0 = self.w0[0,0,0]

        def integrand(r, Q,k,h,g,ratio,r0=r0):
            const = (Q*self.alpha) / (k * h * np.pi * r0**2)
            if ratio is not None: 
                const = 1 / (r0**2)
                T = const * integrate.quad(integrand_T_solve, 0, upper, args=(r, ratio, r0, self.h))[0] 
            else: 
                T = const * integrate.quad(integrand_T, 0, upper, args=(r, k, g, r0, self.h))[0] 
            T = T * r * np.exp(-r**2/(r0**2))
            # popt, pcov = curve_fit(gaussian, r,T, p0=[np.max(T),0,r0,300])
            # T = gaussian(r, *popt)
            return T 
        
        if ratio is not None: 
            Tav = integrate.quad(integrand, 0, 10e-6, args=(Q,k,self.h,g,ratio))[0] * 1/(r0**2)
        else: 
            Tav = integrate.quad(integrand, 0, 10e-6, args=(Q,k,self.h,g,ratio))[0] / (r0**2/2)
        return Tav

    def dTdQ(self, kx: float, ky: float, g: float, Q, **kwargs):
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
        """
        T1 = self.Txy(kx=kx, ky=ky, g=g, Q=Q[0], **kwargs)
        Tav1 = self.weighted_average(T1)
        T2 = self.Txy(kx=kx, ky=ky, g=g, Q=Q[-1], **kwargs)
        Tav2 = self.weighted_average(T2)

        slope = (Tav2 - Tav1) / (Q[-1] - Q[0])
        
        return slope
        
    def dTdQ_sim(self, kx : float, ky : float, g : float, Q, w0=None, l0=None, alpha=None, Ta=300):
        """
        Generate dTdQ for materials parameters and simulate experimental T vs. Q 

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
        w0 : float, optional 
            w0 for simulation, default None passes to current w0  
        l0 : float, optional 
            l0 for simulation, default None passes to current l0 
        alpha : float, optional 
            alpha for simulation, default None passes to current alpha 
        Ta : float or int, optional 
            stage temperature for simulation, default None passes to 300K  

        """
        """
        Doesn't work for dwdz array? 
        """
        if w0 is not None: 
            self.update_l0(l0)
        if l0 is not None: 
            self.update_w0(w0)
        if alpha is not None: 
            self.update_alpha(alpha)
        
        dTdQ = self.dTdQ(kx,ky,g,Q)
        sim = linear(Q, dTdQ, Ta)

        return sim
    
    def resid_dTdQ(self, kx, ky, g, Q, dT_arr, dT_err, w0=None,l0=None,alpha=None,Ta=300, off=0): 
        """
        Calculate residual between simulated dTdQ and experimental dTdQ curves

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
        dTdQ_arr : array-like 
            Experimental dT values 
        dT_err : array_like 
            TODO: Errors on experimental dT values ; can be used to weight residuals but currently not
        w0 : float, optional 
            w0 for simulation, default None passes to current w0  
        l0 : float, optional 
            l0 for simulation, default None passes to current l0 
        alpha : float, optional 
            alpha for simulation, default None passes to current alpha 
        Ta : float or int, optional 
            stage temperature for simulation, default None passes to 300K  

        """
        sim = self.dTdQ_sim(kx, ky, g, Q, w0, l0, alpha, Ta)
        sim += off 

        return (dT_arr - sim)**2 #/dT_err 
    
    def analytic_loss(self, ratio, dTdQ_r1, dTdQ_r2, r1, r2):
        experimental_ratio = dTdQ_r1/dTdQ_r2
        self.update_w0(r1)
        self.update_l0(r1)
        Tm1 = self.weighted_average_analytic(ratio=ratio)

        self.update_w0(r2)
        self.update_l0(r2)
        Tm2 = self.weighted_average_analytic(ratio=ratio)
        print(np.sqrt((Tm1/Tm2 - experimental_ratio)**2))

        return np.sqrt((Tm1/Tm2 - experimental_ratio)**2)
    
    def get_ratio(self, dTdQ_1, dTdQ_2, r1, r2, x0, **ls_kwargs): 
        root = minimize(self.analytic_loss,  x0=x0, 
                        args=(dTdQ_1, dTdQ_2, r1, r2),
                        **ls_kwargs)
        self.found_ratio = root.x[0]
        return root 
    
    def solve_k_from_ratio(self, k, ratio, r, dTdQ, Q, verbose=False): 
        g = k*ratio
        self.update_w0(r)
        self.update_l0(r)
        experimental_value = dTdQ
        guess = self.dTdQ(kx=k,ky=k,g=g,Q=Q)
        if verbose: 
            print(np.sqrt((guess-experimental_value)**2))
        return np.sqrt((guess-experimental_value)**2)
    
    def get_props_analytic(self, x0, ratio, r1, r2, dTdQ_1, dTdQ_2, Q, verbose=False, **ls_kwargs): 
        if verbose: 
            print('solving k1')
        root_k1 = least_squares(self.solve_k_from_ratio, x0=x0, args=(ratio,r1, dTdQ_1, Q, verbose), **ls_kwargs)
        if verbose: 
            print('solving k2')
        root_k2 = least_squares(self.solve_k_from_ratio, x0=x0, args=(ratio, r2, dTdQ_2, Q, verbose), **ls_kwargs)

        k1 = root_k1.x[0]
        g1 = k1 * ratio 

        k2 = root_k2.x[0]
        g2 = k2 * ratio 

        k_av = (k1 + k2)/2
        g_av = (g1 + g2)/2
        return k1, g1, k2, g2, k_av, g_av





    #####################################################################################################
    # Functions to perform least squares curve fitting between simulation and experimental points
    def resid_full(self, p, dTdQ_x_arr, dTdQ_x_err, Qx, w0_dTdQx, l0_dTdQx, alpha_x,
                                dTdQ_y_arr, dTdQ_y_err, Qy, w0_dTdQy, l0_dTdQy, alpha_y,
                                dTdQ_g_arr, dTdQ_g_err, Qg, w0_dTdQg, l0_dTdQg, alpha_g,
                                Ta): 
        """
        Helper function for minimizing the residual between dTdQ simulations and 3 dT curves (isotropic, laserx, lasery)
        
        Parameters
        ----------
        p : array-like 
            kx,ky,g; parameters to minimize to
        dTdQ_x_arr : array-like 
            experimental dT for solving for kx (lasery)
        dTdQ_x_err: array-like 
            error on dTdQ_x_arr
        Qx : array-like
            powers corresponding to dTdQ_x_arr 
        w0_dTdQx : float 
            w0 for dTdQ_x_arr
        l0_dTdQx : float 
            l0 for dTdQ_x_arr 
        alpha_x : float 
            alpha for dTdQ_x_arr (for this orientation of laser)
        dTdQ_y_arr : array-like 
            experimental dT for solving for ky (laserx)
        dTdQ_y_err: array-like 
            error on dTdQ_y_arr
        Qy : array-like
            powers corresponding to dTdQ_y_arr 
        w0_dTdQy : float 
            w0 for dTdQ_y_arr
        l0_dTdQy : float 
            l0 for dTdQ_y_arr 
        alpha_y : float 
            alpha for dTdQ_y_arr (for this orientation of laser)
        dTdQ_g_arr : array-like 
            experimental dT for solving for kg (isolaser)
        dTdQ_g_err: array-like 
            error on dTdQ_g_arr
        Qg : array-like
            powers corresponding to dTdQ_g_arr 
        w0_dTdQg : float 
            w0 for dTdQ_g_arr
        l0_dTdQg : float 
            l0 for dTdQ_g_arr 
        alpha_g : float 
            alpha for dTdQ_g_arr (for this orientation of laser)
        Ta : int or float 
            substrate temperature 
        """
        
        kx, ky, g = p
        offx, offy, offg = 0, 0, 0
        
        eq1 = self.resid_dTdQ(kx, ky, g, Qx, dTdQ_x_arr, dTdQ_x_err, w0_dTdQx, l0_dTdQx, alpha_x, Ta, offx)
        eq2 = self.resid_dTdQ(kx, ky, g, Qy, dTdQ_y_arr, dTdQ_y_err, w0_dTdQy, l0_dTdQy, alpha_y, Ta, offy)
        eq3 = self.resid_dTdQ(kx, ky, g, Qg, dTdQ_g_arr, dTdQ_g_err, w0_dTdQg, l0_dTdQg, alpha_g, Ta, offg)
        
        eqtns = np.concatenate([eq1, eq2, eq3])
        
        print(np.sum(eqtns), kx, ky, g)
        return eqtns

    def minimize_resid_full(self, x0, dTdQ_x_arr, dTdQ_x_err, Qx, w0_dTdQx, l0_dTdQx, alpha_x,
                                dTdQ_y_arr, dTdQ_y_err, Qy, w0_dTdQy, l0_dTdQy, alpha_y,
                                dTdQ_g_arr, dTdQ_g_err, Qg, w0_dTdQg, l0_dTdQg, alpha_g,
                                  Ta,**ls_kwargs):
        """
        Least squares minimization between simulation and experimental curves for isotropic + anisotropic measurement
        
        Parameters
        ----------
        x0 : array-like 
            kx,ky,g initial guesses 
        dTdQ_x_arr : array-like 
            experimental dT for solving for kx (lasery)
        dTdQ_x_err: array-like 
            error on dTdQ_x_arr
        Qx : array-like
            powers corresponding to dTdQ_x_arr 
        w0_dTdQx : float 
            w0 for dTdQ_x_arr
        l0_dTdQx : float 
            l0 for dTdQ_x_arr 
        alpha_x : float 
            alpha for dTdQ_x_arr (for this orientation of laser)
        dTdQ_y_arr : array-like 
            experimental dT for solving for ky (laserx)
        dTdQ_y_err: array-like 
            error on dTdQ_y_arr
        Qy : array-like
            powers corresponding to dTdQ_y_arr 
        w0_dTdQy : float 
            w0 for dTdQ_y_arr
        l0_dTdQy : float 
            l0 for dTdQ_y_arr 
        alpha_y : float 
            alpha for dTdQ_y_arr (for this orientation of laser)
        dTdQ_g_arr : array-like 
            experimental dT for solving for kg (isolaser)
        dTdQ_g_err: array-like 
            error on dTdQ_g_arr
        Qg : array-like
            powers corresponding to dTdQ_g_arr 
        w0_dTdQg : float 
            w0 for dTdQ_g_arr
        l0_dTdQg : float 
            l0 for dTdQ_g_arr 
        alpha_g : float 
            alpha for dTdQ_g_arr (for this orientation of laser)
        Ta : int or float 
            substrate temperature 
        ls_kwargs : dict 
            keyword arguments for scipy.optimize.least_squares

        """
        
        root = least_squares(self.resid_full, x0=x0, args=(dTdQ_x_arr, dTdQ_x_err, Qx, w0_dTdQx, l0_dTdQx, alpha_x,
                                dTdQ_y_arr, dTdQ_y_err, Qy, w0_dTdQy, l0_dTdQy, alpha_y,
                                dTdQ_g_arr, dTdQ_g_err, Qg, w0_dTdQg, l0_dTdQg, alpha_g, Ta),**ls_kwargs)
        return root

    def resid_fixg(self, p, dTdQ_x_arr, dTdQ_x_err, Qx, w0_dTdQx, l0_dTdQx, alpha_x,
                                dTdQ_y_arr,dTdQ_y_err,  Qy, w0_dTdQy, l0_dTdQy, alpha_y,
                                g,Ta): 
        """
        Helper function for minimizing the residual between dTdQ simulations and 2 dT curves (laserx, lasery),
        fixing g 

        Parameters 
        ----------
        See resid_full; just removes all g parameters 

        g : float
            Fixed value for g 
        """
        kx, ky = p
        eq1 = self.resid_dTdQ(kx, ky, g, Qx, dTdQ_x_arr, dTdQ_x_err, w0_dTdQx, l0_dTdQx, alpha_x, Ta)
        eq2 = self.resid_dTdQ(kx, ky, g, Qy, dTdQ_y_arr, dTdQ_y_err, w0_dTdQy, l0_dTdQy, alpha_y, Ta)
        eqtns = np.concatenate([eq1, eq2])
        print(np.sum(eqtns), kx, ky)
        return eqtns

    def minimize_resid_fixg(self, x0, dTdQ_x_arr, dTdQ_x_err, Qx, w0_dTdQx, l0_dTdQx, alpha_x,
                                dTdQ_y_arr, dTdQ_y_err, Qy, w0_dTdQy, l0_dTdQy, alpha_y, g,Ta,
                                 **ls_kwargs):
        """
        Least squares minimization between simulation and experimental curves for anisotropic measurement

        Patameters 
        ---------
        See minimize_resid_full; just removes all g parameters

        g : float
            Fixed value for g 
        """
        root = least_squares(self.resid_fixg, x0=x0, args=(dTdQ_x_arr, dTdQ_x_err, Qx, w0_dTdQx, l0_dTdQx, alpha_x,
                                dTdQ_y_arr, dTdQ_y_err, Qy, w0_dTdQy, l0_dTdQy, alpha_y, 
                                g,Ta),**ls_kwargs)
        return root    
    
    def resid_fixk(self, p, dTdQ_g_arr, dTdQ_g_err, Qg, w0_dTdQg, l0_dTdQg,alpha_g, 
                                kx,ky,Ta): 
        """
        Helper function for minimizing the residual between dTdQ simulations and 1 dT curve (isolaser)

        Parameters 
        ----------
        See resid_full; keeps only g parameters 
        kx : float 
            Fixed value for kx 
        ky : float 
            Fixed value for ky 
        """
        g = p
        eq1 = self.resid_dTdQ(kx, ky, g, Qg, dTdQ_g_arr, dTdQ_g_err, w0_dTdQg, l0_dTdQg, alpha_g, Ta)
        # print(np.sum(eq1), g)
        return eq1

    def minimize_resid_fixk(self, x0, dTdQ_g_arr, dTdQ_g_err,Qg, w0_dTdQg, l0_dTdQg,alpha_g, 
                                kx,ky,Ta, **ls_kwargs):
        """
        Least squares minimization between simulation and experimental curves for anisotropic measurement
        
        Parameters 
        ----------
        See minimize_resid_full; keeps only g parameters 
        kx : float 
            Fixed value for kx 
        ky : float 
            Fixed value for ky
        """
        root = least_squares(self.resid_fixk, x0=x0, args=(dTdQ_g_arr, dTdQ_g_err,Qg, w0_dTdQg, l0_dTdQg,alpha_g, 
                                kx,ky,Ta),**ls_kwargs)
        return root
    
    def resid_full_varyall(self, p, dTdQ_x_arr, dTdQ_x_err, Qx,dTdQ_y_arr, dTdQ_y_err, Qy, dTdQ_g_arr, dTdQ_g_err,Qg): 
        """
        Helper function for minimizing the residual between dTdQ simulations and 3 dT curves (isolaser, laserx, lasery)
        Allows all parameters to vary 

        Parameters 
        ----------
        p : array-like 
            Parameters to minimize to
            kx, ky, g, h, alpha_x, alpha_y, alpha_g, w0_dTdQx, l0_dTdQx,w0_dTdQy, l0_dTdQy, w0_dTdQg, l0_dTdQg 
        dTdQ_x_arr : array-like 
            experimental dT for solving for kx (lasery)
        dTdQ_x_err: array-like 
            error on dTdQ_x_arr
        dTdQ_y_arr : array-like 
            experimental dT for solving for ky (laserx)
        dTdQ_y_err: array-like 
            error on dTdQ_y_arr
        dTdQ_g_arr : array-like 
            experimental dT for solving for g (isolaser)
        dTdQ_g_err: array-like 
            error on dTdQ_g_arr
        Ta : int or float 
            substrate temperature 
        """
        
        kx, ky, g, h, alpha_x, alpha_y, alpha_g, w0_dTdQx, l0_dTdQx,w0_dTdQy, l0_dTdQy, w0_dTdQg, l0_dTdQg, Ta = p #w0_dTdQy, l0_dTdQy,
        offx, offy, offg = 0,0,0

        # enforcing anisotropic beam to be the same for both measurements 
        # w0_dTdQy = l0_dTdQx 
        # l0_dTdQy = w0_dTdQx
        # enforcing isotropic beam to be perfectly isotropic
        # l0_dTdQg = w0_dTdQg 
        
        self.h = h 

        eq1 = self.resid_dTdQ(kx, ky, g, Qx, dTdQ_x_arr, dTdQ_x_err, w0_dTdQx, l0_dTdQx, alpha_x, Ta, offx)
        eq2 = self.resid_dTdQ(kx, ky, g, Qy, dTdQ_y_arr, dTdQ_y_err, w0_dTdQy, l0_dTdQy, alpha_y, Ta, offy)
        eq3 = self.resid_dTdQ(kx, ky, g, Qg, dTdQ_g_arr, dTdQ_g_err, w0_dTdQg, l0_dTdQg, alpha_g, Ta, offg)
        eqtns = np.concatenate([eq1, eq2, eq3])
        # print(np.sum(eqtns), kx, ky, g,)
        return eqtns
         
    def minimize_resid_full_varyall(self, x0, dTdQ_x_arr, dTdQ_x_err, Qx,dTdQ_y_arr, dTdQ_y_err, Qy, dTdQ_g_arr, dTdQ_g_err,Qg,**ls_kwargs):
        """
        Least squares minimization between simulation and experimental curves for 3 dT curves when varying all parameters 

        Parameters 
        ----------
        x0 : array-like 
            Initial guess 
            kx, ky, g, h, alpha_x, alpha_y, alpha_g, w0_dTdQx, l0_dTdQx,w0_dTdQy, l0_dTdQy, w0_dTdQg, l0_dTdQg 
        dTdQ_x_arr : array-like 
            experimental dT for solving for kx (lasery)
        dTdQ_x_err: array-like 
            error on dTdQ_x_arr
        dTdQ_y_arr : array-like 
            experimental dT for solving for ky (laserx)
        dTdQ_y_err: array-like 
            error on dTdQ_y_arr
        dTdQ_g_arr : array-like 
            experimental dT for solving for g (isolaser)
        dTdQ_g_err: array-like 
            error on dTdQ_g_arr
        Ta : int or float 
            substrate temperature
        ls_kwargs : dict 
            keyword arguments for scipy.optimize.least_squares
        """
        root = least_squares(self.resid_full_varyall, x0=x0, args=(dTdQ_x_arr, dTdQ_x_err, Qx,dTdQ_y_arr, dTdQ_y_err, Qy, dTdQ_g_arr, dTdQ_g_err,Qg),**ls_kwargs)
        return root

    #################################################################################################################################################################
    # Functions for solving using just slopes 
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
        return ((guess - experimental_value) ** 2) #got rid of sqrt

    def solve_ky(
        self, ky, dTdQ_y: float, Qy, kx: float, g: float, w0_dTdQy : float = 0.4E-6, l0_dTdQy : float = 0.4E-6, alpha=None, verbose: bool = False
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
        self.update_alpha(alpha)

        experimental_value = dTdQ_y
        guess = self.dTdQ(kx=kx, ky=ky, g=g, Q=Qy)
        if verbose:
            print("solving ky:", np.sqrt((guess - experimental_value) ** 2))
        return ((guess - experimental_value) ** 2)

    def solve_g(
        self, g, dTdQ_g: float, Qg, kx: float, ky: float, w0_dTdQg : float  = 0.4E-6, l0_dTdQg : float  = 0.4E-6, alpha=None, verbose: bool = False
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
        self.update_alpha(alpha)

        experimental_value = dTdQ_g
        guess = self.dTdQ(kx=kx, ky=ky, g=g, Q=Qg)

        if verbose:
            print("solving g:", np.sqrt((guess - experimental_value) ** 2))
        return ((guess - experimental_value) ** 2)
    
    def solve_k(
        self, k, dTdQ: float, Q, g: float, w0_dTdQg : float = 0.4E-6, l0_dTdQg : float = 0.4E-6, alpha=None, verbose: bool = False
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
        self.update_alpha(alpha)

        experimental_value = dTdQ
        guess = self.dTdQ(kx=k, ky=k, g=g, Q=Q)

        if verbose:
            print("solving kx:", np.sqrt((guess - experimental_value) ** 2))
        return np.sqrt((guess - experimental_value) ** 2)
    
    def solve_anisotropic_raman(self, p, dTdQ_x, Qx, w0_dTdQx, l0_dTdQx, alpha_x,
                                dTdQ_y, Qy, w0_dTdQy, l0_dTdQy, alpha_y,
                                dTdQ_g, Qg, w0_dTdQg, l0_dTdQg, alpha_g): 
        """
        Helper function for solving kx,ky,g using experimental slopes from isotropic, laserx, lasery experiments
        Different from functions for minimizing to residuals as now we are just using the slopes, not all the points 

        Parameters
        ----------
        p : array-like 
            kx,ky,g ; parameters to minimize to 
        dTdQ_x : float 
            dTdQ slope for measurement used to solve for kx (lasery)
        Qx : array-like
            powers used for dTdQ_x 
        w0_dTdQx : float 
            w0 for dTdQ_x
        l0_dTdQx : float 
            l0 for dTdQ_x 
        alpha_x : float 
            alpha for dTdQ_x (for this orientation of laser)
        dTdQ_y : float 
            dTdQ slope for measurement used to solve for ky (laserx)
        Qy : array-like
            powers used for dTdQ_y 
        w0_dTdQy : float 
            w0 for dTdQ_y
        l0_dTdQy : float 
            l0 for dTdQ_y
        alpha_y : float 
            alpha for dTdQ_y (for this orientation of laser)
        dTdQ_g : float 
            dTdQ slope for measurement used to solve for ky (laserx)
        Qg : array-like
            powers used for dTdQ_g 
        w0_dTdQg : float 
            w0 for dTdQ_g
        l0_dTdQg : float 
            l0 for dTdQ_g
        alpha_g : float 
            alpha for dTdQ_g (for this orientation of laser)
        
        """
        kx, ky, g = p 

        eq1 = self.solve_kx(kx, dTdQ_x, Qx, ky, g, w0_dTdQx, l0_dTdQx, alpha_x)
        eq2 = self.solve_ky(ky, dTdQ_y, Qy, kx, g, w0_dTdQy, l0_dTdQy, alpha_y)
        eq3 = self.solve_g(g, dTdQ_g, Qg, kx, ky, w0_dTdQg, l0_dTdQg, alpha_g)
        eqtns = np.concatenate([eq1, eq2, eq3])#is this generalizable?
        # print(np.sum(eqtns), kx, ky, g/1E6)
        return eqtns
    
    def get_kx_ky_g(self, x0, dTdQ_x, Qx, w0_dTdQx, l0_dTdQx, alpha_x, 
                    dTdQ_y, Qy, w0_dTdQy, l0_dTdQy, alpha_y,
                    dTdQ_g, Qg, w0_dTdQg, l0_dTdQg, alpha_g, **ls_kwargs):
        """
        Least squares minimization on slopes from isotropic, laserx, lasery experiments 
        Parameters
        ----------
        x0 : array-like 
            Parameters initial guess 
        dTdQ_x : float 
            dTdQ slope for measurement used to solve for kx (lasery)
        Qx : array-like
            powers used for dTdQ_x 
        w0_dTdQx : float 
            w0 for dTdQ_x
        l0_dTdQx : float 
            l0 for dTdQ_x 
        alpha_x : float 
            alpha for dTdQ_x (for this orientation of laser)
        dTdQ_y : float 
            dTdQ slope for measurement used to solve for ky (laserx)
        Qy : array-like
            powers used for dTdQ_y 
        w0_dTdQy : float 
            w0 for dTdQ_y
        l0_dTdQy : float 
            l0 for dTdQ_y
        alpha_y : float 
            alpha for dTdQ_y (for this orientation of laser)
        dTdQ_g : float 
            dTdQ slope for measurement used to solve for ky (laserx)
        Qg : array-like
            powers used for dTdQ_g 
        w0_dTdQg : float 
            w0 for dTdQ_g
        l0_dTdQg : float 
            l0 for dTdQ_g
        alpha_g : float 
            alpha for dTdQ_g (for this orientation of laser)
        ls_kwargs : dict 
            keyword arguments for scipy.optimize.least_squares
        """
        
        root = least_squares(self.solve_anisotropic_raman, x0=x0, 
                      args=(dTdQ_x, Qx, w0_dTdQx, l0_dTdQx, alpha_x, 
                    dTdQ_y, Qy, w0_dTdQy, l0_dTdQy,alpha_y, 
                    dTdQ_g, Qg, w0_dTdQg, l0_dTdQg, alpha_g), **ls_kwargs)
        return root 

    def solve_anisotropic_raman_fixg(self, p, g, dTdQ_x, Qx, w0_dTdQx, l0_dTdQx, alpha_x, 
                                dTdQ_y, Qy, w0_dTdQy, l0_dTdQy, alpha_y): 
        """
        Helper function for solving kx,ky using experimental slopes from laserx, lasery experiments. 
        Fixing g. 

        Parameters 
        ----------
        See solve_anisotropic_raman, gets rid of g parameters  
        g : float 
            fixed value for g 

        """
        kx, ky = p 

        eq1 = self.solve_kx(kx, dTdQ_x, Qx, ky, g, w0_dTdQx, l0_dTdQx, alpha_x)
        eq2 = self.solve_ky(ky, dTdQ_y, Qy, kx, g, w0_dTdQy, l0_dTdQy, alpha_y)
        eqtns = np.concatenate([eq1, eq2])#is this generalizable?
        # print(np.sum(eqtns), kx, ky)
        return eqtns
    
    def get_kx_ky_fixg(self, x0, g, dTdQ_x, Qx, w0_dTdQx, l0_dTdQx, alpha_x, 
                    dTdQ_y, Qy, w0_dTdQy, l0_dTdQy, alpha_y, **ls_kwargs):
        """
        Least squares minimization on solve_anisotropic_raman_fixg; slopes laserx, lasery experiments 
        
        Parameters
        ----------
        See get_kx_ky_g, gets rid of g parameters 
        g : float 
            fixed value for g 

        """
        
        root = least_squares(self.solve_anisotropic_raman_fixg, x0=x0, 
                      args=(g, dTdQ_x, Qx, w0_dTdQx, l0_dTdQx, alpha_x,
                    dTdQ_y, Qy, w0_dTdQy, l0_dTdQy, alpha_y), **ls_kwargs)
        return root

    def solve_anisotropic_raman_varyall(self, p, dTdQ_x, Qx,dTdQ_y, Qy,dTdQ_g, Qg): 
        """
        Helper function for solving for kx, ky, g usin slopes from isotropic, laserx, lasery experiments. allowing spot sizes and thickness to vary 

        Parameters
        ----------
        p : array-like 
            kx, ky, g, h, alpha_x, alpha_y, alpha_g, w0_dTdQx, l0_dTdQx, w0_dTdQg  ; parameters to minimize to 
        dTdQ_x : float 
            dTdQ slope for measurement used to solve for kx (lasery)
        Qx : array-like
            powers used for dTdQ_x 
        dTdQ_y : float 
            dTdQ slope for measurement used to solve for ky (laserx)
        Qy : array-like
            powers used for dTdQ_y
        dTdQ_g : float 
            dTdQ slope for measurement used to solve for kx (isolaser)
        Qg : array-like
            powers used for dTdQ_g

        """

        kx, ky, g, h, alpha_x, alpha_y, alpha_g, w0_dTdQx, l0_dTdQx, w0_dTdQg = p #w0_dTdQy, l0_dTdQy,
        w0_dTdQy = l0_dTdQx # enforcing anisotropic beam to be the same for both measurements 
        l0_dTdQy = w0_dTdQx
        l0_dTdQg = w0_dTdQg # enforcing isotropic beam to be perfectly isotropic
        
        self.h = h 

        eq1 = self.solve_kx(kx, dTdQ_x, Qx, ky, g, w0_dTdQx, l0_dTdQx, alpha_x)
        eq2 = self.solve_ky(ky, dTdQ_y, Qy, kx, g, w0_dTdQy, l0_dTdQy, alpha_y)
        eq3 = self.solve_g(g, dTdQ_g, Qg, kx, ky, w0_dTdQg, l0_dTdQg, alpha_g)
        eqtns = np.concatenate([eq1, eq2, eq3])#is this generalizable?
        # print(np.sum(eqtns), kx, ky, g/1E6)
        return eqtns

    def get_kx_ky_g_varyall(self, x0, dTdQ_x, Qx, dTdQ_y, Qy, dTdQ_g, Qg, **ls_kwargs):
        """
        Least squares minimization on solve_anisotropic_raman_varyall
        
        Parameters
        ----------
        x0 : array-like 
            kx, ky, g, h, alpha_x, alpha_y, alpha_g, w0_dTdQx, l0_dTdQx, w0_dTdQg  ; parameters to minimize to 
        dTdQ_x : float 
            dTdQ slope for measurement used to solve for kx (lasery)
        Qx : array-like
            powers used for dTdQ_x 
        dTdQ_y : float 
            dTdQ slope for measurement used to solve for ky (laserx)
        Qy : array-like
            powers used for dTdQ_y
        dTdQ_g : float 
            dTdQ slope for measurement used to solve for kx (isolaser)
        Qg : array-like
            powers used for dTdQ_g

        """
        
        root = least_squares(self.solve_anisotropic_raman_varyall, x0=x0, 
                      args=(dTdQ_x, Qx, dTdQ_y, Qy, dTdQ_g, Qg, ), **ls_kwargs)
        return root  
    
    def solve_anisotropic_raman_kxky_varyall(self, p, dTdQ_x, Qx,dTdQ_y, Qy,g): 
        """
        Helper function for solving for kx, ky usin slopes from laserx, lasery experiments. fix g, allow spot sizes and thickness to vary  

        Parameters
        ----------
        p : array-like 
            kx, ky, h, alpha_x, alpha_y, w0_dTdQx, l0_dTdQx  ; parameters to minimize to 
        dTdQ_x : float 
            dTdQ slope for measurement used to solve for kx (lasery)
        Qx : array-like
            powers used for dTdQ_x 
        dTdQ_y : float 
            dTdQ slope for measurement used to solve for ky (laserx)
        Qy : array-like
            powers used for dTdQ_y

        """

        kx, ky, h, alpha_x, alpha_y, w0_dTdQx, l0_dTdQx = p #w0_dTdQy, l0_dTdQy,
        w0_dTdQy = l0_dTdQx # enforcing anisotropic beam to be the same for both measurements 
        l0_dTdQy = w0_dTdQx
        
        self.h = h 

        eq1 = self.solve_kx(kx, dTdQ_x, Qx, ky, g, w0_dTdQx, l0_dTdQx, alpha_x)
        eq2 = self.solve_ky(ky, dTdQ_y, Qy, kx, g, w0_dTdQy, l0_dTdQy, alpha_y)
        eqtns = np.concatenate([eq1, eq2])#is this generalizable?
        # print(np.sum(eqtns), kx, ky)
        return eqtns

    def get_kx_ky_varyall(self, x0, dTdQ_x, Qx, dTdQ_y, Qy, g, **ls_kwargs):
        """
        Least squares minimization on solve_anisotropic_raman_kxky_varyall. 

        Parameters
        ----------
        x0 : array-like 
            kx, ky, h, alpha_x, alpha_y, w0_dTdQx, l0_dTdQx  ; parameters to minimize to 
        dTdQ_x : float 
            dTdQ slope for measurement used to solve for kx (lasery)
        Qx : array-like
            powers used for dTdQ_x 
        dTdQ_y : float 
            dTdQ slope for measurement used to solve for ky (laserx)
        Qy : array-like
            powers used for dTdQ_y
        """
        
        root = least_squares(self.solve_anisotropic_raman_kxky_varyall, x0=x0, 
                      args=(dTdQ_x, Qx, dTdQ_y, Qy, g), **ls_kwargs)
        return root  
    
    
    def solve_isotropic_raman(self, p, dTdQ_array, Q): 
        """
        Solve for k, g from isotropic raman measurements sweeping through z/spot size 

        Parameters
        ---------
        p : array-like
            k, g ; parameters to minimize to 
        dTdQ_array : array-like
            array of slopes dTdQ vs. z / spot size 
        Q : array-like
            powers used to get dTdQ slopes 
        """
        k,g=p 
        dTdQ_sim = self.dTdQ(k,k,g,Q)
        resid = np.sqrt((dTdQ_array - dTdQ_sim)**2)/1e6
        print(k,g,np.sum(resid))
        return resid
    
    def solve_isotropic_raman_fixk(self, p, dTdQ_array, Q, kx, ky): 
        """
        Solve for g from isotropic raman measurements sweeping through z/spot size, fix kx,ky

        Parameters
        ---------
        p : array-like
            k, g ; parameters to minimize to 
        dTdQ_array : array-like
            array of slopes dTdQ vs. z / spot size 
        Q : array-like
            powers used to get dTdQ slopes 
        kx : float
            fixed value for kx
        ky : float 
            fixed value for ky 
        """
        g=p 
        dTdQ_sim = self.dTdQ(kx,ky,g,Q)
        resid = np.sqrt((dTdQ_array - dTdQ_sim)**2)
        # print(g,resid)
        return resid

    def solve_isotropic_raman_varyall(self, p, dTdQ_array, Q):
        """
        Solve for k,g from isotropic raman measurements sweeping through z/spot size, allow alpha thickness,w0 to vary 

        Parameters
        ---------
        p : array-like
            k, g ; parameters to minimize to 
        dTdQ_array : array-like
            array of slopes dTdQ vs. z / spot size 
        Q : array-like
            powers used to get dTdQ slopes 
        kx : float
            fixed value for kx
        ky : float 
            fixed value for ky 
        """
        k, g, h, alpha_g, w0_dTdQg = p 
        self.h = h 
        self.alpha = alpha_g 

        self.update_w0(w0_dTdQg)
        self.update_l0(w0_dTdQg)
        
        dTdQ_sim = self.dTdQ(k,k,g,Q)
        resid = np.sqrt((dTdQ_array - dTdQ_sim)**2)
        print(k,g,resid)

        return resid 
    
    def get_k_g_varyall(self, x0, dTdQ_array, Q, **ls_kwargs): 
        """
        least squares minimization for solve_isotropic_raman_varyall

        Parameters
        ---------
        x0 : array-like
            parameter initial guess 
        dTdQ_array : array-like
            array of slopes dTdQ vs. z / spot size 
        Q : array-like
            powers used to get dTdQ slopes 
        ls_kwargs : dict
            keyword arguments for scipy.optimize.least_squares 
        """
        root = least_squares(self.solve_isotropic_raman_varyall, x0=x0, args=(dTdQ_array, Q), **ls_kwargs)
        return root 
    
    def get_k_g(self, x0, dTdQ_array, Q, **ls_kwargs): 
        """
        least squares minimization for solve_isotropic_raman

        Parameters
        ---------
        x0 : array-like
            parameter initial guess 
        dTdQ_array : array-like
            array of slopes dTdQ vs. z / spot size 
        Q : array-like
            powers used to get dTdQ slopes 
        ls_kwargs : dict
            keyword arguments for scipy.optimize.least_squares 
        """
        root = least_squares(self.solve_isotropic_raman, x0=x0, args=(dTdQ_array, Q), **ls_kwargs)
        return root
    
    def get_g_fixk(self, x0, dTdQ_array, Q, kx,ky,**ls_kwargs): 
        """
        Least squares minimization for solve_isotropic_raman_fixk

        Parameters
        ---------
        x0 : array-like
            parameter initial guess 
        dTdQ_array : array-like
            array of slopes dTdQ vs. z / spot size 
        Q : array-like
            powers used to get dTdQ slopes 
        kx : float
            fixed value for kx
        ky : float 
            fixed value for ky 
        ls_kwargs : dict
            keyword arguments for scipy.optimize.least_squares 
        """
        root = least_squares(self.solve_isotropic_raman_fixk, x0=x0, args=(dTdQ_array, Q,kx,ky), **ls_kwargs)
        return root 
    
    ########################################################################################################################################
    # Ignore for now 1/13/25
    
    def resid_full_isotropic(self, p, dT_arrs, Q, Ta): 
        k, g = p #w0_dTdQy, l0_dTdQy,

        dTdQ_sim = self.dTdQ(k,k,g,Q)
        sim = linear(Q, dTdQ_sim, Ta)
        sim = sim - sim[0]
        resid = np.sqrt((dTdQ_sim - dT_arrs)**2)
        resid = resid.ravel()
        # print(np.sum(resid), k, k, g)
        return resid
         
    def minimize_resid_full_isotropic(self, x0, dT_arrs, Q, Ta,**ls_kwargs):
        root = least_squares(self.resid_full_isotropic, x0=x0, args=(dT_arrs, Q, Ta),**ls_kwargs)
        return root
    
    def resid_full_isotropic_varyall(self, p, dT_arrs, Q, Ta): 
        k, g, h, alpha = p #w0_dTdQy, l0_dTdQy,
        self.h = h 
        self.alpha = alpha

        dTdQ_sim = self.dTdQ(k,k,g,Q)
        sim = linear(Q, dTdQ_sim, Ta)
        sim = sim - sim[0]
        resid = np.sqrt((dTdQ_sim - dT_arrs)**2)
        resid = resid.ravel()
        # print(np.sum(resid), k, k, g)
        return resid
         
    def minimize_resid_full_isotropic_varyall(self, x0, dT_arrs, Q, Ta, **ls_kwargs):
        root = least_squares(self.resid_full_isotropic_varyall, x0=x0, args=(dT_arrs, Q, Ta),**ls_kwargs)
        return root
    
    def solve_anisotropic_raman_curve(self, p, dTdQ_array, Q): 
        kx,ky,g=p
        dTdQ_sim = self.dTdQ(kx,ky,g,Q)
        resid = np.sqrt((dTdQ_array - dTdQ_sim)**2)
        # print(resid)
        return resid
    
    def get_kx_ky_g_curve(self, x0, dTdQ_array, Q, **ls_kwargs): 
        root = least_squares(self.solve_anisotropic_raman_curve, x0=x0, args=(dTdQ_array, Q), **ls_kwargs)
        return root 
    
    ####################################################################################################################
    # Ignore for now .. 1/10/25
    def mc_sim_resid_full_varyall(self, x0, dT_x, dT_x_err, Qx, 
                        dT_y, dT_y_err, Qy, 
                        dT_g, dT_g_err, Qg,
                        ntrials, **ls_kwargs):
        self.mcpars_resid_full_varyall = np.zeros([ntrials,len(x0)])
        for i in range(ntrials): 
            # resampling 
            xidx = np.random.choice(np.arange(len(Qx)), len(Qx), replace=False)
            pwr_x_trial, dT_x_trial, dT_x_err_trial = Qx[xidx], dT_x[xidx], dT_x_err[xidx]

            yidx = np.random.choice(np.arange(len(Qy)), len(Qy), replace=False)
            pwr_y_trial, dT_y_trial, dT_y_err_trial = Qy[yidx], dT_y[yidx], dT_y_err[yidx]
                    
            gidx = np.random.choice(np.arange(len(Qg)), len(Qg), replace=False)
            pwr_g_trial, dT_g_trial, dT_g_err_trial = Qg[gidx], dT_g[gidx], dT_g_err[gidx]  

            root = self.minimize_resid_full_varyall(x0, dT_x_trial, dT_x_err_trial,pwr_x_trial, 
                                            dT_y_trial, dT_y_err_trial,pwr_y_trial, 
                                            dT_g_trial, dT_g_err_trial,pwr_g_trial, 0,
                                        **ls_kwargs)
            self.mcpars_resid_full_varyall[i] = root.x

    def mc_sim_resid_full(self, x0, dT_x, dT_x_err, Qx, w0_dTdQx, l0_dTdQx, 
                                dT_y, dT_y_err, Qy, w0_dTdQy, l0_dTdQy,
                                dT_g, dT_g_err, Qg, w0_dTdQg, l0_dTdQg, 
                                ntrials, **ls_kwargs):
        self.mcpars_resid_full = np.zeros([ntrials,len(x0)])
        for i in range(ntrials): 
            # resampling 
            xidx = np.random.choice(np.arange(len(Qx)), len(Qx), replace=False)
            pwr_x_trial, dT_x_trial, dT_x_err_trial = Qx[xidx], dT_x[xidx], dT_x_err[xidx]

            yidx = np.random.choice(np.arange(len(Qy)), len(Qy), replace=False)
            pwr_y_trial, dT_y_trial, dT_y_err_trial = Qy[yidx], dT_y[yidx], dT_y_err[yidx]
                    
            gidx = np.random.choice(np.arange(len(Qg)), len(Qg), replace=False)
            pwr_g_trial, dT_g_trial, dT_g_err_trial = Qg[gidx], dT_g[gidx], dT_g_err[gidx]  

            root = self.minimize_resid_full(x0, dT_x_trial, dT_x_err_trial,pwr_x_trial, w0_dTdQx, l0_dTdQx,
                                            dT_y_trial, dT_y_err_trial,pwr_y_trial, w0_dTdQy, l0_dTdQy,
                                            dT_g_trial, dT_g_err_trial,pwr_g_trial,  w0_dTdQg, l0_dTdQg,0,
                                        **ls_kwargs)
            self.mcpars_resid_full[i] = root.x 

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
