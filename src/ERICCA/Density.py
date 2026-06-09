
import numpy as np
from scipy.optimize import minimize

class Density:
    """ 
    Input paramters
        C_m_p  : (float), half-density radius
        a_m_p  : (float), diffuseness [fm]
        rho_0_p : (float), the saturation density [fm^-3]
    """
    def __init__(self):
     self.C_m_p = 0 
     self.a_m_p = 0 
     self.rho_0_p = 0

    def rho_m(self, r):
        """
        Calculates the density of a two point fermi funciton at r radius
        Input paramters
            r       : (float), radius in spherical coordinates [fm]
        """
        return self.rho_0_p /(1 + np.exp((r - self.C_m_p)/self.a_m_p))
    
    def rms(self, r_mesh ,rho_mesh , A):
        """Calculates the root mean squared matter radius of an isotope. r_mesh and rho_m_mesh should be the same size.

            Input parameters
            r_mesh : (list/array)  , radial mesh [fm]
            rho_m_mesh: (list/array), matter density of a nucleius as a function of radius [1/fm^3]
            A: (float), Mass number [ ]
        """
        return np.sqrt( ( 4 * np.pi * 1/A)  * np.sum(np.array(rho_mesh) * (np.array(r_mesh)**4)) * (r_mesh[1] - r_mesh[0]))
    
    def rho_m_2pt_fermi(self, A, rms_measured, ra =0, rb=30 , r_points = 1000):
        """
        Given an mass number and An rms matter radius of a nucleius, will fit parameters C_m_p, a_m_p, and rho_0_p for
        function rho_m(r). The minimized parameters are found by fitting to the rms matter radius, the mass number A, and
        the saturation density rho_m(0).

        Input Parameters
        A: (float), Mass number [ ]
        rms_measured: (float), measured matter radius of a nuclei [fm]
        ra: (float), lower bound for integration of the r mesh [fm]
        rb: (float), upper bound of intergration for the r mesh [fm]
        r_points: (float), number of points in the mesh [ ]
        """
        r_meshy =  np.linspace(ra, rb, r_points)

        def chi_2_rms_min(params):
            """Calculates the chi squared for the rms matter radius"""
            self.C_m_p, self.a_m_p, self.rho_0_p = params
            return (self.rms(r_meshy , self.rho_m(r_meshy) , A) - rms_measured)**2

        def A_return(rho):
            """
            Given a function rho(r) calculates the A of the nucleus. This was ment to be a benchmark
            """
            integrand = lambda r :  4 * np.pi* (r**2)  * rho(r)
            return np.sum(integrand(r_meshy)) * (r_meshy[1]-r_meshy[0])

        def A_min(params):
            self.C_m_p, self.a_m_p, self.rho_0_p = params
            return abs(A_return(self.rho_m) - A)

        def sat_min(params):
            self.C_m_p, self.a_m_p, self.rho_0_p = params
            return abs(self.rho_m(0) - 0.176)

        constraints=  [{'type': 'eq', 'fun': A_min},
                     {'type': 'eq', 'fun': sat_min}]

        dens_min = minimize(chi_2_rms_min, [ 4.1, .5, .176], constraints=constraints)

        self.C_m_p, self.a_m_p, self.rho_0_p =  dens_min.x
        return


