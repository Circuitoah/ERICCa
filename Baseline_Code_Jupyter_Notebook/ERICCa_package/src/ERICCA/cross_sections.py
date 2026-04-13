#Version 4_11_2026

import numpy as np
from numpy.polynomial import legendre
from scipy.interpolate import Rbf

def mesh_creator(xa = 0, xb = 10 ,x_numpoints = 20):

    x_roots, x_weights = legendre.leggauss(x_numpoints)
    x_weights = x_weights* 0.5 * (xb - xa)
    # Map the roots from [-1, 1] to the interval [a, b]
    x_mapped_roots = 0.5 * (x_roots + 1) * (xb - xa) + xa

    return x_weights, x_mapped_roots

class cross_section:

    def __init__(self):
        """
        Class parameters
        rmax : (float), max values for r mesh [fm]
        bmax : (float), max values for b mesh [fm]
        zmax : (float), max and min values for z mesh [fm]
        st_max : (float),  max values for s and t mesh [fm]

        numpoints : (int),  number of points for the s and t mesh
        numpoints_theta : (int), number of points for the theta_s and theta_t mesh
        r_numpoints : (int), number of points for the r mesh
        b_numpoints : (int), number of points for the b mesh
        z_numpoints : (int), number of points for the z mesh
        """
       #max values for our mesh
        self.rmax = 25
        self.bmax = 20
        self.zmax = 5
        self.st_max = 15

        #Number of points for the mesh
        self.numpoints = 30 
        self.numpoints_theta = 30
        self.r_numpoints = 20
        self.b_numpoints = 35
        self.z_numpoints =20
        
        #Defining bounds of integrations
        ra, rb = 0, self.rmax #fm 
        ba, bb = 0, self.bmax #fm   
        za, zb = -self.zmax, self.zmax#fm 
        sa,sb =  0, self.st_max #fm
        s_theta_a, s_theta_b = 0, 2*np.pi 
        ta, tb =0, self.st_max #fm
        t_theta_a , t_theta_b = 0, 2 * np.pi
        
        # Defining legendre mesh
        self.r_weights, self.r_mapped_roots = mesh_creator(xa =ra, xb =rb ,x_numpoints = self.r_numpoints)
        self.b_weights, self.b_mapped_roots = mesh_creator(xa =ba, xb =bb ,x_numpoints = self.b_numpoints)
        self.z_weights, self.z_mapped_roots = mesh_creator(xa =za, xb =zb ,x_numpoints = self.z_numpoints)
        self.s_weights, self.s_mapped_roots = mesh_creator(xa =sa, xb = sb, x_numpoints = self.numpoints)
        self.s_theta_weights, self.s_theta_mapped_roots = mesh_creator(xa =s_theta_a, xb = s_theta_b, x_numpoints = self.numpoints_theta)
        self.t_weights, self.t_mapped_roots = mesh_creator(xa =ta, xb =tb, x_numpoints = self.numpoints)
        self.t_theta_weights, self.t_theta_mapped_roots = mesh_creator(xa =t_theta_a , xb =t_theta_b, x_numpoints = self.numpoints_theta)

#_ Vector operations____________________________________________________________________________________________________________________________

    def add_sub_vec_mag( self, a, theta_a, b, theta_b, c, theta_c):
        r"""
        Takes the magnitude of the operation \vec a + \vec b - \ vec c
        """
        arg_x =  a * np.cos(theta_a) + b * np.cos(theta_b) - c * np.cos(theta_c)
        arg_y =  a * np.sin(theta_a) + b * np.sin(theta_b) - c * np.sin(theta_c)

        return np.sqrt(arg_x**2 + arg_y**2)

# Functions that require integrals____________________________________________________________________________________________________

    def Chi_mol_1(self, b, rho_t, rho_p, Gamma):

        """
        Calculates the half of the eikonal phase in the MOL model.
    
        Input parameters
        b     : (float), Impact parameter [fm]
        rho_t : (list/array size = t_numpoints), target density [fm]
        rho_p : (list/array size = t_numpoints), projectile density [fm]
        Gamma : (function(b)), profile function [1/fm^2]
        """
        
        #defining integrand

        def chi_t(b,s, theta_s, theta_t):

            t_integrand = lambda t :  t * Gamma(self.add_sub_vec_mag(b, 0, s, theta_s, t, theta_t)) 
            func_values =t_integrand(self.t_mapped_roots[:, np.newaxis, np.newaxis, np.newaxis])*rho_t[:, np.newaxis, np.newaxis, np.newaxis]
            # Perform Gaussian Legendre integration
            t_result = np.sum(self.t_weights[:, np.newaxis, np.newaxis, np.newaxis] * func_values, axis= 0)

            return t_result
    
        def chi_theta_t(b,s, theta_s):

            t_theta_integrand = lambda theta_t : chi_t(b,s, theta_s, theta_t)
            func_values = t_theta_integrand(self.t_theta_mapped_roots[:, np.newaxis, np.newaxis])
            # Perform Gaussian Legendre integration
            t_theta_result = np.sum(self.t_theta_weights[:, np.newaxis, np.newaxis] * func_values, axis= 0) 

            return t_theta_result

        def chi_s(b, theta_s):

            s_integrand = lambda s: s * (1 - np.exp(-chi_theta_t(b,s, theta_s)) )
            func_values = s_integrand(self.s_mapped_roots[:, np.newaxis])* rho_p[:, np.newaxis]
            # Perform Gaussian Legendre integration
            s_result = np.sum(self.s_weights[:, np.newaxis] * func_values, axis = 0) 

            return s_result

        integrand = lambda theta_s: .5j * chi_s(b, theta_s)
        func_values = integrand(self.s_theta_mapped_roots)
        # Perform Gaussian Legendre integration
        result = np.sum(self.s_theta_weights * func_values, axis = 0)  
    
        return result

    def Chi_mol(self,b, rho_t, rho_p, Gamma):
        """
        Calculates the eikonal phase in the MOL model.
    
        Input parameters
        b     : (float), Impact parameter [fm]
        rho_t : (list/array size = t_numpoints), target density [fm]
        rho_p : (list/array size = t_numpoints), projectile density [fm]
        Gamma : (function(b)), profile function [1/fm^2]
        """
        return self.Chi_mol_1(b, rho_t, rho_p, Gamma) + self.Chi_mol_1(b, rho_p, rho_t, Gamma)

#_____________________________________________________________________________________________________________________________________
    def chi(self,b, rho_t, rho_p, Gamma):
            """
            Calculates the eikonal phase in the OLA model.
    
            Input parameters
            b     : (float), Impact parameter [fm]
            rho_t : (list/array size = t_numpoints), target density [fm]
            rho_p : (list/array size = t_numpoints), projectile density [fm]
            Gamma : (function(b)), profile function [1/fm^2]

            uses s t and theta meshs
            """
            #defining integrand
            def chi_t(b,s, theta_s, theta_t):

                t_integrand = lambda t : 1j* s * t * Gamma(self.add_sub_vec_mag(b, 0, s, theta_s, t, theta_t)) 
                func_values = t_integrand(self.t_mapped_roots[:, np.newaxis, np.newaxis, np.newaxis]) * rho_t[:, np.newaxis, np.newaxis, np.newaxis]
                # Perform Gaussian Legendre integration
                t_result = np.sum(self.t_weights[:, np.newaxis, np.newaxis, np.newaxis] * func_values, axis = 0)

                return t_result
    
            def chi_theta_t(b,s, theta_s):

                t_theta_integrand = lambda theta_t : chi_t(b,s, theta_s, theta_t)
                func_values =t_theta_integrand(self.t_theta_mapped_roots[:, np.newaxis, np.newaxis])        
                # Perform Gaussian Legendre integration
                t_theta_result = np.sum(self.t_theta_weights[:, np.newaxis, np.newaxis] * func_values, axis = 0) 

                return t_theta_result

            def chi_s(b, theta_s):

                s_integrand = lambda s: chi_theta_t(b,s, theta_s)
                func_values =s_integrand(self.s_mapped_roots[:, np.newaxis])* rho_p [:, np.newaxis]        
               # Perform Gaussian Legendre integration
                s_result = np.sum(self.s_weights[:, np.newaxis] * func_values, axis = 0) 

                return s_result

            integrand = lambda theta_s: chi_s(b, theta_s)    
            func_values =integrand(self.s_theta_mapped_roots)
            # Perform Gaussian Legendre integration
            result = np.sum(self.s_theta_weights * func_values, axis = 0)  

            return result

    def chi_no_dens_term(self, b , rho, Gamma):
        """
        Calculates the eikonal phase in the OLA model for proton-nucleus scattering.
    
        Input parameters
        b     : (float), Impact parameter [fm]
        rho_t : (list/array size = t_numpoints), target density [fm]
        rho_p : (list/array size = t_numpoints), projectile density [fm]
        Gamma : (function(b)), profile function [1/fm^2]

        uses s, t and theta meshs
        """

        def chi_s(b, theta_s):

            s_integrand = lambda  s, : 1j * s * (Gamma(self.add_sub_vec_mag(b,0,s,theta_s,0,0)))      
            func_values =s_integrand(self.s_mapped_roots[:, np.newaxis])* rho[:, np.newaxis]
            # Perform Gaussian Legendre integration
            s_result = np.sum(self.s_weights[:, np.newaxis] * func_values, axis = 0) 

            return s_result

        integrand = lambda theta_s: chi_s(b, theta_s)
        func_values =integrand(self.s_theta_mapped_roots)
        # Perform Gaussian Legendre integration
        result = np.sum(self.s_theta_weights * func_values, axis = 0)  

        return result


    def chi_no_dens(self, b , rho_p, rho_n, Gamma_pp, Gamma_pn):
        return self.chi_no_dens_term(b , rho_p, Gamma_pp) + self.chi_no_dens_term( b , rho_n, Gamma_pn)


    def sigma_R_matter(self, rho_t,  rho_p , Gamma, Model = "OLA"):

        """
        Calculates the reaction cross section for matter densities as inputs
    
        Input parameters
        rho_t : (list/array size = t_numpoints), target density [fm]
        rho_p : (list/array size = t_numpoints), projectile density [fm]
        Gamma : (function(b)), profile function [1/fm^2]
        Model : (string),  indicates what model should be used in the calculation
                        -> "MOL","OLA","OLA p-n"

        uses b mesh
        """
        
        if (Model == "MOL"):
            sigma_R_int= lambda b: 2 * np.pi * b * (1 - np.exp(- 2 * self.Chi_mol(b, rho_t, rho_p, Gamma ).imag ) )
                
        else:
            #if both target and projectiles are composite particles 
            sigma_R_int= lambda b: 2 * np.pi * b * (1 - np.exp(- 2 * self.chi(b, rho_t, rho_p, Gamma ).imag ) )

        func_values = np.array( [sigma_R_int(i) for i in self.b_mapped_roots.tolist()])
        # Perform Gaussian Legendre integration
        result = np.sum(self.b_weights * func_values) 

        return result * 10

    def chi_mol_micro(self, b, rho_t_p, rho_t_n, rho_p_p, rho_p_n, Gamma_pp, Gamma_pn, Gamma_nn ):
        """
        Calculates the eikonal phase for proton and neutron densities as inputs in the MOL model
    
        Input parameters
        b : (float), impact parameter [fm]
        rho_t_p : (list/array size = t_numpoints), target density [fm]
        rho_t_n : (list/array size = t_numpoints), target density [fm]
        rho_p_p : (list/array size = t_numpoints), projectile density [fm]
        rho_p_n : (list/array size = t_numpoints), projectile density [fm]
        Gamma_pp : (function(b)), proton-proton profile function [1/fm^2]
        Gamma_pn : (function(b)), proton-neutron profile function [1/fm^2]
        Gamma_nn : (function(b)), neutron-neutron profile function [1/fm^2]
        """
    
        chi_pp = self.Chi_mol(b, rho_t_p, rho_p_p, Gamma_pp )
        chi_pn = self.Chi_mol(b, rho_t_p, rho_p_n, Gamma_pn ) + self.Chi_mol(b, rho_t_n, rho_p_p, Gamma_pn )
        chi_nn = self.Chi_mol(b, rho_t_n, rho_p_n, Gamma_nn )
        return (chi_pp + chi_pn + chi_nn)

    def chi_ola_micro(self, b, rho_t_p, rho_t_n, rho_p_p, rho_p_n, Gamma_pp, Gamma_pn, Gamma_nn ):
        """
        Calculates the eikonal phase for proton and neutron densities as inputs in the OLA model
    
        Input parameters
        b : (float), impact parameter [fm]
        rho_t_p : (list/array size = t_numpoints), target density [fm]
        rho_t_n : (list/array size = t_numpoints), target density [fm]
        rho_p_p : (list/array size = t_numpoints), projectile density [fm]
        rho_p_n : (list/array size = t_numpoints), projectile density [fm]
        Gamma_pp : (function(b)), proton-proton profile function [1/fm^2]
        Gamma_pn : (function(b)), proton-neutron profile function [1/fm^2]
        Gamma_nn : (function(b)), neutron-neutron profile function [1/fm^2]
        """
    
        chi_pp = self.chi(b, rho_t_p, rho_p_p, Gamma_pp )
        chi_pn = self.chi(b, rho_t_p, rho_p_n,  Gamma_pn) + self.chi(b, rho_t_n, rho_p_p, Gamma_pn) 
        chi_nn = self.chi(b, rho_t_n, rho_p_n, Gamma_nn )
        return (chi_pp + chi_pn + chi_nn)


    def sigma_R_pn(self, rho_t_p, rho_t_n, rho_p_p = 0 , rho_p_n = 0, Gamma_pp = lambda b : np.exp(-b), 
                   Gamma_pn = lambda b : np.exp(-b), Gamma_nn = lambda b : np.exp(-b) , Model = "OLA"):

        """
        Calculates the reaction cross section using proton and neutron densities as inputs 
    
        Input parameters
        rho_t_p : (list/array size = t_numpoints), target density [fm]
        rho_t_n : (list/array size = t_numpoints), target density [fm]
        rho_p_p : (list/array size = t_numpoints), projectile density [fm]
        rho_p_n : (list/array size = t_numpoints), projectile density [fm]
        Gamma_pp : (function(b)), proton-proton profile function [1/fm^2]
        Gamma_pn : (function(b)), proton-neutron profile function [1/fm^2]
        Gamma_nn : (function(b)), neutron-neutron profile function [1/fm^2]
        Model : (string), indicates choice of model -> "MOL", "OLA"

        uses b mesh
        """
        
        if (Model == "MOL"):
            sigma_R_int= lambda b: 2 * np.pi * b * (1 - np.exp(- 2 * self.chi_mol_micro(b, rho_t_p, rho_t_n, rho_p_p, rho_p_n, Gamma_pp, Gamma_pn, Gamma_nn).imag ) )
         
        elif (Model == "OLA p-n"):
        #if only the target/projectile are composite particles
            sigma_R_int =lambda b: 2 * np.pi * b * (1 - np.exp(- 2 * self.chi_no_dens( b , rho_t_p, rho_t_n, Gamma_pp, Gamma_pn).imag ) )
            
        else:
            #if both target and projectiles are composite particles 
            sigma_R_int= lambda b: 2 * np.pi * b * (1 - np.exp(- 2 * self.chi_ola_micro(b, rho_t_p, rho_t_n, rho_p_p, rho_p_n,Gamma_pp, Gamma_pn, Gamma_nn).imag ) )
            
        func_values = np.array( [sigma_R_int(i) for i in self.b_mapped_roots.tolist()])
        # Perform Gaussian Legendre integration
        result = np.sum(self.b_weights * func_values) 

        return result * 10
    

    def dens_b_interpolator(self, array_r, array_rho):
        """
        Takes a spherecial densties and maps/interpolates spherical mesh onto mesh used 
        in the phase shift calculation. Can be used for matter and proton and neutron 
        densties.
    
        Input parameters
            array_r   : list or np.array, radius r mesh in spherical coordinates [fm]
            array_rho : list or np.array, density mesh [?] 

            . array_r and array_rho needs to be the same size

            . Requires z mesh
        """
    
        def rho_funct(r):
            if r <= (array_r[-1]) :  
                den_rho_f = Rbf(array_r, array_rho)
                return den_rho_f(r)
    
            else:
                return 0
    
        def rho_z_funct(b):
            integrand = lambda z: rho_funct(np.sqrt(b**2 + z**2))
            funct_values = np.array([integrand(i) for i in  self.z_mapped_roots.tolist()])
            results_b = np.sum(self.z_weights * funct_values)
            return results_b 

        results_z = np.array( [rho_z_funct(i) for i in self.t_mapped_roots.tolist()])
        return results_z

