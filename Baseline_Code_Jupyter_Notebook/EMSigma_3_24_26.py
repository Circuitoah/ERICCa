"""
the difference between this and the other version of Functions.py Gamma is an input to the X and sigma 

Edit: Febuary 2nd, 2026.  Commenting the code in prep for publications if the code stops giving proper results please go to \Final_Results_with_Figures

Edit:Febuary 3rd, 2026. Found a bug in the code, bmax was set to rmax. This means our calculatations were off by ~5 mb. BUG HAS BEEN FIXED. Please use this version of the code.

Edit: Febuary 24th, 2026. Include a discription of the purpose and "function" of each function in the code.

Edit: March 4th, 2026.  -Removed all of the commented out code spaghetti code.
                        -Created function mesh_creator to simplify the defining the mesh for integration
                        -Successfully integrate mesh_creator into creating meshs for the code
                        -Removed global variables A_p, C_m_p, a_m_p, rho_0_p, A_t = 0 and C_m_t = 0 , a_m_t = 0, rho_0_t = 0 and associated local variables into rho(r)

Edit: March 5th, 2026.  - Removed Gammap(b) and Gamman(b) 
                        - Removed global variables sigma_pp, alphapp,  betapp, sigma_pn, alphapn, betapn                      

Edit: March 13th, 2026  - Removed Gamma(b) and profile_funct_param as well as parameters alpha, beta, sigma_n
                        - Created the class Profile_Function
                        - renamed sigma_R_micro to sigma_R_pn
                        - renamed sigma_R to sigma_R_matter

Edit:March 20th, 2026.  - Removed Funcitons A_return, A_min, sat_min, rho_param_finder, C(E)
                        - Removed variables A, and sigma R

Edit: March 23rd, 2026. - Found a bug in chi_no_dens. The input file was  chi_no_dens(b , rho, Gammap) instead of  chi_no_dens(b , rho, Gamma). This apear during 5_14_25 update when Gamma was added as an input. This therefore post-dates the benchmarked that  chi_no_dens was used in. 

Edit: March 24th, 2026. - Turn everything into one class, to remove global variables
                        - removed rm_rms
                        - removed all of the roots variables from the initialization
                        - commented out all of the global variables used for meshes
                        - removed whitespaces to make code more readable
"""
import numpy as np
from numpy.polynomial import legendre
from scipy.interpolate import Rbf, CubicSpline, interp1d
from scipy.optimize import curve_fit, minimize
import scipy as sp


class Eikonal_Model:

    def __init__(self):
       #max values for our mesh
        self.rmax = 25
        self.bmax = 20
        self.zmax = 5
        self.st_max = 15

        #Number of points for the mesh
        self.numpoints = 30 # What numpoints does this go to?
        self.numpoints_theta = 30
        self.r_numpoints = 20
        self.b_numpoints = 35
        self.z_numpoints =20

        ra, rb = 0, self.rmax #fm 
        ba, bb = 0, self.bmax #fm   
        za, zb = -self.zmax, self.zmax#fm 
        sa,sb =  0, self.st_max #fm
        s_theta_a, s_theta_b = 0, 2*np.pi 
        ta, tb =0, self.st_max #fm
        t_theta_a , t_theta_b = 0, 2 * np.pi

        #Defining bounds of integrations
        def mesh_creator(xa = 0, xb = 10 ,x_numpoints = 20):

            x_roots, x_weights = legendre.leggauss(x_numpoints)
            x_weights = x_weights* 0.5 * (xb - xa)
            # Map the roots from [-1, 1] to the interval [a, b]
            x_mapped_roots = 0.5 * (x_roots + 1) * (xb - xa) + xa

            return x_weights, x_mapped_roots

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

#__functions_____________________________________________________________________________________________________________________________


    def rho_m( self, r, C_m_p = 0, a_m_p = 0, rho_0_p = 0):
        """
        Calculates the density of a two point fermi funciton at r radius
    
        Input paramters
            r       : (float), radius in spherical coordinates [fm]
            #C_m_p  : (float), half-density radius
            #a_m_p  : (float), diffuseness [fm]
            rho_0_p : (float), the saturation density [fm^-3]"""
        return rho_0_p /(1 + np.exp((r - C_m_p)/a_m_p))


    class Profile_Function:
        """
        Interpolates profile function paramters for a certain reaction energy and plugs parameters into Gamma(b). Results are only relayable between  E= 40 - 1000 MeV.
         
        Input paramters: 
        interaction_type: (string), Type of reaction [ "general", "np", "matter"] 
        E      : (float),  Energy of reaction [MeV]

        Built in general form of the profile function -> Gamma(b)
    
        Class parameters
        b: float, but could also be an array
        alpha  : (float), the NN scattering amplitude [ ]
        beta   : (float), the finite range parameter [fm^2]
        sigma_n:(float), the nucleon nucleon cross section [fm^2]
        """ 

        def __init__(self, Model_type = "general", E = 1):
            self.alpha = 1
            self.beta = 1
            self.sigma_n = 1
        
            self.Gamma = lambda b: (1 - 1j * self.alpha) / (4 * np.pi * self.beta) * self.sigma_n * np.exp(-b**2 / (2 * self.beta))
        
            if (Model_type == "np"):
    
                profile_funct_table = np.genfromtxt("new_profile_funct_params.txt", unpack = True, skip_header= 2)
    

                sigma_pp_fun = CubicSpline(profile_funct_table[0],profile_funct_table[1])
                alphapp_fun = CubicSpline(profile_funct_table[0],profile_funct_table[2])
                betapp_fun  = CubicSpline(profile_funct_table[0],profile_funct_table[3])

                sigma_pn_fun = CubicSpline(profile_funct_table[0],profile_funct_table[4])
                alphapn_fun = CubicSpline(profile_funct_table[0],profile_funct_table[5])
                betapn_fun  = CubicSpline(profile_funct_table[0],profile_funct_table[6])

                sigma_pp = sigma_pp_fun(E)
                alphapp = alphapp_fun(E)
                betapp  = betapp_fun(E)

                sigma_pn = sigma_pn_fun(E)
                alphapn = alphapn_fun(E)
                betapn  = betapn_fun(E) 
        
                self.Gamma = lambda b :(1 -  1j * alphapn)/( 4 * np.pi * betapn)*sigma_pn * np.exp( - b**2/(2 *betapn) ) + (1 -  1j * alphapp)/( 4 * np.pi * betapp) * sigma_pp * np.exp( - b**2/(2 *betapp) )

            if (Model_type == "matter"):
    
                profile_funct_table = np.genfromtxt("profile_funct_param_matter.txt", unpack = True, skip_header= 2)

                sigma_NN_fun = CubicSpline(profile_funct_table[0],profile_funct_table[1])
                alpha_fun = CubicSpline(profile_funct_table[0],profile_funct_table[2])
                beta_fun  = CubicSpline(profile_funct_table[0],profile_funct_table[3])

                self.sigma_n = sigma_NN_fun(E)
                self.alpha = alpha_fun(E)
                self.beta  = beta_fun(E)
    

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


#_____________________________________________________________________________________________________________________________________

    def rhoz_p(self, b):
        """
        Calculates the density of a two point fermi funciton as a function of impact parameter b at point b
    
        Input parameters
        b: (float), impact parameter [fm]
        """
        #global za, zb, z_numpoints, z_roots, z_weights, z_mapped_roots

        integrand = lambda z : self.rho_m( np.sqrt(z**2 + b**2 ))
    
        func_values = integrand(self.z_mapped_roots)
    
         # Perform Gaussian Legendre integration
        result = np.sum(self.z_weights * func_values) 

        return result


# Functions that require integrals____________________________________________________________________________________________________

    def Chi_mol_1(self, b, rho_t, rho_p, Gamma):

        """
        Calculates the (half?) eikonal phase in the MOL model.
    
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

    def chi_no_dens(self, b , rho, Gamma):
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

    def sigma_R_matter(self, rho_t,  rho_p  = 0, Gamma = lambda b: np.exp(-b), Model = "OLA"):

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
 
        if (Model == "OLA p-n"):
        
        #if only the target/projectile are composite particles
            sigma_R_nd_int =lambda b: 2 * np.pi * b * (1 - np.exp(- 2 * self.chi_no_dens(b, rho_t, Gamma ).imag ) )
            func_values = np.array( [sigma_R_nd_int(i) for i in self.b_mapped_roots.tolist()])
        
        elif (Model == "MOL"):
            sigma_R_int= lambda b: 2 * np.pi * b * (1 - np.exp(- 2 * self.Chi_mol(b, rho_t, rho_p, Gamma ).imag ) )
            func_values = np.array( [sigma_R_int(i) for i in self.b_mapped_roots.tolist()])
                
        else:
            #if both target and projectiles are composite particles 
            sigma_R_int= lambda b: 2 * np.pi * b * (1 - np.exp(- 2 * self.chi(b, rho_t, rho_p, Gamma ).imag ) )

            func_values = np.array( [sigma_R_int(i) for i in self.b_mapped_roots.tolist()])

        # Perform Gaussian Legendre integration
        result = np.sum(self.b_weights * func_values) 

        return result * 10

    def chi_mol_micro(self, b, rho_t_p, rho_t_n, rho_p_n,rho_p_p, Gamma_pp, Gamma_pn, Gamma_nn ):
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


    def sigma_R_pn(self, rho_t_p, rho_t_n, rho_p_p, rho_p_n, Gamma_pp, Gamma_pn, Gamma_nn , Model = "OLA"):

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
            func_values = np.array( [sigma_R_int(i) for i in self.b_mapped_roots.tolist()])
    
        else:
            #if both target and projectiles are composite particles 
            sigma_R_int= lambda b: 2 * np.pi * b * (1 - np.exp(- 2 * self.chi_ola_micro(b, rho_t_p, rho_t_n, rho_p_p, rho_p_n,Gamma_pp, Gamma_pn, Gamma_nn).imag ) )
            func_values = np.array( [sigma_R_int(i) for i in self.b_mapped_roots.tolist()])
        
        # Perform Gaussian Legendre integration
        result = np.sum(self.b_weights * func_values) 

        return result * 10
