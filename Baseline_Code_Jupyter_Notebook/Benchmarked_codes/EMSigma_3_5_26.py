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
                        
"""

import numpy as np
from numpy.polynomial import legendre
from scipy.interpolate import Rbf, CubicSpline, interp1d
from scipy.optimize import curve_fit, minimize
import scipy as sp

#_Constants and meshs

#
A = 12

#constant for C
sigma_R_measured = 218

#Constants for gamma

alpha =   1.808 
beta  = .268 
sigma_n = 3.16

def mesh_creator(xa = 0, xb = 10 ,x_numpoints = 20):

    x_roots, x_weights = legendre.leggauss(x_numpoints)
    x_weights = x_weights* 0.5 * (xb - xa)
    # Map the roots from [-1, 1] to the interval [a, b]
    x_mapped_roots = 0.5 * (x_roots + 1) * (xb - xa) + xa

    return x_roots, x_weights, x_mapped_roots

#max values for our mesh
rmax = 25
bmax = 20
zmax = 5
st_max = 15

#Number of points for the mesh
numpoints = 30 # What numpoints does this go to?
numpoints_theta = 30
r_numpoints = 20
b_numpoints = 35
z_numpoints =20

#Defining bounds of integrations
ra, rb = 0, rmax #fm 
ba, bb = 0, bmax #fm   
za, zb = -zmax, zmax#fm 
sa,sb =  0, st_max #fm
s_theta_a, s_theta_b = 0, 2*np.pi 
ta, tb =0, st_max #fm
t_theta_a , t_theta_b = 0, 2 * np.pi

# Defining legendre mesh
r_roots, r_weights, r_mapped_roots = mesh_creator(xa =ra, xb =rmax ,x_numpoints =r_numpoints)
b_roots, b_weights, b_mapped_roots = mesh_creator(xa =ba, xb =bb ,x_numpoints =b_numpoints)
z_roots, z_weights, z_mapped_roots = mesh_creator(xa =za, xb =zb ,x_numpoints =z_numpoints)
s_roots, s_weights, s_mapped_roots = mesh_creator(xa =sa, xb = sb, x_numpoints = numpoints)
s_theta_roots, s_theta_weights, s_theta_mapped_roots = mesh_creator(xa =s_theta_a, xb = s_theta_b, x_numpoints = numpoints_theta)
t_roots, t_weights, t_mapped_roots = mesh_creator(xa =ta, xb =tb, x_numpoints = numpoints)
t_theta_roots, t_theta_weights, t_theta_mapped_roots = mesh_creator(xa =t_theta_a , xb =t_theta_b, x_numpoints = numpoints_theta)

#_ Vector operations____________________________________________________________________________________________________________________________

def add_sub_vec_mag(a, theta_a, b, theta_b, c, theta_c):
    r"""
    Takes the magnitude of the operation \vec a + \vec b - \ vec c
    """
    arg_x =  a * np.cos(theta_a) + b * np.cos(theta_b) - c * np.cos(theta_c)
    arg_y =  a * np.sin(theta_a) + b * np.sin(theta_b) - c * np.sin(theta_c)

    return np.sqrt(arg_x**2 + arg_y**2)

#__functions_____________________________________________________________________________________________________________________________

def C( E):
    """

    
    """
    global  sigma_R_measured
    #Defining our variables needed for the function 

    #Caluation of neutron grazing distance
    B_c = np. sqrt(sigma_R_measured / np.pi )
    output = 1.0 + (B_c/E)
    return output

def rho_m(r, C_m_p = 0, a_m_p = 0, rho_0_p = 0):
    """
    Calculates the density of a two point fermi funciton at r radius
    
    Input paramters
        r       : (float), radius in spherical coordinates [fm]
        #C_m_p  : (float), half-density radius
        #a_m_p  : (float), diffuseness [fm]
        rho_0_p : (float), the saturation density [fm^-3]
    """
    return rho_0_p /(1 + np.exp((r - C_m_p)/a_m_p))


def Gamma(b):
    """
    
    Built in general form of the profile function.
    
    Input parameters
        b: float, but could also be an array
        alpha : (float), the NN scattering amplitude [ ]
        beta  : (float), the finite range parameter [fm^2]
        sigma_n:(float), the nucleon nucleon cross section [fm^2]
    
    """
    global alpha, beta, sigma_n
    
    arg1 = (1 -  1j * alpha)/( 4 * np.pi * beta)
    arg2 = sigma_n * np.exp( - b**2/(2 *beta) ) 
    return arg1 * arg2 
    
def profile_funct_param(E, interaction_type = "matter"):
    """ 
    Function interpolates profile function paramters for a certain reaction energy and plugs parameters into Gamma(b)
    
    Input paramters 
        E               : (float),  Energy of reaction [MeV]
        interaction_type: (string), Type of reaction ["np", "matter", "matter_input"] 

   Results are only relayable between  E= 40 - 1000 MeV
    """
    
    global Gamma    
    
    if (interaction_type == "np"):# and ( E < 1000 and E > 40) :
    
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
        
        Gamma = lambda b :(1 -  1j * alphapn)/( 4 * np.pi * betapn)*sigma_pn * np.exp( - b**2/(2 *betapn) ) + (1 -  1j * alphapp)/( 4 * np.pi * betapp) * sigma_pp * np.exp( - b**2/(2 *betapp) )
        return 

    elif (interaction_type == "matter"):# and ( E < 1000 and E > 40) :
    
        profile_funct_table = np.genfromtxt("profile_funct_param_matter.txt", unpack = True, skip_header= 2)
    

        sigma_NN_fun = CubicSpline(profile_funct_table[0],profile_funct_table[1])
        alpha_fun = CubicSpline(profile_funct_table[0],profile_funct_table[2])
        beta_fun  = CubicSpline(profile_funct_table[0],profile_funct_table[3])

        sigma_n = sigma_NN_fun(E)
        alpha = alpha_fun(E)
        beta  = beta_fun(E)
    
        mGamma = lambda b : ((1 -  1j * alpha)/( 4 * np.pi * beta) ) * sigma_n * np.exp( - b**2/(2 *beta) ) 
        
        Gamma = lambda b: mGamma(b)
        return 
    
    
    elif interaction_type == "matter_input":
        print("please input the following the values")
        take_alpha = input (" funct.alpha =")
        take_beta = input ("funct.beta =")
        take_sigma_n = input ("funct.sigma_n =")
    
  
        alpha = take_alpha
        beta = take_beta
        sigma_n = take_sigma_n
        
        def mGamma(b):
            # calculating the profile function 
   
            global alpha, beta, sigma_n

            arg1 = (1 -  1j * alpha)/( 4 * np.pi * beta)
            arg2 = sigma_n * np.exp( - b**2/(2 *beta) ) 

            return arg1 * arg2 
        Gamma = mGamma
        return 
        
    else :
        print("Something didn't work :-( , check to make sure your inputs are valid")
        return

    
    
def dens_b_interpolator(array_r, array_rho):
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
        funct_values = np.array([integrand(i) for i in  z_mapped_roots.tolist()])
        results_b = np.sum(z_weights * funct_values)
        return results_b 
    
    results_z = np.array( [rho_z_funct(i) for i in t_mapped_roots.tolist()])
    return results_z


#_____________________________________________________________________________________________________________________________________

def rhoz_p(b):
    """
    Calculates the density of a two point fermi funciton as a function of impact parameter b at point b
    
    Input parameters
        b: (float), impact parameter [fm]
    """
    global A_p

    global za, zb, z_numpoints, z_roots, z_weights, z_mapped_roots

    integrand = lambda z : rho_m( np.sqrt(z**2 + b**2 ))
    
    func_values = integrand(z_mapped_roots)
    
    # Perform Gaussian Legendre integration
    result = np.sum(z_weights * func_values) 

    return result


# Functions that require integrals____________________________________________________________________________________________________

def Chi_mol_1(b, rho_t, rho_p, Gamma):

    """
    Calculates the (half?) eikonal phase in the MOL model.
    
    Input parameters
        b     : (float), Impact parameter [fm]
        rho_t : (list/array size = t_numpoints), target density [fm]
        rho_p : (list/array size = t_numpoints), projectile density [fm]
        Gamma : (function(b)), profile function [1/fm^2]
    """

    # import s mesh
    global sa, sb, s_weights, s_mapped_roots 
    global ta, tb, t_weights, t_mapped_roots 

    global t_theta_a, t_theta_b, t_theta_weights, t_theta_mapped_roots

    #defining integrand

    def chi_t(b,s, theta_s, theta_t):

        t_integrand = lambda t :  t * Gamma(add_sub_vec_mag(b, 0, s, theta_s, t, theta_t)) 
       
        func_values =t_integrand(t_mapped_roots[:, np.newaxis, np.newaxis, np.newaxis])*rho_t[:, np.newaxis, np.newaxis, np.newaxis]

        # Perform Gaussian Legendre integration
        t_result = np.sum(t_weights[:, np.newaxis, np.newaxis, np.newaxis] * func_values, axis= 0)

        return t_result
    
    def chi_theta_t(b,s, theta_s):

        t_theta_integrand = lambda theta_t : chi_t(b,s, theta_s, theta_t)

        func_values = t_theta_integrand(t_theta_mapped_roots[:, np.newaxis, np.newaxis])
        
        # Perform Gaussian Legendre integration
        t_theta_result = np.sum(t_theta_weights[:, np.newaxis, np.newaxis] * func_values, axis= 0) 

        return t_theta_result

    def chi_s(b, theta_s):

        s_integrand = lambda s: s * (1 - np.exp(-chi_theta_t(b,s, theta_s)) )
        
        func_values = s_integrand(s_mapped_roots[:, np.newaxis])* rho_p[:, np.newaxis]
        
        # Perform Gaussian Legendre integration
        s_result = np.sum(s_weights[:, np.newaxis] * func_values, axis = 0) 

        return s_result

    integrand = lambda theta_s: .5j * chi_s(b, theta_s)

    func_values = integrand(s_theta_mapped_roots)
                                  
        # Perform Gaussian Legendre integration
    result = np.sum(s_theta_weights * func_values, axis = 0)  
    
    
    return result



def Chi_mol(b, rho_t, rho_p, Gamma):
    """
    Calculates the eikonal phase in the MOL model.
    
    Input parameters
        b     : (float), Impact parameter [fm]
        rho_t : (list/array size = t_numpoints), target density [fm]
        rho_p : (list/array size = t_numpoints), projectile density [fm]
        Gamma : (function(b)), profile function [1/fm^2]
    """
    return Chi_mol_1(b, rho_t, rho_p, Gamma) + Chi_mol_1(b, rho_p, rho_t, Gamma)


#_____________________________________________________________________________________________________________________________________
def chi(b, rho_t, rho_p, Gamma):
    """
    Calculates the eikonal phase in the OLA model.
    
    Input parameters
        b     : (float), Impact parameter [fm]
        rho_t : (list/array size = t_numpoints), target density [fm]
        rho_p : (list/array size = t_numpoints), projectile density [fm]
        Gamma : (function(b)), profile function [1/fm^2]

    uses s t and theta meshs
    """
    
    # import s mesh
    global sa, sb, s_weights, s_mapped_roots 
    global ta, tb, t_weights, t_mapped_roots 

    global t_theta_a, t_theta_b, t_theta_weights, t_theta_mapped_roots

    #defining integrand
    
    
    def chi_t(b,s, theta_s, theta_t):

        t_integrand = lambda t : 1j* s * t * Gamma(add_sub_vec_mag(b, 0, s, theta_s, t, theta_t)) 
       
        func_values = t_integrand(t_mapped_roots[:, np.newaxis, np.newaxis, np.newaxis]) * rho_t[:, np.newaxis, np.newaxis, np.newaxis]
        
        # Perform Gaussian Legendre integration
        t_result = np.sum(t_weights[:, np.newaxis, np.newaxis, np.newaxis] * func_values, axis = 0)

        return t_result
    
    def chi_theta_t(b,s, theta_s):

        t_theta_integrand = lambda theta_t : chi_t(b,s, theta_s, theta_t)

        func_values =t_theta_integrand(t_theta_mapped_roots[:, np.newaxis, np.newaxis])
        
        # Perform Gaussian Legendre integration
        t_theta_result = np.sum(t_theta_weights[:, np.newaxis, np.newaxis] * func_values, axis = 0) 

        return t_theta_result

    def chi_s(b, theta_s):

        s_integrand = lambda s: chi_theta_t(b,s, theta_s)

        func_values =s_integrand(s_mapped_roots[:, np.newaxis])* rho_p [:, np.newaxis]
        
        # Perform Gaussian Legendre integration
        s_result = np.sum(s_weights[:, np.newaxis] * func_values, axis = 0) 

        return s_result

    integrand = lambda theta_s: chi_s(b, theta_s)
    
    func_values =integrand(s_theta_mapped_roots)

        # Perform Gaussian Legendre integration
    result = np.sum(s_theta_weights * func_values, axis = 0)  

    return result

def chi_no_dens(b , rho, Gammap):
    """
    Calculates the eikonal phase in the OLA model for proton-nucleus scattering.
    
    Input parameters
        b     : (float), Impact parameter [fm]
        rho_t : (list/array size = t_numpoints), target density [fm]
        rho_p : (list/array size = t_numpoints), projectile density [fm]
        Gamma : (function(b)), profile function [1/fm^2]

    uses s, t and theta meshs
    """

    # import s mesh
    global sa, sb, s_weights, s_mapped_roots 
    global ta, tb, t_weights, t_mapped_roots 

    global t_theta_a, t_theta_b, t_theta_weights, t_theta_mapped_roots

    #defining integrand

    def chi_s(b, theta_s):

        s_integrand = lambda  s, : 1j * s * (Gammap(add_sub_vec_mag(b,0,s,theta_s,0,0))  + Gamman(add_sub_vec_mag(b,0,s,theta_s,0,0)) ) 

        
        func_values =s_integrand(s_mapped_roots[:, np.newaxis])* rho[:, np.newaxis]
        # Perform Gaussian Legendre integration
        s_result = np.sum(s_weights[:, np.newaxis] * func_values, axis = 0) 

        return s_result

    integrand = lambda theta_s: chi_s(b, theta_s)

    func_values =integrand(s_theta_mapped_roots)
        # Perform Gaussian Legendre integration
    result = np.sum(s_theta_weights * func_values, axis = 0)  

    return result

def sigma_R( rho_t,  rho_p  = 0, Gamma = lambda b: np.exp(-b), Model = "OLA"):

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
   
    global ba, bb, b_numpoints, b_roots, b_weights, b_mapped_roots
 
    if (Model == "OLA p-n"):
        
        #if only the target/projectile are composite particles
        sigma_R_nd_int =lambda b: 2 * np.pi * b * (1 - np.exp(- 2 * chi_no_dens(b, rho_t, Gamma ).imag ) )

        func_values = np.array( [sigma_R_nd_int(i) for i in b_mapped_roots.tolist()])
        
    elif (Model == "MOL"):
        sigma_R_int= lambda b: 2 * np.pi * b * (1 - np.exp(- 2 * Chi_mol(b, rho_t, rho_p, Gamma ).imag ) )
        func_values = np.array( [sigma_R_int(i) for i in b_mapped_roots.tolist()])
                
    else:
        #if both target and projectiles are composite particles 
        sigma_R_int= lambda b: 2 * np.pi * b * (1 - np.exp(- 2 * chi(b, rho_t, rho_p, Gamma ).imag ) )

        func_values = np.array( [sigma_R_int(i) for i in b_mapped_roots.tolist()])

    # Perform Gaussian Legendre integration
    result = np.sum(b_weights * func_values) 

    return result * 10

def chi_mol_micro(b, rho_t_p, rho_t_n, rho_p_n,rho_p_p, Gamma_pp, Gamma_pn, Gamma_nn ):
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
    
    chi_pp = Chi_mol(b, rho_t_p, rho_p_p, Gamma_pp )
    chi_pn = Chi_mol(b, rho_t_p, rho_p_n, Gamma_pn ) + Chi_mol(b, rho_t_n, rho_p_p, Gamma_pn )
    chi_nn = Chi_mol(b, rho_t_n, rho_p_n, Gamma_nn )
    return (chi_pp + chi_pn + chi_nn)

def chi_ola_micro(b, rho_t_p, rho_t_n, rho_p_p, rho_p_n, Gamma_pp, Gamma_pn, Gamma_nn ):

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
    
    chi_pp = chi(b, rho_t_p, rho_p_p, Gamma_pp )
    chi_pn = chi(b, rho_t_p, rho_p_n,  Gamma_pn) + chi(b, rho_t_n, rho_p_p, Gamma_pn) 
    chi_nn = chi(b, rho_t_n, rho_p_n, Gamma_nn )
    return (chi_pp + chi_pn + chi_nn)


def sigma_R_micro(rho_t_p, rho_t_n, rho_p_p, rho_p_n, Gamma_pp, Gamma_pn, Gamma_nn , Model = "OLA"):

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
    global ba, bb, b_numpoints, b_roots, b_weights, b_mapped_roots
        
    if (Model == "MOL"):
        sigma_R_int= lambda b: 2 * np.pi * b * (1 - np.exp(- 2 * chi_mol_micro(b, rho_t_p, rho_t_n, rho_p_p, rho_p_n, Gamma_pp, Gamma_pn, Gamma_nn).imag ) )
        func_values = np.array( [sigma_R_int(i) for i in b_mapped_roots.tolist()])
    
    else:
        #if both target and projectiles are composite particles 
        sigma_R_int= lambda b: 2 * np.pi * b * (1 - np.exp(- 2 * chi_ola_micro(b, rho_t_p, rho_t_n, rho_p_p, rho_p_n,Gamma_pp, Gamma_pn, Gamma_nn).imag ) )
        func_values = np.array( [sigma_R_int(i) for i in b_mapped_roots.tolist()])
        
    # Perform Gaussian Legendre integration
    result = np.sum(b_weights * func_values) 

    return result * 10

#________________________________________________________________________________________________________________________________________

def rm_rms(rho):
    """
    Given a function rho(r) calculates the rms matter radius of the nucleus
    """
    global A

    #defining integrand

    integrand = lambda r: 4 * np.pi* (r**4)  * rho(r)/A
    
    #bounds for the integral

    global ra, rb, r_numpoints, r_roots, r_weights, r_mapped_roots

    func_values = integrand(r_mapped_roots)

    # Perform Gaussian Legendre integration
    result = np.sum(r_weights * func_values)

    return np.sqrt(result)

def A_return(rho):
    """
    Given a function rho(r) calculates the A of the nucleus. This was ment to be a benchmark
    """
    
    global A

    #defining integrand
    
    integrand = lambda r :  4 * np.pi* (r**2)  * rho(r)
    
    #bounds for the integral

    global ra, rb, r_numpoints, r_roots, r_weights, r_mapped_roots

    func_values = integrand(r_mapped_roots)
    
    # Perform Gaussian Legendre integration
    result = np.sum(r_weights * func_values)

    return result

""" This is part of a routine that would calculate a two point fermi density given a matter density of the target which was never complete. I am not going to touch any of this until everything is finished. """

def A_min(params):
    global A
    C_m_p, a_m_p, rho_0_p = params
    rhom = lambda r : rho_m(r, C_m_p = params[0], a_m_p = params[1], rho_0_p = params[2])
    return abs(A_return(rhom) - A)

def sat_min(params):
    C_m_p, a_m_p, rho_0_p = params
    rhom = lambda r : rho_m(r, C_m_p = params[0], a_m_p = params[1], rho_0_p = params[2])
    return abs(rhom(0) - 0.176)

def rho_param_finder(E, array_rho_t, sigma_R_measured, dsigma, guess_param = [ 4.1, .5, .176] ):

    def sigma_min(params):
        C_m_p, a_m_p, rho_0_p = params
    
   
        array_rho_p = np.array( [rhoz_p(i) for i in s_mapped_roots.tolist()])
        
        return (sigma_R( E, array_rho_t , array_rho_p, Model = "MOL" )*10 - sigma_R_measured)**2/ ((dsigma)) 

    
    
    constraints=  [{'type': 'eq', 'fun': A_min},
                   {'type': 'eq', 'fun': sat_min}]

    return minimize(sigma_min, guess_param, constraints=constraints)
