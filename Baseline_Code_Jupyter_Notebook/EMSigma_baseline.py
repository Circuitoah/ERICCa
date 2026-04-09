"""
the difference between this and the other version of Functions.py Gamma is an input to the X and sigma 

Edit: Febuary 2nd, 2026.  Commenting the code in prep for publications if the code stops giving proper results please go to \Final_Results_with_Figures

Edit:Febuary 3rd, 2026. Found a bug in the code, bmax was set to rmax. This means our calculatations were off by ~5 mb. BUG HAS BEEN FIXED. Please use this version of the code.

Edit: March 23rd, 2026. Found a bug in chi_no_dens. The input file was  chi_no_dens(b , rho, Gammap) instead of  chi_no_dens(b , rho, Gamma). This apear during 5_14_25 update when Gamma was added as an input. This therefore post-dates the benchmarked that  chi_no_dens was used in. 

Edit: April 9th, 2026.  I found a bug in my code, in chi_mol_micro the rho_n_p and rho_p_n switched places in the function. THIS BUG HAS BEEN FIXED. Pleas euse this version of the code.
"""


import numpy as np
from numpy.polynomial import legendre
from scipy.interpolate import Rbf, CubicSpline, interp1d
from scipy.optimize import curve_fit, minimize
import scipy as sp

#_Constants and meshs

#
A = 12

#Constants for rho
A_p = 0
C_m_p = 0 
a_m_p = 0 
rho_0_p = 0

A_t = 0
C_m_t = 0 
a_m_t = 0 
rho_0_t = 0

#constant for C
sigma_R_measured = 218

#Constants for gamma

alpha =   1.808 
beta  = .268 
sigma_n = 3.16

alphapp = 1.808 
betapp = .268  
sigma_pp =3.16

alphapn = 0
betapn = 0
sigma_pn = 0

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

#r mesh
ra = 0 #fm 
rb = rmax #fm

r_roots, r_weights = legendre.leggauss(r_numpoints)
r_weights = r_weights* 0.5 * (rb - ra)
    # Map the roots from [-1, 1] to the interval [a, b]
r_mapped_roots = 0.5 * (r_roots + 1) * (rb - ra) + ra

#r mesh
ba = 0 #fm 
bb = bmax #fm        

b_roots, b_weights = legendre.leggauss(b_numpoints)
b_weights = b_weights* 0.5 * (bb - ba)
    # Map the roots from [-1, 1] to the interval [a, b]
b_mapped_roots = 0.5 * (b_roots + 1) * (bb - ba) + ba

#z mesh
za = - zmax#fm 
zb = zmax#fm 
z_roots, z_weights = legendre.leggauss(z_numpoints)

z_weights = z_weights * 0.5 * (zb - za)

# Map the roots from [-1, 1] to the interval [a, b]
z_mapped_roots = 0.5 * (z_roots + 1) * (zb - za) + za


#horrible 4d s and t vector mesh

# Get Legendre roots and weights for the specified number of points in each dimension

#s mesh
sa = 0 #fm 
sb = st_max #fm
s_numpoints = numpoints
s_roots, s_weights = legendre.leggauss(s_numpoints)

s_weights = s_weights * 0.5 * (sb - sa)

    # Map the roots from [-1, 1] to the interval [a, b]
s_mapped_roots = 0.5 * (s_roots + 1) * (sb - sa) + sa

#s theta mesh 
s_theta_a = 0 #fm 
s_theta_b = 2*np.pi #fm
s_theta_numpoints = numpoints_theta
s_theta_roots, s_theta_weights = legendre.leggauss(s_theta_numpoints)

s_theta_weights = s_theta_weights *  0.5 * (s_theta_b - s_theta_a)   

    # Map the roots from [-1, 1] to the interval [a, b]
s_theta_mapped_roots = 0.5 * (s_theta_roots + 1) * (s_theta_b - s_theta_a) + s_theta_a

#t mesh
ta =0  #fm 
tb = st_max #fm
t_numpoints = numpoints
t_roots, t_weights = legendre.leggauss(t_numpoints)

t_weights = t_weights * 0.5 * (tb - ta)
    # Map the roots from [-1, 1] to the interval [a, b]
t_mapped_roots = 0.5 * (t_roots + 1) * (tb - ta) + ta

#t theta mesh 
t_theta_a = 0 #fm 
t_theta_b = 2 * np.pi#fm
t_theta_numpoints = numpoints_theta
t_theta_roots, t_theta_weights = legendre.leggauss(t_theta_numpoints)

t_theta_weights = t_theta_weights* 0.5 * (t_theta_b - t_theta_a)

    # Map the roots from [-1, 1] to the interval [a, b]
t_theta_mapped_roots = 0.5 * (t_theta_roots + 1) * (t_theta_b - t_theta_a) + t_theta_a




#_ Vector operations____________________________________________________________________________________________________________________________

def add_sub_vec_mag(a, theta_a, b, theta_b, c, theta_c):
    arg_x =  a * np.cos(theta_a) + b * np.cos(theta_b) - c * np.cos(theta_c)
    arg_y =  a * np.sin(theta_a) + b * np.sin(theta_b) - c * np.sin(theta_c)

    return np.sqrt(arg_x**2 + arg_y**2)

#__functions_____________________________________________________________________________________________________________________________



def C( E):
    
    global  sigma_R_measured
    #Defining our variables needed for the function 

    #Caluation of neutron grazing distance
    B_c = np. sqrt(sigma_R_measured / np.pi )
    output = 1.0 + (B_c/E)
    return output

def rho_m(r):
# calcualating the projectile/target density

    global A_p, C_m_p, a_m_p, rho_0_p

    #C_m = 1.2 * A**(1/3) # half-density radius
    #a_m = 0.5 # fm diffuseness 

    #the saturation density
    #rho_0 = .176 * (1 + np.exp(( - C_m * A**(1/3))/a_m)) #fm^-3


    arg = (1 + np.exp((r - C_m_p)/a_m_p))

    return rho_0_p / arg


def Gamma(b):
# calculating the profile function 
   
    global alpha, beta, sigma_n

    # alpha  :   real and imaginary part of the scattering NN scattering amplitude
    # beta   :   finite range parameter
    # sigma_N:   does not need to be calculated 

    arg1 = (1 -  1j * alpha)/( 4 * np.pi * beta)
    arg2 = sigma_n * np.exp( - b**2/(2 *beta) ) 

    #arg1 = (1 -  1j * alpha)/(4 * np.pi)
    #arg2 = (sigma_n)* np.exp( - (beta * b**2)/2 )
    
    return arg1 * arg2 

def Gammap(b ):
# calculating the profile function 
   
    global alphapp, betapp, sigma_pp
    global alphapn, betapn, sigma_pn

    # alpha  :   real and imaginary part of the scattering NN scattering amplitude
    # beta   :   finite range parameter
    # sigma_N:   does not need to be calculated 

    arg1 = (1 -  1j * alphapp)/( 4 * np.pi * betapp)
    arg2 = sigma_pp * np.exp( - b**2/(2 *betapp) ) 

    arg3 = (1 -  1j * alphapn)/( 4 * np.pi * betapn)
    arg4 = sigma_pn * np.exp( - b**2/(2 *betapn) ) 

    return arg1 * arg2 


def Gamman(b ):
# calculating the profile function 
   
    global alphapp, betapp, sigma_pp
    global alphapn, betapn, sigma_pn

    # alpha  :   real and imaginary part of the scattering NN scattering amplitude
    # beta   :   finite range parameter
    # sigma_N:   does not need to be calculated 

    arg1 = (1 -  1j * alphapp)/( 4 * np.pi * betapp)
    arg2 = sigma_pp * np.exp( - b**2/(2 *betapp) ) 

    arg3 = (1 -  1j * alphapn)/( 4 * np.pi * betapn)
    arg4 = sigma_pn * np.exp( - b**2/(2 *betapn) ) 

    return arg3 * arg4 

def profile_funct_param(E, interaction_type = "matter"):
    
    global alphapp, betapp, sigma_pp
    global alphapn, betapn, sigma_pn
    global Gamma
    """Given an energy outputs parameters to the profile function
        alpha_nn, a, and sigma_nn"""
    
    
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
    
        Gamma = lambda b : Gamman(b) + Gammap(b)
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

            # alpha  :   real and imaginary part of the scattering NN scattering amplitude
            # beta   :   finite range parameter
            # sigma_N:   does not need to be calculated 

            arg1 = (1 -  1j * alpha)/( 4 * np.pi * beta)
            arg2 = sigma_n * np.exp( - b**2/(2 *beta) ) 

            #arg1 = (1 -  1j * alpha)/(4 * np.pi)
            #arg2 = (sigma_n)* np.exp( - (beta * b**2)/2 )
    
            return arg1 * arg2 
        Gamma = mGamma
        return 
        
    else :
        print("Something didn't work :-( , check to make sure your inputs are valid")
        return

    
    
def dens_b_interpolator(array_r, array_rho):
    """"""
    #rho_funct = Rbf(array_r, array_rho)
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
# calcualating the projectile/target density

    global A_p

    global za, zb, z_numpoints, z_roots, z_weights, z_mapped_roots

    integrand = lambda z : rho_m( np.sqrt(z**2 + b**2 ))
    #the saturation density
    
    #func_values = np.array( [integrand(i) for i in z_mapped_roots.tolist()])
    func_values = integrand(z_mapped_roots)
    
    #print(func_values)
    # Perform Gaussian Legendre integration
    result = np.sum(z_weights * func_values) 

    return result


# Functions that require integrals____________________________________________________________________________________________________

def Chi_mol_1(b, rho_t, rho_p, Gamma):

    # import s mesh
    global sa, sb, s_weights, s_mapped_roots 
    global ta, tb, t_weights, t_mapped_roots 

    global t_theta_a, t_theta_b, t_theta_weights, t_theta_mapped_roots

    #defining integrand

    def chi_t(b,s, theta_s, theta_t):

        t_integrand = lambda t :  t * Gamma(add_sub_vec_mag(b, 0, s, theta_s, t, theta_t)) 
       

        #func_values = np.array( [t_integrand(i) for i in t_mapped_roots.tolist()]) * rho_t
        func_values =t_integrand(t_mapped_roots[:, np.newaxis, np.newaxis, np.newaxis])*rho_t[:, np.newaxis, np.newaxis, np.newaxis]

        # Perform Gaussian Legendre integration
        t_result = np.sum(t_weights[:, np.newaxis, np.newaxis, np.newaxis] * func_values, axis= 0)

        return t_result
    
    def chi_theta_t(b,s, theta_s):

        t_theta_integrand = lambda theta_t : chi_t(b,s, theta_s, theta_t)

        #func_values = np.array( [t_theta_integrand(i)  for i in t_theta_mapped_roots.tolist()])
        func_values = t_theta_integrand(t_theta_mapped_roots[:, np.newaxis, np.newaxis])
        
        # Perform Gaussian Legendre integration
        t_theta_result = np.sum(t_theta_weights[:, np.newaxis, np.newaxis] * func_values, axis= 0) 

        return t_theta_result

    def chi_s(b, theta_s):

        s_integrand = lambda s: s * (1 - np.exp(-chi_theta_t(b,s, theta_s)) )
        
        #func_values = np.array( [s_integrand(i)  for i in s_mapped_roots.tolist()]) * rho_p
        func_values = s_integrand(s_mapped_roots[:, np.newaxis])* rho_p[:, np.newaxis]
        
        # Perform Gaussian Legendre integration
        s_result = np.sum(s_weights[:, np.newaxis] * func_values, axis = 0) 

        return s_result

    integrand = lambda theta_s: .5j * chi_s(b, theta_s)

    #func_values = np.array( [integrand(i) for i in s_theta_mapped_roots.tolist()])
    func_values = integrand(s_theta_mapped_roots)
                                  
        # Perform Gaussian Legendre integration
    result = np.sum(s_theta_weights * func_values, axis = 0)  
    
    
    return result



def Chi_mol(b, rho_t, rho_p, Gamma):
    #print(Chi_mol_1(b, rho_t, rho_p, Gamma) + Chi_mol_1(b, rho_p, rho_t, Gamma))
    return Chi_mol_1(b, rho_t, rho_p, Gamma) + Chi_mol_1(b, rho_p, rho_t, Gamma)


#_____________________________________________________________________________________________________________________________________
def chi(b, rho_t, rho_p, Gamma):

    # import s mesh
    global sa, sb, s_weights, s_mapped_roots 
    global ta, tb, t_weights, t_mapped_roots 

    global t_theta_a, t_theta_b, t_theta_weights, t_theta_mapped_roots

    #defining integrand
    
    #array_rho_t = np.array( [rho_t(i) for i in t_mapped_roots.tolist()])
    #array_rho_p = np.array( [rho_p(i) for i in s_mapped_roots.tolist()])

    def chi_t(b,s, theta_s, theta_t):

        #t_integrand = lambda t : s**2 + theta_s**2 + t**2 +  theta_t**2 

        t_integrand = lambda t : 1j* s * t * Gamma(add_sub_vec_mag(b, 0, s, theta_s, t, theta_t)) 
       

        #func_values = np.array( [t_integrand(i) for i in t_mapped_roots.tolist()]) * rho_t
        func_values = t_integrand(t_mapped_roots[:, np.newaxis, np.newaxis, np.newaxis]) * rho_t[:, np.newaxis, np.newaxis, np.newaxis]
        
        # Perform Gaussian Legendre integration
        t_result = np.sum(t_weights[:, np.newaxis, np.newaxis, np.newaxis] * func_values, axis = 0)

        return t_result
    
    def chi_theta_t(b,s, theta_s):

        t_theta_integrand = lambda theta_t : chi_t(b,s, theta_s, theta_t)

        #func_values = np.array( [t_theta_integrand(i)  for i in t_theta_mapped_roots.tolist()])

        func_values =t_theta_integrand(t_theta_mapped_roots[:, np.newaxis, np.newaxis])
        
        # Perform Gaussian Legendre integration
        t_theta_result = np.sum(t_theta_weights[:, np.newaxis, np.newaxis] * func_values, axis = 0) 

        return t_theta_result

    def chi_s(b, theta_s):

        s_integrand = lambda s: chi_theta_t(b,s, theta_s)

        #func_values = np.array( [s_integrand(i)  for i in s_mapped_roots.tolist()]) * rho_p
        func_values =s_integrand(s_mapped_roots[:, np.newaxis])* rho_p [:, np.newaxis]
        
        # Perform Gaussian Legendre integration
        s_result = np.sum(s_weights[:, np.newaxis] * func_values, axis = 0) 

        return s_result

    integrand = lambda theta_s: chi_s(b, theta_s)

    #func_values = np.array( [integrand(i) for i in s_theta_mapped_roots.tolist()])
    func_values =integrand(s_theta_mapped_roots)

        # Perform Gaussian Legendre integration
    result = np.sum(s_theta_weights * func_values, axis = 0)  

    return result

def chi_no_dens(b , rho, Gamma):

    # import s mesh
    global sa, sb, s_weights, s_mapped_roots 
    global ta, tb, t_weights, t_mapped_roots 

    global t_theta_a, t_theta_b, t_theta_weights, t_theta_mapped_roots

    #defining integrand
    
    #array_rho_t = np.array( [rho_t(i) for i in t_mapped_roots.tolist()])
    #array_rho_p = np.array( [rho_p(i) for i in s_mapped_roots.tolist()])

    def chi_s(b, theta_s):

        s_integrand = lambda  s, : 1j * s * (Gamma(add_sub_vec_mag(b,0,s,theta_s,0,0))) 

        #func_values = np.array( [s_integrand(i)  for i in s_mapped_roots.tolist()]) * rho
        func_values =s_integrand(s_mapped_roots[:, np.newaxis])* rho[:, np.newaxis]
        # Perform Gaussian Legendre integration
        s_result = np.sum(s_weights[:, np.newaxis] * func_values, axis = 0) 

        return s_result

    integrand = lambda theta_s: chi_s(b, theta_s)

    #func_values = np.array( [integrand(i) for i in s_theta_mapped_roots.tolist()])
    func_values =integrand(s_theta_mapped_roots)
        # Perform Gaussian Legendre integration
    result = np.sum(s_theta_weights * func_values, axis = 0)  

    return result

def sigma_R( rho_t,  rho_p  = 0, Gamma = lambda b: np.exp(-b), Model = "OLA"):
   
    global ba, bb, b_numpoints, b_roots, b_weights, b_mapped_roots
 
    if (Model == "OLA p-n"):
        
        #if only the target/projectile are composite particles
        sigma_R_nd_int =lambda b: 2 * np.pi * b * (1 - np.exp(- 2 * chi_no_dens(b, rho_t, Gamma ).imag ) )

        func_values = np.array( [sigma_R_nd_int(i) for i in b_mapped_roots.tolist()])
        
    elif (Model == "MOL"):
        sigma_R_int= lambda b: 2 * np.pi * b * (1 - np.exp(- 2 * Chi_mol(b, rho_t, rho_p, Gamma ).imag ) )
        func_values = np.array( [sigma_R_int(i) for i in b_mapped_roots.tolist()])
        #func_values =sigma_R_int(b_mapped_roots)
                
    else:
        #if both target and projectiles are composite particles 
        sigma_R_int= lambda b: 2 * np.pi * b * (1 - np.exp(- 2 * chi(b, rho_t, rho_p, Gamma ).imag ) )

        func_values = np.array( [sigma_R_int(i) for i in b_mapped_roots.tolist()])

        

    #plt.plot(r_mapped_roots,func_values)

    #print(func_values)
    #print(r_mapped_roots , sigma_R_int(r_mapped_roots) #*r_weights *0.5 * (rb - ra))

    # Perform Gaussian Legendre integration
    result = np.sum(b_weights * func_values) 

    return result * 10

def chi_mol_micro(b, rho_t_p, rho_t_n, rho_p_p ,rho_p_n, Gamma_pp, Gamma_pn, Gamma_nn ):
    chi_pp = Chi_mol(b, rho_t_p, rho_p_p, Gamma_pp )
    chi_pn = Chi_mol(b, rho_t_p, rho_p_n, Gamma_pn ) + Chi_mol(b, rho_t_n, rho_p_p, Gamma_pn )
    chi_nn = Chi_mol(b, rho_t_n, rho_p_n, Gamma_nn )
    return (chi_pp + chi_pn + chi_nn)

def chi_ola_micro(b, rho_t_p, rho_t_n, rho_p_p, rho_p_n, Gamma_pp, Gamma_pn, Gamma_nn ):
    chi_pp = chi(b, rho_t_p, rho_p_p, Gamma_pp )
    chi_pn = chi(b, rho_t_p, rho_p_n,  Gamma_pn) + chi(b, rho_t_n, rho_p_p, Gamma_pn) 
    chi_nn = chi(b, rho_t_n, rho_p_n, Gamma_nn )
    return (chi_pp + chi_pn + chi_nn)


def sigma_R_micro(rho_t_p, rho_t_n, rho_p_p, rho_p_n, Gamma_pp, Gamma_pn, Gamma_nn , Model = "OLA"):
    
    global ba, bb, b_numpoints, b_roots, b_weights, b_mapped_roots
        
    if (Model == "MOL"):
        sigma_R_int= lambda b: 2 * np.pi * b * (1 - np.exp(- 2 * chi_mol_micro(b, rho_t_p, rho_t_n, rho_p_p, rho_p_n, Gamma_pp, Gamma_pn, Gamma_nn).imag ) )
        func_values = np.array( [sigma_R_int(i) for i in b_mapped_roots.tolist()])
        #func_values =sigma_R_int(b_mapped_roots)

    
    else:
        #if both target and projectiles are composite particles 
        sigma_R_int= lambda b: 2 * np.pi * b * (1 - np.exp(- 2 * chi_ola_micro(b, rho_t_p, rho_t_n, rho_p_p, rho_p_n,Gamma_pp, Gamma_pn, Gamma_nn).imag ) )
        func_values = np.array( [sigma_R_int(i) for i in b_mapped_roots.tolist()])
        
    # Perform Gaussian Legendre integration
    result = np.sum(b_weights * func_values) 

    return result * 10

#________________________________________________________________________________________________________________________________________

def rm_rms(rho):

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
    
    global A

    #defining integrand
    
    integrand = lambda r :  4 * np.pi* (r**2)  * rho(r)
    
    #bounds for the integral

    global ra, rb, r_numpoints, r_roots, r_weights, r_mapped_roots

    func_values = integrand(r_mapped_roots)
    #func_values = np.array( [integrand(i) for i in r_mapped_roots.tolist()])

    # Perform Gaussian Legendre integration
    result = np.sum(r_weights * func_values)

    return result


def A_min(params):
    global A
    global C_m_p, a_m_p, rho_0_p
    C_m_p, a_m_p, rho_0_p = params
    rhom = rho_m
    
    return abs(A_return(rhom) - A)

def sat_min(params):
    global C_m_p, a_m_p, rho_0_p
    C_m_p, a_m_p, rho_0_p = params
    rhom = rho_m
    return abs(rhom(0) - 0.176)

def rho_param_finder(E, array_rho_t, sigma_R_measured, dsigma, guess_param = [ 4.1, .5, .176] ):

    def sigma_min(params):
        global C_m_p, a_m_p, rho_0_p
        C_m_p, a_m_p, rho_0_p = params
    
   
        array_rho_p = np.array( [rhoz_p(i) for i in s_mapped_roots.tolist()])
        
        return (sigma_R( E, array_rho_t , array_rho_p, Model = "MOL" )*10 - sigma_R_measured)**2/ ((dsigma)) 

    
    
    constraints=  [{'type': 'eq', 'fun': A_min},
                   {'type': 'eq', 'fun': sat_min}]

    return minimize(sigma_min, guess_param, constraints=constraints)
