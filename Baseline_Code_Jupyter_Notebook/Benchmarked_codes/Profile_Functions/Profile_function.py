import numpy as np
from scipy.interpolate import CubicSpline


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
        
            self.Gamma_pn = lambda b :(1 -  1j * alphapn)/( 4 * np.pi * betapn)*sigma_pn * np.exp( - b**2/(2 *betapn) )
            self.Gamma_pp = lambda b :(1 -  1j * alphapp)/( 4 * np.pi * betapp) * sigma_pp * np.exp( - b**2/(2 *betapp) )


        if (Model_type == "matter"):
    
            profile_funct_table = np.genfromtxt("profile_funct_param_matter.txt", unpack = True, skip_header= 2)

            sigma_NN_fun = CubicSpline(profile_funct_table[0],profile_funct_table[1])
            alpha_fun = CubicSpline(profile_funct_table[0],profile_funct_table[2])
            beta_fun  = CubicSpline(profile_funct_table[0],profile_funct_table[3])

            self.sigma_n = sigma_NN_fun(E)
            self.alpha = alpha_fun(E)
            self.beta  = beta_fun(E)
