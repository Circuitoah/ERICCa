#importing the necessary libraries and modules for test
import matplotlib.pyplot as plt
import numpy as np
import EMSigma_baseline as EM0_base
import _init_ as EM0_Class

cross_section = EM0_Class.cross_section()
Profile_Function = EM0_Class.Profile_Function()

Test_results =[]

#Test 1: Testing the add_sub_vec_mag function for both the baseline and the updated code to ensure they produce the same results.
#test vectors
a, theta_a = 2.0, np.pi/3.0
b, theta_b = 3.0, np.pi/4.0
c, theta_c = 4.0, np.pi/6.0

Baseline_t1 = EM0_base.add_sub_vec_mag(a, theta_a, b, theta_b, c, theta_c)
Updated_t1 = cross_section.add_sub_vec_mag(a, theta_a, b, theta_b, c, theta_c)

if np.isclose(Baseline_t1, Updated_t1):
    Test_results.append(True)

else:
    Test_results.append(False)
    print("Test 1 Failed: The add_sub_vec_mag function produces different results for the baseline and updated code.")
    print("Baseline    :", Baseline_t1)
    print("Updated Code:", Updated_t1)

#Test 2: Testing the Density function for both the baseline and the updated code to ensure they produce the same results.
test_dens = EM0_Class.Density()

test_mesh = np.linspace(0.01,5, 30)

EM0_base.C_m_p = 1 
EM0_base.a_m_p = 1 
EM0_base.rho_0_p = 1

test_dens.C_m_p = 1 
test_dens.a_m_p = 1
test_dens.rho_0_p = 1


Baseline_t2 = EM0_base.rho_m(test_mesh )[-1]
Updated_t2 = test_dens.rho_m(test_mesh )[-1]

if np.isclose(Baseline_t2, Updated_t2):
    Test_results.append(True)

else:
    Test_results.append(False)
    print("Test 2 Failed: The Density function produces different results for the baseline and updated code.")
    print("Baseline    :", Baseline_t2)
    print("Updated Code:", Updated_t2)


#Profile function tests

#Test 3: Testing the Gamma function for both the baseline and the updated code to ensure they produce the same results.
b = 3.0
Baseline_t3 = EM0_base.Gamma(b)

Profile_Function.alpha = 1.808
Profile_Function.beta =  .268 
Profile_Function.sigma_n = 3.16

Updated_t3 = Profile_Function.Gamma(b)

if np.isclose(Baseline_t3, Updated_t3):
    Test_results.append(True)
else:
    Test_results.append(False)
    print("Test 3 Failed: The Profile_Function.Gamma(b) produces different results for the baseline and updated code.")
    print("Baseline    :", Baseline_t3)
    print("Updated Code:", Updated_t3)

#Test 4: Testing the profile function using the "matter" setting
E = 325 #MeV
EM0_base.profile_funct_param(E, interaction_type = "matter")
Profile_Function = EM0_Class.Profile_Function(Model_type = "matter", E = 325)

b = 3.0
Baseline_t4 = EM0_base.Gamma(b)
Updated_t4 = Profile_Function.Gamma(b)

if np.isclose(Baseline_t4, Updated_t4):
    Test_results.append(True)
else:    
    Test_results.append(False)
    print("Test 4 Failed: The Profile function with interaction_type ='matter' produces different results for the baseline and updated code when using the matter setting.")
    print("Baseline    :", Baseline_t4)
    print("Updated Code:", Updated_t4)

#Test 5: Testing the profile function using the "np" setting
E = 300 #MeV
EM0_base.profile_funct_param(E, interaction_type = "np")
Profile_Function = EM0_Class.Profile_Function(Model_type = "np", E = 300)

b = 3.0
Baseline_t5 = EM0_base.Gamma(b)
Updated_t5 = Profile_Function.Gamma_pp(b) + Profile_Function.Gamma_pn(b) 

if np.isclose(Baseline_t5, Updated_t5):
    Test_results.append(True)
else:
    Test_results.append(False)
    print("Test 5 Failed: The Profile function with interaction_type ='np' produces different results for the baseline and updated code when using the np setting.")
    print("Baseline    :", Baseline_t5)
    print("Updated Code:", Updated_t5)

#Cross section tests

#Test 6: Testing the density mesh interpolator for both the baseline and the updated code to ensure they produce the same results.
C = np.genfromtxt("Cor_dens_Update/Cor_Ca_C_Mass_dens/12Crho-mass.txt", unpack= True) 
C_r_mesh = np.genfromtxt("Cor_dens_Update/Cor_Ca_C_Mass_dens/12C_radius.txt", unpack= True)

C_rho_baseline = EM0_base.dens_b_interpolator(C_r_mesh[0],C[0])
C_rho_update = cross_section.dens_b_interpolator(C_r_mesh[0],C[0])

Baseline_t6 = C_rho_baseline[0]
Updated_t6 = C_rho_update[0]

if np.isclose(Baseline_t6, Updated_t6):
    Test_results.append(True)
else:
    Test_results.append(False)
    print("Test 6 Failed: The density mesh interpolator produces different results for the baseline and updated code.")
    print("Baseline    :", Baseline_t6)
    print("Updated Code:", Updated_t6)

    plt.plot(EM0_base.t_mapped_roots.tolist(),C_rho_baseline, color = "blue", label = "Baseline")
    plt.plot( cross_section.t_mapped_roots.tolist(),C_rho_update, color = "red", label = "Update")

    plt.xlabel("b")
    plt.ylabel(r"$\rho(b)$")
    plt.legend()
    plt.show()

#Load in densities for tests
C_r_Filename = np.array(["Cor_dens_Update/Cor_Ca_C_Mass_dens/12C_radius.txt",])     
C_p_Filename = np.array(["Cor_dens_Update/Cor_Ca_C_Mass_dens/12Crho-prot.txt",])
C_n_Filename = np.array(["Cor_dens_Update/Cor_Ca_C_Mass_dens/12Crho-neut.txt",])
C_p = np.genfromtxt(C_p_Filename[0], unpack= True)
C_r_mesh = np.genfromtxt(C_r_Filename[0], unpack= True)
C_n = np.genfromtxt(C_n_Filename[0], unpack= True)

C_rho_p = cross_section.dens_b_interpolator(C_r_mesh[0], C_p[0]) 
C_rho_n = cross_section.dens_b_interpolator(C_r_mesh[0], C_n[0])

Ca_r_Filename = np.array(["Cor_dens_Update/Cor_Ca_C_Mass_dens/42Ca_radius.txt",])     
Ca_p_Filename = np.array(["Cor_dens_Update/Cor_Ca_C_Mass_dens/42Carho-prot.txt",])
Ca_n_Filename = np.array(["Cor_dens_Update/Cor_Ca_C_Mass_dens/42Carho-neut.txt",])
Ca_p = np.genfromtxt(Ca_p_Filename[0], unpack= True)
Ca_r_mesh = np.genfromtxt(Ca_r_Filename[0], unpack= True)
Ca_n = np.genfromtxt(Ca_n_Filename[0], unpack= True)

Ca_rho_p = cross_section.dens_b_interpolator(Ca_r_mesh[0], Ca_p[0]) 
Ca_rho_n = cross_section.dens_b_interpolator(Ca_r_mesh[0], Ca_n[0])

#Test 7: Testing the Chi_mol_1 function for both the baseline and the updated code to ensure they produce the same results when using the "matter" setting for the profile function.
b = 3
rho_t = C_rho_p + C_rho_n
rho_p = Ca_rho_p +Ca_rho_n

EM0_base.profile_funct_param(E, interaction_type = "matter")
Profile_Function = EM0_Class.Profile_Function(Model_type = "matter", E = E)

Baseline_t7 = EM0_base.Chi_mol_1(b, rho_t, rho_p, EM0_base.Gamma)
Updated_t7 = cross_section.Chi_mol_1(b, rho_t, rho_p, Profile_Function.Gamma)

if np.isclose(Baseline_t7, Updated_t7):
    Test_results.append(True)
else:    
    Test_results.append(False)
    print("Test 7 Failed: The Chi_mol_1 function produces different results for the baseline and updated code when using the matter setting for the profile function.")
    print("Baseline    :", Baseline_t7 )
    print("Updated Code:",Updated_t7)

#Test 8: Testing the Chi_mol function for both the baseline and the updated code to ensure they produce the same results.
Baseline_t8 = EM0_base.Chi_mol(b, rho_t, rho_p, EM0_base.Gamma)
Updated_t8 = cross_section.Chi_mol(b, rho_t, rho_p, Profile_Function.Gamma)

if np.isclose(Baseline_t8, Updated_t8):
    Test_results.append(True)
else:    
    Test_results.append(False)
    print("Test 8 Failed: The Chi_mol function produces different results for the baseline and updated code when using the matter setting for the profile function.")
    print("Baseline    :", Baseline_t8)
    print("Updated Code:", Updated_t8)

#Test 9: Testing the chi function for both the baseline and the updated code to ensure they produce the same results.
Baseline_t9 = EM0_base.chi(b, rho_t, rho_p, EM0_base.Gamma)
Updated_t9 = cross_section.chi(b, rho_t, rho_p, Profile_Function.Gamma)

if np.isclose(Baseline_t9, Updated_t9):
    Test_results.append(True)
else:    
    Test_results.append(False)
    print("Test 9 Failed: The chi function produces different results for the baseline and updated code when using the matter setting for the profile function.")
    print("Baseline    :", Baseline_t9)
    print("Updated Code:", Updated_t9)

#Test 10: Testing the chi_no_dens function for both the baseline and the updated code to ensure they produce the same results.
Baseline_t10 = EM0_base.chi_no_dens(b , C_rho_p + C_rho_p, EM0_base.Gamma)
Updated_t10 = cross_section.chi_no_dens(b , C_rho_p, C_rho_p, EM0_base.Gamma, EM0_base.Gamma)

if np.isclose(Baseline_t10, Updated_t10):
    Test_results.append(True)
else:    
    Test_results.append(False)
    print("Test 10 Failed: The chi_no_dens function produces different results for the baseline and updated code when using the matter setting for the profile function.")
    print("Baseline    :", Baseline_t10)
    print("Updated Code:", Updated_t10)

#Test 11: Testing the sigma_R_matter function for both the baseline and the updated code to ensure they produce the same results when using the "matter" setting for the profile function.
#in the model ="MOL"
Baseline_t11 = EM0_base.sigma_R( rho_t,  rho_p  = rho_p , Gamma = EM0_base.Gamma, Model = "MOL")
Updated_t11 = cross_section.sigma_R_matter( rho_t,  rho_p  = rho_p , Gamma = Profile_Function.Gamma, Model = "MOL")

if np.isclose(Baseline_t11, Updated_t11):
    Test_results.append(True)
else:    
    Test_results.append(False)
    print("Test 11 Failed: The sigma_R_matter function produces different results for the baseline and updated code when using the matter setting for the profile function")
    print("In the MOL model")
    print("Baseline    :", Baseline_t11)
    print("Updated Code:", Updated_t11)

#Test 12: Testing the sigma_R_matter function for both the baseline and the updated code to ensure they produce the same results when using the "matter" setting for the profile function.
#in the model ="OLA"
Baseline_t12 = EM0_base.sigma_R( rho_t,  rho_p  = rho_p , Gamma = EM0_base.Gamma, Model = "OLA")
Updated_t12 = cross_section.sigma_R_matter( rho_t,  rho_p  = rho_p , Gamma = Profile_Function.Gamma, Model = "OLA")

if np.isclose(Baseline_t12, Updated_t12):
    Test_results.append(True)
else:    
    Test_results.append(False)
    print("Test 12 Failed: The sigma_R_matter function produces different results for the baseline and updated code when using the matter setting for the profile function")
    print("In the OLA model")
    print("Baseline    :", Baseline_t12)
    print("Updated Code:", Updated_t12)
    

#Test 13: Testing the sigma_R_pn function for both the baseline and the updated code to ensure they produce the same results when using the "np" setting for the profile function.
E = 200 #MeV
EM0_base.profile_funct_param(E, interaction_type = "np")
Profile_Function = EM0_Class.Profile_Function(Model_type = "np", E = E)

b_mesh = np.linspace(0, 10, 100)
test_gamma = lambda b: Profile_Function.Gamma_pp(b) + Profile_Function.Gamma_pn(b)

Baseline_t13 = EM0_base.sigma_R( C_rho_p, Gamma = EM0_base.Gamma, Model = "OLA p-n")
Updated_t13 = cross_section.sigma_R_pn( C_rho_p, C_rho_p, Gamma_pp = Profile_Function.Gamma_pp, 
                                       Gamma_pn = Profile_Function.Gamma_pn, Model = "OLA p-n")

if np.isclose(Baseline_t13, Updated_t13):
    Test_results.append(True)
else:
    Test_results.append(False)
    print("Test 13 Failed: The sigma_R_pn function produces different results for the baseline and updated code when using the np setting for the profile function")
    print("In the OLA p-n model")
    print("Baseline    :", Baseline_t13)
    print("Updated Code:", Updated_t13)

    plt.plot(b_mesh, EM0_base.Gamma(b_mesh), color = "b")
    plt.plot(b_mesh, test_gamma(b_mesh), color = "r" )
    plt.xlabel("b")
    plt.ylabel("$\Gamma(b)$")
    plt.legend()
    plt.show()

#Test 14: Testing the chi_mol_micro function for both the baseline and the updated code to ensure they produce the same results when using the "np" setting for the profile function.
b = 3
rho_t_p = C_rho_p 
rho_t_n = C_rho_n
rho_p_p = Ca_rho_p 
rho_p_n = Ca_rho_n

Baseline_t14 = EM0_base.chi_mol_micro(b, rho_t_p, rho_t_n, rho_p_p, rho_p_n, EM0_base.Gammap, EM0_base.Gamman, EM0_base.Gammap)
Updated_t14 = cross_section.chi_mol_micro(b, rho_t_p, rho_t_n, rho_p_p,rho_p_n, EM0_base.Gammap, EM0_base.Gamman, EM0_base.Gammap)

if np.isclose(Baseline_t14, Updated_t14):
    Test_results.append(True)
else:
    Test_results.append(False)
    print("Test 14 Failed: The chi_mol_micro function produces different results for the baseline and updated code when using the np setting for the profile function")
    print("In the np model")
    print("Baseline    :", Baseline_t14)
    print("Updated Code:", Updated_t14)

#Test 15: Testing the chi_ola_micro function for both the baseline and the updated code to ensure they produce the same results when using the "np" setting for the profile function.
Baseline_t15 = EM0_base.chi_ola_micro(b, rho_t_p, rho_t_n, rho_p_p,rho_p_n, EM0_base.Gammap, EM0_base.Gamman, EM0_base.Gammap)
Updated_t15 = cross_section.chi_ola_micro(b, rho_t_p, rho_t_n, rho_p_p,rho_p_n, EM0_base.Gammap, EM0_base.Gamman, EM0_base.Gammap)

if np.isclose(Baseline_t15, Updated_t15):
    Test_results.append(True)
else:
    Test_results.append(False)
    print("Test 15 Failed: The chi_ola_micro function produces different results for the baseline and updated code when using the np setting for the profile function")
    print("In the np model")
    print("Baseline    :", Baseline_t15)
    print("Updated Code:", Updated_t15)

#Test 16: Testing the sigma_R_pn function for both the baseline and the updated code to ensure they produce the same results when using the "np" setting for the profile function.
#Model = "OLA"
Baseline_t16 = EM0_base.sigma_R_micro(rho_t_p, rho_t_n, rho_p_p, rho_p_n, 
                             EM0_base.Gammap, EM0_base.Gamman, EM0_base.Gammap , Model = "OLA")
Updated_t16 = cross_section.sigma_R_pn(rho_t_p, rho_t_n, rho_p_p, rho_p_n, 
                             EM0_base.Gammap, EM0_base.Gamman, EM0_base.Gammap, Model = "OLA")

if np.isclose(Baseline_t16, Updated_t16):
    Test_results.append(True)
else:
    Test_results.append(False)
    print("Test 16 Failed: The sigma_R_pn function produces different results for the baseline and updated code when using the np setting for the profile function")
    print("In the OLA model")
    print("Baseline    :",Baseline_t16)
    print("Updated Code:", Updated_t16)

#Test 17: Testing the sigma_R_pn function for both the baseline and the updated code to ensure they produce the same results when using the "np" setting for the profile function.
#Model = "MOL"
Baseline_t17 = EM0_base.sigma_R_micro(rho_t_p, rho_t_n, rho_p_p, rho_p_n, 
                             EM0_base.Gammap, EM0_base.Gamman, EM0_base.Gammap , Model = "MOL")
Updated_t17 = cross_section.sigma_R_pn(rho_t_p, rho_t_n, rho_p_p, rho_p_n, 
                             EM0_base.Gammap, EM0_base.Gamman, EM0_base.Gammap, Model = "MOL")

if np.isclose(Baseline_t17, Updated_t17):
    Test_results.append(True)
else:    
    Test_results.append(False)
    print("Test 17 Failed: The sigma_R_pn function produces different results for the baseline and updated code when using the np setting for the profile function")
    print("In the MOL model")
    print("Baseline    :",Baseline_t17)
    print("Updated Code:",Updated_t17)

#Test 18: Testing the density function for both the baseline and the updated code to ensure they produce the same results.
test_dens = EM0_Class.Density()
A = 42
Z = 20 

Baseline_t18 =[3.4481556494276426, 3.4481556494276426, 0.014023505542324788]
Updated_t18 = [test_dens.rms(Ca_r_mesh[1] ,Ca_p[1] + Ca_n[1], A), test_dens.rms(Ca_r_mesh[1], Ca_p[1] + Ca_n[1], A), -test_dens.rms(Ca_r_mesh[1], Ca_p[1],Z) + test_dens.rms(Ca_r_mesh[1], Ca_n[1], A-Z ) ]
if np.isclose(Baseline_t18, Updated_t18).all():
    Test_results.append(True)
else:    
    Test_results.append(False)
    print("Test 18 Failed: The density function produces different results for the baseline and updated code.")
    print ("Baseline     :",Baseline_t18)
    print ("Updated code :", Updated_t18)

if all(Test_results):
    print("All tests passed successfully!")
else:
    failed_tests = [i+1 for i, result in enumerate(Test_results) if not result]
    print(f"Tests {failed_tests} failed. Please review the output for details.")
