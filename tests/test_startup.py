"""
Baseline comparison tests.

Each test calls both the legacy Baseline module and the refactored class-based
API and asserts they produce numerically identical results.  Tests marked
``slow`` involve full sigma_R computations and take a few seconds each.
"""
import numpy as np
import pytest

import ERICCA.baseline as Baseline
from ERICCA import CrossSection, Density, ProfileFunction


# ---------------------------------------------------------------------------
# Vector operations
# ---------------------------------------------------------------------------

def test_add_sub_vec_mag(cs):
    a, theta_a = 2.0, np.pi / 3.0
    b, theta_b = 3.0, np.pi / 4.0
    c, theta_c = 4.0, np.pi / 6.0

    expected = Baseline.add_sub_vec_mag(a, theta_a, b, theta_b, c, theta_c)
    result   = cs.add_sub_vec_mag(a, theta_a, b, theta_b, c, theta_c)

    assert np.isclose(expected, result), f"Expected {expected}, got {result}"


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------

def test_rho_m():
    mesh = np.linspace(0.01, 5, 30)

    Baseline.C_m_p   = 1
    Baseline.a_m_p   = 1
    Baseline.rho_0_p = 1

    dens = Density()
    dens.C_m_p   = 1
    dens.a_m_p   = 1
    dens.rho_0_p = 1

    assert np.isclose(Baseline.rho_m(mesh)[-1], dens.rho_m(mesh)[-1])


# ---------------------------------------------------------------------------
# Profile function
# ---------------------------------------------------------------------------

def test_gamma_general():
    b = 3.0
    Baseline.alpha   = 1.808
    Baseline.beta    = 0.268
    Baseline.sigma_n = 3.16

    pf = ProfileFunction()
    pf.alpha   = 1.808
    pf.beta    = 0.268
    pf.sigma_n = 3.16

    assert np.isclose(Baseline.Gamma(b), pf.Gamma(b))


def test_gamma_matter():
    E = 325
    Baseline.profile_funct_param(E, interaction_type="matter")
    pf = ProfileFunction(model_type="matter", E=E)

    assert np.isclose(Baseline.Gamma(3.0), pf.Gamma(3.0))


def test_gamma_np():
    E = 300
    Baseline.profile_funct_param(E, interaction_type="np")
    pf = ProfileFunction(model_type="np", E=E)

    baseline_gamma = Baseline.Gamma(3.0)
    updated_gamma  = pf.Gamma_pp(3.0) + pf.Gamma_pn(3.0)

    assert np.isclose(baseline_gamma, updated_gamma)


def test_profile_function_invalid_model():
    with pytest.raises(ValueError, match="Unknown model_type"):
        ProfileFunction(model_type="bad_model")


# ---------------------------------------------------------------------------
# Density interpolator
# ---------------------------------------------------------------------------

def test_dens_b_interpolator(cs):
    r_mesh = np.linspace(0.01, 15, 100)
    dens = Density()
    dens.rho_m_2pt_fermi(12, 2.32)
    rho = dens.rho_m(r_mesh)

    expected = Baseline.dens_b_interpolator(r_mesh, rho)[0]
    result   = cs.dens_b_interpolator(r_mesh, rho)[0]

    assert np.isclose(expected, result)


# ---------------------------------------------------------------------------
# Eikonal phase functions  (matter profile, E=300)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def matter_profile_300():
    E = 300
    Baseline.profile_funct_param(E, interaction_type="matter")
    return ProfileFunction(model_type="matter", E=E)


def test_chi_mol_half(cs, densities, matter_profile_300):
    b     = 3
    rho_t = densities["C_rho_p"] + densities["C_rho_n"]
    rho_p = densities["Ca_rho_p"] + densities["Ca_rho_n"]

    expected = Baseline.Chi_mol_1(b, rho_t, rho_p, Baseline.Gamma)
    result   = cs._chi_mol_half(b, rho_t, rho_p, matter_profile_300.Gamma)

    assert np.isclose(expected, result)


def test_chi_mol(cs, densities, matter_profile_300):
    b     = 3
    rho_t = densities["C_rho_p"] + densities["C_rho_n"]
    rho_p = densities["Ca_rho_p"] + densities["Ca_rho_n"]

    expected = Baseline.Chi_mol(b, rho_t, rho_p, Baseline.Gamma)
    result   = cs.chi_mol(b, rho_t, rho_p, matter_profile_300.Gamma)

    assert np.isclose(expected, result)


def test_chi_ola(cs, densities, matter_profile_300):
    b     = 3
    rho_t = densities["C_rho_p"] + densities["C_rho_n"]
    rho_p = densities["Ca_rho_p"] + densities["Ca_rho_n"]

    expected = Baseline.chi(b, rho_t, rho_p, Baseline.Gamma)
    result   = cs.chi(b, rho_t, rho_p, matter_profile_300.Gamma)

    assert np.isclose(expected, result)


def test_chi_no_dens(cs, densities, matter_profile_300):
    b       = 3
    rho_sum = densities["C_rho_p"] + densities["C_rho_p"]

    expected = Baseline.chi_no_dens(b, rho_sum, Baseline.Gamma)
    result   = cs.chi_no_dens(
        b,
        densities["C_rho_p"], densities["C_rho_p"],
        Baseline.Gamma, Baseline.Gamma,
    )

    assert np.isclose(expected, result)


# ---------------------------------------------------------------------------
# Cross section  (matter, MOL and OLA)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sigma_R_matter_mol(cs, densities, matter_profile_300):
    rho_t = densities["C_rho_p"] + densities["C_rho_n"]
    rho_p = densities["Ca_rho_p"] + densities["Ca_rho_n"]

    expected = Baseline.sigma_R(rho_t, rho_p=rho_p, Gamma=Baseline.Gamma, Model="MOL")
    result   = cs.sigma_R_matter(rho_t, rho_p, matter_profile_300.Gamma, Model="MOL")

    assert np.isclose(expected, result)


@pytest.mark.slow
def test_sigma_R_matter_ola(cs, densities, matter_profile_300):
    rho_t = densities["C_rho_p"] + densities["C_rho_n"]
    rho_p = densities["Ca_rho_p"] + densities["Ca_rho_n"]

    expected = Baseline.sigma_R(rho_t, rho_p=rho_p, Gamma=Baseline.Gamma, Model="OLA")
    result   = cs.sigma_R_matter(rho_t, rho_p, matter_profile_300.Gamma, Model="OLA")

    assert np.isclose(expected, result)


# ---------------------------------------------------------------------------
# Proton-neutron eikonal phases and cross sections  (np profile, E=200)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def np_profile_200():
    E = 200
    Baseline.profile_funct_param(E, interaction_type="np")
    return ProfileFunction(model_type="np", E=E)


def test_chi_mol_micro(cs, densities):
    b = 3
    expected = Baseline.chi_mol_micro(
        b,
        densities["C_rho_p"], densities["C_rho_n"],
        densities["Ca_rho_p"], densities["Ca_rho_n"],
        Baseline.Gammap, Baseline.Gamman, Baseline.Gammap,
    )
    result = cs.chi_mol_micro(
        b,
        densities["C_rho_p"], densities["C_rho_n"],
        densities["Ca_rho_p"], densities["Ca_rho_n"],
        Baseline.Gammap, Baseline.Gamman, Baseline.Gammap,
    )
    assert np.isclose(expected, result)


def test_chi_ola_micro(cs, densities):
    b = 3
    expected = Baseline.chi_ola_micro(
        b,
        densities["C_rho_p"], densities["C_rho_n"],
        densities["Ca_rho_p"], densities["Ca_rho_n"],
        Baseline.Gammap, Baseline.Gamman, Baseline.Gammap,
    )
    result = cs.chi_ola_micro(
        b,
        densities["C_rho_p"], densities["C_rho_n"],
        densities["Ca_rho_p"], densities["Ca_rho_n"],
        Baseline.Gammap, Baseline.Gamman, Baseline.Gammap,
    )
    assert np.isclose(expected, result)


@pytest.mark.slow
def test_sigma_R_pn_ola(cs, densities):
    expected = Baseline.sigma_R_micro(
        densities["C_rho_p"], densities["C_rho_n"],
        densities["Ca_rho_p"], densities["Ca_rho_n"],
        Baseline.Gammap, Baseline.Gamman, Baseline.Gammap,
        Model="OLA",
    )
    result = cs.sigma_R_pn(
        densities["C_rho_p"], densities["C_rho_n"],
        densities["Ca_rho_p"], densities["Ca_rho_n"],
        Baseline.Gammap, Baseline.Gamman, Baseline.Gammap,
        Model="OLA",
    )
    assert np.isclose(expected, result)


@pytest.mark.slow
def test_sigma_R_pn_mol(cs, densities):
    expected = Baseline.sigma_R_micro(
        densities["C_rho_p"], densities["C_rho_n"],
        densities["Ca_rho_p"], densities["Ca_rho_n"],
        Baseline.Gammap, Baseline.Gamman, Baseline.Gammap,
        Model="MOL",
    )
    result = cs.sigma_R_pn(
        densities["C_rho_p"], densities["C_rho_n"],
        densities["Ca_rho_p"], densities["Ca_rho_n"],
        Baseline.Gammap, Baseline.Gamman, Baseline.Gammap,
        Model="MOL",
    )
    assert np.isclose(expected, result)


@pytest.mark.slow
def test_sigma_R_pn_ola_nucleon(cs, densities, np_profile_200):
    E = 200
    Baseline.profile_funct_param(E, interaction_type="np")

    expected = Baseline.sigma_R(
        densities["C_rho_p"], Gamma=Baseline.Gamma, Model="OLA p-n"
    )
    result = cs.sigma_R_pn(
        densities["C_rho_p"], densities["C_rho_p"],
        Gamma_pp=np_profile_200.Gamma_pp,
        Gamma_pn=np_profile_200.Gamma_pn,
        Model="OLA p-n",
    )
    assert np.isclose(expected, result)


# ---------------------------------------------------------------------------
# Density rms
# ---------------------------------------------------------------------------

def test_density_rms(densities):
    A, Z = 42, 20
    Ca_r_mesh = densities["Ca_r_mesh"]
    Ca_p      = densities["Ca_p"]
    Ca_n      = densities["Ca_n"]

    dens     = Density()
    expected = [3.3527555195621557, 3.3527555195621557, 0.10000080309561588]
    result   = [
        dens.rms(Ca_r_mesh, Ca_p + Ca_n, A),
        dens.rms(Ca_r_mesh, Ca_p + Ca_n, A),
        -dens.rms(Ca_r_mesh, Ca_p, Z) + dens.rms(Ca_r_mesh, Ca_n, A - Z),
    ]
    assert np.isclose(expected, result).all()
