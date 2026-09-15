"""
Regression tests for ERICCa's public API.

Each test calls the current class-based API and asserts the result against a
fixed numerical reference value. Those reference values were originally
produced by comparing this API against ``ERICCA.baseline`` (the legacy
procedural reference implementation); that module has since been removed
from the package, and its outputs have been frozen here as literals so this
file no longer needs it. Tests marked ``slow`` involve full sigma_R
computations and take a few seconds each.
"""
import numpy as np
import pytest

from ERICCA import CrossSection, Density, ProfileFunction


# ---------------------------------------------------------------------------
# Vector operations
# ---------------------------------------------------------------------------

def test_add_sub_vec_mag(cs):
    a, theta_a = 2.0, np.pi / 3.0
    b, theta_b = 3.0, np.pi / 4.0
    c, theta_c = 4.0, np.pi / 6.0

    expected = 1.8848033382770102
    result   = cs.add_sub_vec_mag(a, theta_a, b, theta_b, c, theta_c)

    assert np.isclose(expected, result), f"Expected {expected}, got {result}"


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------

def test_rho_m():
    mesh = np.linspace(0.01, 5, 30)

    dens = Density()
    dens.C_m_p   = 1
    dens.a_m_p   = 1
    dens.rho_0_p = 1

    expected = 0.01798620996209156

    with np.errstate(over="ignore"):
        result = dens.rho_m(mesh)[-1]

    assert np.isclose(expected, result)


# ---------------------------------------------------------------------------
# Profile function
# ---------------------------------------------------------------------------

def test_gamma_general():
    b = 3.0

    pf = ProfileFunction()
    pf.alpha   = 1.808
    pf.beta    = 0.268
    pf.sigma_n = 3.16

    expected = 4.7872304350836485e-08 - 8.655312626631236e-08j

    assert np.isclose(expected, pf.Gamma(b))


def test_gamma_matter():
    E = 325
    pf = ProfileFunction(model_type="matter", E=E)

    expected = 2.815156783683478e-26 - 8.586228190234607e-27j

    assert np.isclose(expected, pf.Gamma(3.0))


def test_gamma_np():
    E = 300
    pf = ProfileFunction(model_type="np", E=E)

    expected = 2.7104334286135547e-24 - 8.836071405087366e-25j
    updated_gamma = pf.Gamma_pp(3.0) + pf.Gamma_pn(3.0)

    assert np.isclose(expected, updated_gamma)


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

    expected = 0.823645221836711
    result   = cs.dens_b_interpolator(r_mesh, rho)[0]

    assert np.isclose(expected, result)


# ---------------------------------------------------------------------------
# Eikonal phase functions  (matter profile, E=300)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def matter_profile_300():
    return ProfileFunction(model_type="matter", E=300)


def test_chi_mol_half(cs, densities, matter_profile_300):
    b     = 3
    rho_t = densities["C_rho_p"] + densities["C_rho_n"]
    rho_p = densities["Ca_rho_p"] + densities["Ca_rho_n"]

    expected = 0.8058407899175231 + 3.4146803318500636j
    result   = cs._chi_mol_half(b, rho_t, rho_p, matter_profile_300.Gamma)

    assert np.isclose(expected, result)


def test_chi_mol(cs, densities, matter_profile_300):
    b     = 3
    rho_t = densities["C_rho_p"] + densities["C_rho_n"]
    rho_p = densities["Ca_rho_p"] + densities["Ca_rho_n"]

    expected = 1.3671025309983922 + 6.341229964730985j
    result   = cs.chi_mol(b, rho_t, rho_p, matter_profile_300.Gamma)

    assert np.isclose(expected, result)


def test_chi_ola(cs, densities, matter_profile_300):
    b     = 3
    rho_t = densities["C_rho_p"] + densities["C_rho_n"]
    rho_p = densities["Ca_rho_p"] + densities["Ca_rho_n"]

    expected = 3.582295922650445 + 9.769112429503105j
    result   = cs.chi(b, rho_t, rho_p, matter_profile_300.Gamma)

    assert np.isclose(expected, result)


def test_chi_nN_pn(cs, densities, matter_profile_300):
    b       = 3
    rho_sum = densities["C_rho_p"] + densities["C_rho_p"]

    expected = 0.022738818164091925 + 0.06200997236283244j
    result   = cs.chi_nN_pn(
        b,
        densities["C_rho_p"], densities["C_rho_p"],
        matter_profile_300.Gamma, matter_profile_300.Gamma,
    )

    assert np.isclose(expected, result)


# ---------------------------------------------------------------------------
# Cross section  (matter, MOL and OLA)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sigma_R_matter_mol(cs, densities, matter_profile_300):
    rho_t = densities["C_rho_p"] + densities["C_rho_n"]
    rho_p = densities["Ca_rho_p"] + densities["Ca_rho_n"]

    expected = 1461.712936462014
    result   = cs.sigma_R_matter(rho_t, rho_p, matter_profile_300.Gamma, Model="MOL")

    assert np.isclose(expected, result)


@pytest.mark.slow
def test_sigma_R_matter_ola(cs, densities, matter_profile_300):
    rho_t = densities["C_rho_p"] + densities["C_rho_n"]
    rho_p = densities["Ca_rho_p"] + densities["Ca_rho_n"]

    expected = 1516.4088622166057
    result   = cs.sigma_R_matter(rho_t, rho_p, matter_profile_300.Gamma, Model="OLA")

    assert np.isclose(expected, result)


# ---------------------------------------------------------------------------
# Proton-neutron eikonal phases and cross sections  (np profile, E=300)
# ---------------------------------------------------------------------------
#
# These reference values match a proton-neutron profile evaluated at
# E=300 MeV (the "pn_profile_300" fixture below), which is the energy that
# was actually in effect for these particular tests in the original
# baseline-comparison suite.

@pytest.fixture(scope="module")
def pn_profile_300():
    return ProfileFunction(model_type="np", E=300)


@pytest.fixture(scope="module")
def np_profile_200():
    return ProfileFunction(model_type="np", E=200)


def test_chi_mol_micro(cs, densities, pn_profile_300):
    b = 3
    expected = 2.5978152405391777 + 7.756004281713967j
    result = cs.chi_mol_micro(
        b,
        densities["C_rho_p"], densities["C_rho_n"],
        densities["Ca_rho_p"], densities["Ca_rho_n"],
        pn_profile_300.Gamma_pp, pn_profile_300.Gamma_pn, pn_profile_300.Gamma_pp,
    )
    assert np.isclose(expected, result)


def test_chi_ola_micro(cs, densities, pn_profile_300):
    b = 3
    expected = 4.312286848585706 + 9.701512429653516j
    result = cs.chi_ola_micro(
        b,
        densities["C_rho_p"], densities["C_rho_n"],
        densities["Ca_rho_p"], densities["Ca_rho_n"],
        pn_profile_300.Gamma_pp, pn_profile_300.Gamma_pn, pn_profile_300.Gamma_pp,
    )
    assert np.isclose(expected, result)


@pytest.mark.slow
def test_sigma_R_pn_ola(cs, densities, pn_profile_300):
    expected = 1512.756688565642
    result = cs.sigma_R_pn(
        densities["C_rho_p"], densities["C_rho_n"],
        densities["Ca_rho_p"], densities["Ca_rho_n"],
        pn_profile_300.Gamma_pp, pn_profile_300.Gamma_pn, pn_profile_300.Gamma_pp,
        Model="OLA",
    )
    assert np.isclose(expected, result)


@pytest.mark.slow
def test_sigma_R_pn_mol(cs, densities, pn_profile_300):
    expected = 1485.0240855004954
    result = cs.sigma_R_pn(
        densities["C_rho_p"], densities["C_rho_n"],
        densities["Ca_rho_p"], densities["Ca_rho_n"],
        pn_profile_300.Gamma_pp, pn_profile_300.Gamma_pn, pn_profile_300.Gamma_pp,
        Model="MOL",
    )
    assert np.isclose(expected, result)


@pytest.mark.slow
def test_sigma_R_pn_ola_nucleon(cs, densities, np_profile_200):
    expected = 191.16942115618656
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
