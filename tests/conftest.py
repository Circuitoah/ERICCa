"""
Fixtures for ERICCa test suite.

The density and cross-section objects are session-scoped because computing
dens_b_interpolator for all test nuclei is expensive (~10 s total).
"""
import numpy as np
import pytest

from ERICCA import CrossSection, Density


@pytest.fixture(scope="session")
def cs():
    return CrossSection()


@pytest.fixture(scope="session")
def densities(cs):
    """Pre-compute all nucleus densities used across tests."""
    dens = Density()
    r_mesh = np.linspace(0.01, 15, 100)

    dens.rho_m_2pt_fermi(6, 2.1, ra=0, rb=15, r_points=1000)
    C_p = dens.rho_m(r_mesh)
    dens.rho_m_2pt_fermi(6, 2.3, ra=0, rb=15, r_points=1000)
    C_n = dens.rho_m(r_mesh)

    Ca_r_mesh = np.linspace(0.01, 15, 100)
    dens.rho_m_2pt_fermi(20, 3.3, ra=0, rb=15, r_points=1000)
    Ca_p = dens.rho_m(Ca_r_mesh)
    dens.rho_m_2pt_fermi(22, 3.4, ra=0, rb=15, r_points=1000)
    Ca_n = dens.rho_m(Ca_r_mesh)

    return {
        "r_mesh":    r_mesh,
        "C_p":       C_p,
        "C_n":       C_n,
        "C_rho_p":   cs.dens_b_interpolator(r_mesh, C_p),
        "C_rho_n":   cs.dens_b_interpolator(r_mesh, C_n),
        "Ca_r_mesh": Ca_r_mesh,
        "Ca_p":      Ca_p,
        "Ca_n":      Ca_n,
        "Ca_rho_p":  cs.dens_b_interpolator(Ca_r_mesh, Ca_p),
        "Ca_rho_n":  cs.dens_b_interpolator(Ca_r_mesh, Ca_n),
    }
