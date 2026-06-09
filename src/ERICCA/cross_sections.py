from collections.abc import Callable
from typing import Optional

import numpy as np
from numpy.polynomial import legendre
from scipy.interpolate import Rbf


def mesh_creator(xa: float = 0, xb: float = 10, x_numpoints: int = 20):
    """Return Gauss-Legendre weights and mapped roots on [xa, xb]."""
    x_roots, x_weights = legendre.leggauss(x_numpoints)
    x_weights = x_weights * 0.5 * (xb - xa)
    x_mapped_roots = 0.5 * (x_roots + 1) * (xb - xa) + xa
    return x_weights, x_mapped_roots


class CrossSection:
    """
    Nucleus-nucleus reaction cross section calculator.

    Implements the eikonal framework using Gaussian-Legendre quadrature on
    configurable integration meshes. Mesh parameters may be adjusted via
    instance attributes followed by a call to :meth:`update_mesh`.

    Attributes
    ----------
    rmax : float
        Radial mesh upper bound [fm]. Default 25.
    bmax : float
        Impact-parameter mesh upper bound [fm]. Default 20.
    zmax : float
        Half-width of z mesh [fm]. Default 5.
    st_max : float
        Upper bound of s and t meshes [fm]. Default 15.
    r_numpoints : int
        Gauss-Legendre points on r mesh. Default 20.
    b_numpoints : int
        Gauss-Legendre points on b mesh. Default 35.
    z_numpoints : int
        Gauss-Legendre points on z mesh. Default 20.
    numpoints : int
        Gauss-Legendre points on s and t meshes. Default 30.
    numpoints_theta : int
        Gauss-Legendre points on angular meshes. Default 30.
    """

    def __init__(self) -> None:
        self.rmax = 25
        self.bmax = 20
        self.zmax = 5
        self.st_max = 15

        self.numpoints = 30
        self.numpoints_theta = 30
        self.r_numpoints = 20
        self.b_numpoints = 35
        self.z_numpoints = 20

        self.update_mesh()

    def update_mesh(self) -> None:
        """Rebuild all integration meshes from the current bound/point attributes."""
        self.ra, self.rb = 0, self.rmax
        self.ba, self.bb = 0, self.bmax
        self.za, self.zb = -self.zmax, self.zmax
        self.sa, self.sb = 0, self.st_max
        self.s_theta_a, self.s_theta_b = 0, 2 * np.pi
        self.ta, self.tb = 0, self.st_max
        self.t_theta_a, self.t_theta_b = 0, 2 * np.pi

        self.r_weights, self.r_mapped_roots = mesh_creator(self.ra, self.rb, self.r_numpoints)
        self.b_weights, self.b_mapped_roots = mesh_creator(self.ba, self.bb, self.b_numpoints)
        self.z_weights, self.z_mapped_roots = mesh_creator(self.za, self.zb, self.z_numpoints)
        self.s_weights, self.s_mapped_roots = mesh_creator(self.sa, self.sb, self.numpoints)
        self.s_theta_weights, self.s_theta_mapped_roots = mesh_creator(self.s_theta_a, self.s_theta_b, self.numpoints_theta)
        self.t_weights, self.t_mapped_roots = mesh_creator(self.ta, self.tb, self.numpoints)
        self.t_theta_weights, self.t_theta_mapped_roots = mesh_creator(self.t_theta_a, self.t_theta_b, self.numpoints_theta)

    # --- Vector operations ---------------------------------------------------

    def add_sub_vec_mag(
        self,
        a: float, theta_a: float,
        b: float, theta_b: float,
        c: float, theta_c: float,
    ) -> float:
        r"""Return the magnitude of :math:`\vec{a} + \vec{b} - \vec{c}`."""
        arg_x = a * np.cos(theta_a) + b * np.cos(theta_b) - c * np.cos(theta_c)
        arg_y = a * np.sin(theta_a) + b * np.sin(theta_b) - c * np.sin(theta_c)
        return np.sqrt(arg_x**2 + arg_y**2)

    # --- Eikonal phase helpers -----------------------------------------------

    def _chi_mol_half(
        self,
        b: float,
        rho_t: np.ndarray,
        rho_p: np.ndarray,
        Gamma: Callable,
    ) -> complex:
        """One symmetric half of the MOL eikonal phase (internal helper)."""

        def chi_t(b, s, theta_s, theta_t):
            t_integrand = lambda t: t * Gamma(self.add_sub_vec_mag(b, 0, s, theta_s, t, theta_t))
            func_values = t_integrand(self.t_mapped_roots[:, np.newaxis, np.newaxis, np.newaxis]) * rho_t[:, np.newaxis, np.newaxis, np.newaxis]
            return np.sum(self.t_weights[:, np.newaxis, np.newaxis, np.newaxis] * func_values, axis=0)

        def chi_theta_t(b, s, theta_s):
            func_values = chi_t(b, s, theta_s, self.t_theta_mapped_roots[:, np.newaxis, np.newaxis])
            return np.sum(self.t_theta_weights[:, np.newaxis, np.newaxis] * func_values, axis=0)

        def chi_s(b, theta_s):
            s_integrand = lambda s: s * (1 - np.exp(-chi_theta_t(b, s, theta_s)))
            func_values = s_integrand(self.s_mapped_roots[:, np.newaxis]) * rho_p[:, np.newaxis]
            return np.sum(self.s_weights[:, np.newaxis] * func_values, axis=0)

        func_values = 0.5j * chi_s(b, self.s_theta_mapped_roots)
        return np.sum(self.s_theta_weights * func_values, axis=0)

    def chi_mol(
        self,
        b: float,
        rho_t: np.ndarray,
        rho_p: np.ndarray,
        Gamma: Callable,
    ) -> complex:
        """
        Eikonal phase in the MOL model.

        Parameters
        ----------
        b : float
            Impact parameter [fm].
        rho_t : np.ndarray
            Target density on the t mesh [fm^-3].
        rho_p : np.ndarray
            Projectile density on the s mesh [fm^-3].
        Gamma : callable
            Profile function Gamma(b) [fm^-2].
        """
        return self._chi_mol_half(b, rho_t, rho_p, Gamma) + self._chi_mol_half(b, rho_p, rho_t, Gamma)

    def chi(
        self,
        b: float,
        rho_t: np.ndarray,
        rho_p: np.ndarray,
        Gamma: Callable,
    ) -> complex:
        """
        Eikonal phase in the OLA model.

        Parameters
        ----------
        b : float
            Impact parameter [fm].
        rho_t : np.ndarray
            Target density on the t mesh [fm^-3].
        rho_p : np.ndarray
            Projectile density on the s mesh [fm^-3].
        Gamma : callable
            Profile function Gamma(b) [fm^-2].
        """

        def chi_t(b, s, theta_s, theta_t):
            t_integrand = lambda t: 1j * s * t * Gamma(self.add_sub_vec_mag(b, 0, s, theta_s, t, theta_t))
            func_values = t_integrand(self.t_mapped_roots[:, np.newaxis, np.newaxis, np.newaxis]) * rho_t[:, np.newaxis, np.newaxis, np.newaxis]
            return np.sum(self.t_weights[:, np.newaxis, np.newaxis, np.newaxis] * func_values, axis=0)

        def chi_theta_t(b, s, theta_s):
            func_values = chi_t(b, s, theta_s, self.t_theta_mapped_roots[:, np.newaxis, np.newaxis])
            return np.sum(self.t_theta_weights[:, np.newaxis, np.newaxis] * func_values, axis=0)

        def chi_s(b, theta_s):
            func_values = chi_theta_t(b, s=self.s_mapped_roots[:, np.newaxis], theta_s=theta_s) * rho_p[:, np.newaxis]
            return np.sum(self.s_weights[:, np.newaxis] * func_values, axis=0)

        func_values = chi_s(b, self.s_theta_mapped_roots)
        return np.sum(self.s_theta_weights * func_values, axis=0)

    def chi_no_dens_term(
        self,
        b: float,
        rho: np.ndarray,
        Gamma: Callable,
    ) -> complex:
        """
        OLA eikonal phase for proton-nucleus scattering (single density term).

        Parameters
        ----------
        b : float
            Impact parameter [fm].
        rho : np.ndarray
            Nuclear density on the s mesh [fm^-3].
        Gamma : callable
            Profile function Gamma(b) [fm^-2].
        """

        def chi_s(b, theta_s):
            s_integrand = lambda s: 1j * s * Gamma(self.add_sub_vec_mag(b, 0, s, theta_s, 0, 0))
            func_values = s_integrand(self.s_mapped_roots[:, np.newaxis]) * rho[:, np.newaxis]
            return np.sum(self.s_weights[:, np.newaxis] * func_values, axis=0)

        func_values = chi_s(b, self.s_theta_mapped_roots)
        return np.sum(self.s_theta_weights * func_values, axis=0)

    def chi_no_dens(
        self,
        b: float,
        rho_p: np.ndarray,
        rho_n: np.ndarray,
        Gamma_pp: Callable,
        Gamma_pn: Callable,
    ) -> complex:
        """OLA eikonal phase for proton-nucleus scattering with p/n separation."""
        return self.chi_no_dens_term(b, rho_p, Gamma_pp) + self.chi_no_dens_term(b, rho_n, Gamma_pn)

    # --- Composite eikonal phases (p/n decomposition) ------------------------

    def chi_mol_micro(
        self,
        b: float,
        rho_t_p: np.ndarray, rho_t_n: np.ndarray,
        rho_p_p: np.ndarray, rho_p_n: np.ndarray,
        Gamma_pp: Callable, Gamma_pn: Callable, Gamma_nn: Callable,
    ) -> complex:
        """
        MOL eikonal phase with proton/neutron density decomposition.

        Parameters
        ----------
        b : float
            Impact parameter [fm].
        rho_t_p, rho_t_n : np.ndarray
            Target proton and neutron densities on the t mesh [fm^-3].
        rho_p_p, rho_p_n : np.ndarray
            Projectile proton and neutron densities on the s mesh [fm^-3].
        Gamma_pp, Gamma_pn, Gamma_nn : callable
            pp, pn, and nn profile functions [fm^-2].
        """
        chi_pp = self.chi_mol(b, rho_t_p, rho_p_p, Gamma_pp)
        chi_pn = self.chi_mol(b, rho_t_p, rho_p_n, Gamma_pn) + self.chi_mol(b, rho_t_n, rho_p_p, Gamma_pn)
        chi_nn = self.chi_mol(b, rho_t_n, rho_p_n, Gamma_nn)
        return chi_pp + chi_pn + chi_nn

    def chi_ola_micro(
        self,
        b: float,
        rho_t_p: np.ndarray, rho_t_n: np.ndarray,
        rho_p_p: np.ndarray, rho_p_n: np.ndarray,
        Gamma_pp: Callable, Gamma_pn: Callable, Gamma_nn: Callable,
    ) -> complex:
        """
        OLA eikonal phase with proton/neutron density decomposition.

        Parameters
        ----------
        b : float
            Impact parameter [fm].
        rho_t_p, rho_t_n : np.ndarray
            Target proton and neutron densities on the t mesh [fm^-3].
        rho_p_p, rho_p_n : np.ndarray
            Projectile proton and neutron densities on the s mesh [fm^-3].
        Gamma_pp, Gamma_pn, Gamma_nn : callable
            pp, pn, and nn profile functions [fm^-2].
        """
        chi_pp = self.chi(b, rho_t_p, rho_p_p, Gamma_pp)
        chi_pn = self.chi(b, rho_t_p, rho_p_n, Gamma_pn) + self.chi(b, rho_t_n, rho_p_p, Gamma_pn)
        chi_nn = self.chi(b, rho_t_n, rho_p_n, Gamma_nn)
        return chi_pp + chi_pn + chi_nn

    # --- Cross section calculators -------------------------------------------

    def sigma_R_matter(
        self,
        rho_t: np.ndarray,
        rho_p: np.ndarray,
        Gamma: Callable,
        Model: str = "OLA",
    ) -> float:
        """
        Reaction cross section using matter densities [mb].

        Parameters
        ----------
        rho_t : np.ndarray
            Target density on the t mesh [fm^-3].
        rho_p : np.ndarray
            Projectile density on the s mesh [fm^-3].
        Gamma : callable
            Profile function Gamma(b) [fm^-2].
        Model : str
            Calculation model: ``"OLA"`` (default) or ``"MOL"``.
        """
        if Model == "MOL":
            sigma_R_int = lambda b: 2 * np.pi * b * (1 - np.exp(-2 * self.chi_mol(b, rho_t, rho_p, Gamma).imag))
        else:
            sigma_R_int = lambda b: 2 * np.pi * b * (1 - np.exp(-2 * self.chi(b, rho_t, rho_p, Gamma).imag))

        func_values = np.array([sigma_R_int(b) for b in self.b_mapped_roots.tolist()])
        return np.sum(self.b_weights * func_values) * 10

    def sigma_R_pn(
        self,
        rho_t_p: np.ndarray,
        rho_t_n: np.ndarray,
        rho_p_p: Optional[np.ndarray] = None,
        rho_p_n: Optional[np.ndarray] = None,
        Gamma_pp: Callable = lambda b: np.exp(-b),
        Gamma_pn: Callable = lambda b: np.exp(-b),
        Gamma_nn: Callable = lambda b: np.exp(-b),
        Model: str = "OLA",
    ) -> float:
        """
        Reaction cross section using proton/neutron densities [mb].

        Parameters
        ----------
        rho_t_p, rho_t_n : np.ndarray
            Target proton and neutron densities on the t mesh [fm^-3].
        rho_p_p, rho_p_n : np.ndarray, optional
            Projectile proton and neutron densities (required for ``"OLA"`` and
            ``"MOL"``; not used for ``"OLA p-n"``).
        Gamma_pp, Gamma_pn, Gamma_nn : callable
            pp, pn, and nn profile functions [fm^-2].
        Model : str
            ``"OLA"`` (default), ``"MOL"``, or ``"OLA p-n"``.
        """
        if Model in ("MOL", "OLA") and (rho_p_p is None or rho_p_n is None):
            raise ValueError(
                f"Model='{Model}' requires rho_p_p and rho_p_n. "
                "Use Model='OLA p-n' for proton-nucleus scattering."
            )

        if Model == "MOL":
            sigma_R_int = lambda b: 2 * np.pi * b * (1 - np.exp(-2 * self.chi_mol_micro(b, rho_t_p, rho_t_n, rho_p_p, rho_p_n, Gamma_pp, Gamma_pn, Gamma_nn).imag))
        elif Model == "OLA p-n":
            sigma_R_int = lambda b: 2 * np.pi * b * (1 - np.exp(-2 * self.chi_no_dens(b, rho_t_p, rho_t_n, Gamma_pp, Gamma_pn).imag))
        else:
            sigma_R_int = lambda b: 2 * np.pi * b * (1 - np.exp(-2 * self.chi_ola_micro(b, rho_t_p, rho_t_n, rho_p_p, rho_p_n, Gamma_pp, Gamma_pn, Gamma_nn).imag))

        func_values = np.array([sigma_R_int(b) for b in self.b_mapped_roots.tolist()])
        return np.sum(self.b_weights * func_values) * 10

    # --- Density interpolation -----------------------------------------------

    def dens_b_interpolator(
        self,
        array_r: np.ndarray,
        array_rho: np.ndarray,
    ) -> np.ndarray:
        """
        Project a spherical density rho(r) onto the impact-parameter mesh.

        Integrates rho(sqrt(b^2 + z^2)) along z for each point on the t mesh,
        returning an array suitable for use in the eikonal phase methods.

        Parameters
        ----------
        array_r : np.ndarray
            Radial mesh in spherical coordinates [fm].
        array_rho : np.ndarray
            Density values at each radius [fm^-3]. Must be same length as array_r.
        """

        def rho_funct(r):
            if r <= array_r[-1]:
                return Rbf(array_r, array_rho)(r)
            return 0

        def rho_z_funct(b):
            integrand = lambda z: rho_funct(np.sqrt(b**2 + z**2))
            funct_values = np.array([integrand(z) for z in self.z_mapped_roots.tolist()])
            return np.sum(self.z_weights * funct_values)

        return np.array([rho_z_funct(b) for b in self.t_mapped_roots.tolist()])
