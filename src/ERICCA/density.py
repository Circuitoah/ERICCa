from typing import Optional

import numpy as np
from scipy.optimize import minimize


class Density:
    """
    Two-parameter Fermi density model for nuclear matter distributions.

    Attributes
    ----------
    C_m_p : float
        Half-density radius [fm].
    a_m_p : float
        Diffuseness [fm].
    rho_0_p : float
        Saturation density [fm^-3].
    """

    def __init__(self) -> None:
        self.C_m_p = 0.0
        self.a_m_p = 0.0
        self.rho_0_p = 0.0

    def rho_m(self, r: np.ndarray) -> np.ndarray:
        """
        Two-point Fermi density at radius r.

        Parameters
        ----------
        r : float or np.ndarray
            Radius in spherical coordinates [fm].

        Returns
        -------
        np.ndarray
            Density [fm^-3].
        """
        return self.rho_0_p / (1 + np.exp((r - self.C_m_p) / self.a_m_p))

    def rms(self, r_mesh: np.ndarray, rho_mesh: np.ndarray, A: float) -> float:
        """
        Root-mean-square matter radius.

        Parameters
        ----------
        r_mesh : np.ndarray
            Uniform radial mesh [fm].
        rho_mesh : np.ndarray
            Matter density at each radius [fm^-3].
        A : float
            Mass number.

        Returns
        -------
        float
            RMS matter radius [fm].
        """
        dr = r_mesh[1] - r_mesh[0]
        return np.sqrt(
            (4 * np.pi / A)
            * np.sum(np.asarray(rho_mesh) * np.asarray(r_mesh) ** 4)
            * dr
        )

    def rho_m_2pt_fermi(
        self,
        A: float,
        rms_measured: float,
        ra: float = 0,
        rb: float = 30,
        r_points: int = 1000,
    ) -> None:
        """
        Fit Fermi parameters to reproduce a measured RMS matter radius.

        Fits ``C_m_p``, ``a_m_p``, and ``rho_0_p`` subject to constraints on the
        mass number and saturation density rho(0) ≈ 0.176 fm^-3.

        Parameters
        ----------
        A : float
            Mass number.
        rms_measured : float
            Measured RMS matter radius [fm].
        ra : float
            Lower integration bound [fm]. Default 0.
        rb : float
            Upper integration bound [fm]. Default 30.
        r_points : int
            Number of mesh points. Default 1000.
        """
        r_mesh = np.linspace(ra, rb, r_points)
        dr = r_mesh[1] - r_mesh[0]

        def _norm(rho_func):
            return np.sum(4 * np.pi * r_mesh**2 * rho_func(r_mesh)) * dr

        def chi2_rms(params):
            self.C_m_p, self.a_m_p, self.rho_0_p = params
            return (self.rms(r_mesh, self.rho_m(r_mesh), A) - rms_measured) ** 2

        def con_A(params):
            self.C_m_p, self.a_m_p, self.rho_0_p = params
            return abs(_norm(self.rho_m) - A)

        def con_sat(params):
            self.C_m_p, self.a_m_p, self.rho_0_p = params
            return abs(self.rho_m(0) - 0.176)

        constraints = [
            {"type": "eq", "fun": con_A},
            {"type": "eq", "fun": con_sat},
        ]

        result = minimize(chi2_rms, [4.1, 0.5, 0.176], constraints=constraints)
        self.C_m_p, self.a_m_p, self.rho_0_p = result.x
