import importlib.resources

import numpy as np
from scipy.interpolate import CubicSpline


class ProfileFunction:
    """
    Interpolates profile function parameters for a given reaction energy and
    evaluates Gamma(b). Results are reliable for E = 40–1000 MeV.

    Parameters
    ----------
    model_type : str
        Type of interaction. One of ``"general"``, ``"np"``, or ``"matter"``.
    E : float
        Reaction energy [MeV].

    Attributes
    ----------
    alpha : float
        Real-to-imaginary ratio of the NN scattering amplitude.
    beta : float
        Finite range parameter [fm^2].
    sigma_n : float
        Nucleon-nucleon cross section [fm^2].
    Gamma : callable
        Profile function Gamma(b) for matter or general interactions.
    Gamma_pp, Gamma_pn : callable
        Proton-proton and proton-neutron profile functions (``"np"`` mode only).
    """

    def __init__(self, model_type: str = "general", E: float = 1):
        if model_type not in ("general", "np", "matter"):
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                "Choose 'general', 'np', or 'matter'."
            )

        self.alpha = 1
        self.beta = 1
        self.sigma_n = 1

        self.Gamma = lambda b: (
            (1 - 1j * self.alpha) / (4 * np.pi * self.beta)
            * self.sigma_n * np.exp(-b**2 / (2 * self.beta))
        )

        if model_type == "np":
            pkg = importlib.resources.files("ERICCA")
            with importlib.resources.as_file(
                pkg.joinpath("new_profile_funct_params.txt")
            ) as path:
                table = np.genfromtxt(path, unpack=True, skip_header=2)

            sigma_pp_fun = CubicSpline(table[0], table[1])
            alphapp_fun  = CubicSpline(table[0], table[2])
            betapp_fun   = CubicSpline(table[0], table[3])
            sigma_pn_fun = CubicSpline(table[0], table[4])
            alphapn_fun  = CubicSpline(table[0], table[5])
            betapn_fun   = CubicSpline(table[0], table[6])

            sigma_pp = sigma_pp_fun(E)
            alphapp  = alphapp_fun(E)
            betapp   = betapp_fun(E)
            sigma_pn = sigma_pn_fun(E)
            alphapn  = alphapn_fun(E)
            betapn   = betapn_fun(E)

            self.Gamma_pp = lambda b: (
                (1 - 1j * alphapp) / (4 * np.pi * betapp)
                * sigma_pp * np.exp(-b**2 / (2 * betapp))
            )
            self.Gamma_pn = lambda b: (
                (1 - 1j * alphapn) / (4 * np.pi * betapn)
                * sigma_pn * np.exp(-b**2 / (2 * betapn))
            )

        if model_type == "matter":
            pkg = importlib.resources.files("ERICCA")
            with importlib.resources.as_file(
                pkg.joinpath("profile_funct_param_matter.txt")
            ) as path:
                table = np.genfromtxt(path, unpack=True, skip_header=2)

            self.sigma_n = CubicSpline(table[0], table[1])(E)
            self.alpha   = CubicSpline(table[0], table[2])(E)
            self.beta    = CubicSpline(table[0], table[3])(E)
