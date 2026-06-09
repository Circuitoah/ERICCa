ERICCa Documentation
====================

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   api/index

ERICCa is a Python package for calculating nucleus-nucleus reaction cross
sections in the eikonal framework using optical limit approximation (OLA)
or modified optical limit (MOL) models.

Quick start
-----------

.. code-block:: python

   import numpy as np
   from ERICCA import CrossSection, Density, ProfileFunction

   # Build nuclear densities (two-point Fermi fit)
   dens = Density()
   r = np.linspace(0.01, 15, 100)
   dens.rho_m_2pt_fermi(12, 2.32)       # A=12, rms=2.32 fm
   rho = dens.rho_m(r)

   # Profile function from tabulated NN parameters at E=325 MeV
   pf = ProfileFunction(model_type="matter", E=325)

   # Calculate reaction cross section
   cs = CrossSection()
   rho_b = cs.dens_b_interpolator(r, rho)
   sigma = cs.sigma_R_matter(rho_b, rho_b, pf.Gamma, Model="OLA")
   print(f"sigma_R = {sigma:.1f} mb")

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
