
<p align="center">
<img src="ERICCa_Logo.png" alt="drawing" width="300" /> 
</p>

# Eikonal Reaction, density Input, Cross section Calculator (ERICCa)

A reaction code that calculates nucleus-nucleus reaction cross sections in the eikonal framework using nuclear densities as inputs.

## Quick start
```
 pip install ERICCA
```
The release versions of the package are hosted at https://github.com/Circuitoah/ERICCa

## Tutorials

Tutorials live in `/tutorials/`.

For the full tutorial use `tutorials/ERICCa_Tutorial.ipynb`.
For a condensed version use `tutorials/ERICCa_TLDR.ipynb`.

## Description

**ERICCa** (Eikonal Reaction, density Input, Cross section Calculator) is a Python package for calculating nucleus-nucleus reaction cross sections within the eikonal approximation framework. The code provides a flexible and robust approach to computing reaction observables by taking nuclear density distributions as direct inputs, allowing for accurate modeling of a wide range of nuclear reactions.

### Key Features

- **Eikonal Framework**: Employs the two-body eikonal framework for fast computation of reaction cross sections
- **Density-Based Approach**: Nuclear density distributions (matter or proton and neutron densities) as inputs
- **Accurate Integration**: Implements multi-dimensional numerical integration with adaptive mesh configurations
- **Validated Results**: Benchmarked against experimental reaction data

### Capabilities

ERICCa can calculate:
- Reaction cross sections using matter densities and proton and neutron densities as inputs and a profile function
- Automatic density generation based on rms matter radius and mass number
- Built-in profile functions for Energies 40-1000 MeV

## Contributing, Developing, and Testing

TBD

## citation
```latex
@software{Smith_AJ_ERICCa_2026,
author = {A. J. Smith, K. Godbey, C. Hebborn},
license = {MIT},
month = July,
title = {{ERICCa}},
url = {https://github.com/Circuitoah/ERICCa},
version = {v0.1.0},
year = {2026}
}
```
