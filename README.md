# FEM-MCT
Finite-Element Method solver combined with the mode-coupling theory of the glass transition, to predict flow properties of viscoelastic shear-thinning / yield-stress fluids using a microscopically derived constitutive equation.

For application of the code, see [Steinhäuser, Treskatis, Turek, and Voigtmann, arXiv:2307.12764 (2023)](https://arxiv.org/abs/2307.12764).

## Installation

Create a python virtual environment and activate it.

Then, install fenics/dolfin. Follow the instructions on https://fenics.readthedocs.io/en/latest/installation.html (we use the development version). Clone the git repositories, and pip install them inside the activated python environment. If you apt install libdolfin-dev, you can skip the cmake of dolfin.

To build mshr: apt install libgmp-dev libmpfr-dev (mshr is no longer in debian testing).

The requirements.txt file here was created using
```
  pip list --format=freeze >requirements.txt
```
instead of pip freeze.
