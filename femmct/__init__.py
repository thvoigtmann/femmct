import dolfin

from . import aux
from .stokessolver import StokesSolver, dirichlet_u, p_inlet, p_outlet
from .integralwm import IntegralWhiteMetznerModel
from . import mct

if 'ghost_mode' in dolfin.parameters:
    dolfin.parameters['ghost_mode'] = 'shared_facet'
if 'form_compiler' in dolfin.parameters:
    dolfin.parameters['form_compiler']['optimize'] = True
    dolfin.parameters['form_compiler']['quadrature_degree'] = 4
