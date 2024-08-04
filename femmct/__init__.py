import dolfin

from . import aux
from .stokessolver import StokesSolver, dirichlet_u, p_inlet, p_outlet
from .integralwm import IntegralWhiteMetznerModel

dolfin.parameters['ghost_mode'] = 'shared_facet'
dolfin.parameters['form_compiler']['optimize'] = True
dolfin.parameters['form_compiler']['quadrature_degree'] = 4
