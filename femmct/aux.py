import dolfin

class FunctionSpaces(object):
    def __init__ (self, mesh, **kwargs):
        """ kwargs: anything that passes onto dolfin.Functionspace
                    in particular constrained_domain
        """
        # velocity element
        FEu = dolfin.VectorElement('CG', mesh.ufl_cell(), 2)
        # pressure element
        FEp = dolfin.FiniteElement('DG', mesh.ufl_cell(), 0)
        # stress element
        FEtau = dolfin.TensorElement('DG', mesh.ufl_cell(), 0)
        # function spaces
        # TODO: do we need FEu, FEp, UP later on?
        kwargs2 = filtered_kwargs(kwargs, dolfin.FunctionSpace._init_from_ufl)
        self.UP = dolfin.FunctionSpace(mesh, dolfin.MixedElement([FEu,FEp]),
                                       **kwargs2)
        self.U = self.UP.sub(0).collapse()
        self.P = self.UP.sub(1).collapse()
        self.Tau = dolfin.FunctionSpace(mesh, FEtau, **kwargs2)
    def projectScalar(self,r):
        p = dolfin.TrialFunction(self.P)
        q = dolfin.TestFunction(self.P)
        LHS = dolfin.inner(p, q)*dolfin.dx
        RHS = dolfin.inner(r, q)*dolfin.dx
        solver = dolfin.LocalSolver(LHS, RHS)
        solver.factorize()
        p = dolfin.Function(self.P)
        solver.solve_local_rhs(p)
        return p
    def projectTensor(self,r):
        tau = dolfin.TrialFunction(self.Tau)
        sig = dolfin.TestFunction(self.Tau)
        LHS = dolfin.inner(tau, sig)*dolfin.dx
        RHS = dolfin.inner(r, sig)*dolfin.dx
        solver = dolfin.LocalSolver(LHS, RHS)
        solver.factorize()
        tau = dolfin.Function(self.Tau)
        solver.solve_local_rhs(tau)
        return tau

class DevNullFile(object):
    def __init__(self):
      return
    def __lshift__(self, data):
      return self

import inspect
def filtered_kwargs(kwargs, func, exclude=[]):
    """ take kwargs and make them suitable to be passed to the given func
        filters out any keyword arguments that are not found in the
        signature of func """
    kwargs2 = {k:v for k,v in kwargs.items() if not k in exclude}
    p = inspect.signature(func).parameters
    if 'kwargs' in p:
        return kwargs2
    kwargs2 = {k:v for k,v in kwargs2.items() if k in p}
    return kwargs2
