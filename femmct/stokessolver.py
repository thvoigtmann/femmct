import numpy as np
import dolfin as fe
from dolfin import div, outer, inner, sym, grad, dot, jump, avg, dS
from .aux import FunctionSpaces, DevNullFile, filtered_kwargs

def dirichlet_u (constant):
    return lambda up, boundaries, idx: fe.DirichletBC(up.sub(0), fe.Constant(constant), boundaries, idx)
def p_inlet (dp):
    return lambda up, boundaries, idx: fe.Constant(dp)
def p_outlet (dp):
    return lambda up, boundaries, idx: fe.Constant(dp)

# rho: density, muS: solvent viscosity
def default_parameters ():
    return {
      'rho': 1.0,
      'muS': 1.0,
    }

TWO = fe.Constant(2.)


class StokesSolver(object):
    def __init__(self, mesh, **kwargs):
        self.I = fe.Constant(((1., 0.), (0., 1.)))

        self.fn = FunctionSpaces(mesh, **kwargs)

        kwargs2 = filtered_kwargs(kwargs, fe.Measure.__call__)
        self.dx = fe.Measure('dx', mesh)
        self.ds = fe.Measure('ds', mesh)(**kwargs2)

        self.n = fe.FacetNormal(mesh)

        self.BClist = []
        self.BCs = {}
        self.dps = {}
        if 'boundary_conditions' in kwargs:
            self.apply_bc (kwargs['boundary_conditions'],
                           kwargs['subdomain_data'])

        self.callback = lambda obj, step: None
        self.post_step_callback = lambda obj, t: None

        self.velocityFile = DevNullFile()
        self.pressureFile = DevNullFile()
        self.stressFile = DevNullFile()
        self.strainrateFile = DevNullFile()
        self.polystressFile = DevNullFile()
        self.N1File = DevNullFile()


    def apply_bc (self, boundary_conditions, subdomain_data = None):
        for mark_idx,bcfunc in boundary_conditions.items():
            bc = bcfunc(self.fn.UP, subdomain_data, mark_idx)
            if isinstance(bc, fe.Constant):
                self.dps[mark_idx] = bc
            else:
                self.BCs[mark_idx] = bc
        self.BClist = list(self.BCs.values())

    # initialize makes new empty arrays for Finger and G tensors
    # ie clears all history-related information
    # Nt: time steps, Na: steps per age block, Nb: age blocks
    def initialize (self, model = None, T = 50., Nt = 500, Na = 16, Nb = 6,
        parameters = default_parameters()):
        self.parameters = parameters
        self.model = model

        self.Nt = Nt
        self.Na = Na
        self.Nb = Nb

        self.Nt, self.Na, self.Nb = self.model.adjust(Nt, Na, Nb)

        # sanitize input: maximum age is (2**Nb - 1)*Na*dt
        # anything that exceeds Nt will never be needed
        Nbmax = int(np.ceil(np.log2(self.Nt/self.Na + 1.)))
        if self.Nb > Nbmax:
            print('(INFO) Using only', Nbmax, 'instead of', self.Nb,
                  'blocks of deformation fields')
            self.Nb = Nbmax

        # total number of deformation fields to store
        self.Nh = self.Na * self.Nb

        # time step
        self.dt = T/self.Nt

        self.Bs = [] # array of Finger tensor fields of increasing age
        # Bs[0] = youngest Finger tensor with age a = dt
        # Bs[Nh-1] = oldest Finger tensor with age a = (2**Nb - 1)*Na*dt
        # initialize Finger tensors:
        for i in range(self.Nh):
            self.Bs.append(fe.Function(self.fn.Tau, name='Finger tensor'))
            self.Bs[i].assign(fe.interpolate(self.I, self.fn.Tau))
        self.model.initialize(self)
        # for reference: store age a = 0 Finger tensor and shear modulus
        # nearest younger Finger tensor of age a = 0 is Id
        # nearest younger shear modulus of age a = 0 is GInf
        self.Bs0 = self.I

    # loop should initialize all instantaneous functions etc afresh
    # kwargs can contain boolean settings for stress, strainrate, polystress,
    # or N1, True to calculate them (and possibly output, depending on whether
    # the file has been created), they all default to False except stress and
    # strainrate that default to true
    def loop (self, stress=True, strainrate=True, polystress=False, N1=False,
              normalize_pressure=True, **kwargs):
        n, dx, ds = self.n, self.dx, self.ds
        Bs = self.Bs

        # trial and test functions
        u_, p_ = fe.TrialFunctions(self.fn.UP)
        v, q = fe.TestFunctions(self.fn.UP)
        # functions to solve for
        u, p = fe.Function(self.fn.U), fe.Function(self.fn.P)

        up = fe.Function(self.fn.UP)
        tau = fe.Function(self.fn.Tau)
        u0 = fe.Function(self.fn.U)

        rho = fe.Constant(self.parameters['rho'])
        two_muS = fe.Constant(2.*self.parameters['muS'])

        t = 0.
        self.velocity = fe.Function(self.fn.U, name = 'velocity')
        self.pressure = fe.Function(self.fn.P, name = 'pressure')
        self.velocityFile << (self.velocity, t)
        self.pressureFile << (self.pressure, t)

        if stress:
            self.stress = fe.Function(self.fn.Tau, name = 'total stress')
            self.stressFile << (self.stress, t)
        else:
            self.stress = None
        if strainrate:
            self.strainrate = fe.Function(self.fn.Tau, name = 'strain rate')
            self.strainrateFile << (self.strainrate, t)
        else:
            self.strainrate = None
        if polystress:
            self.polystress = fe.Function(self.fn.Tau, name = 'polystress')
            self.polystressFile << (self.polystress, t)
        else:
            self.polystress = None
        if N1:
            self.N1 = fe.Function(self.fn.P, name = 'N1')
            self.N1File << (self.N1, t)
        else:
            self.N1 = None

        # to solve the evolution equation for the Finger tensors:
        B_ = fe.TrialFunction(self.fn.Tau)
        C = fe.TestFunction(self.fn.Tau)
        deformationsolver = fe.LUSolver()

        invdt = fe.Constant(1./self.dt)

        # time stepping
        for k in range(1, self.Nt + 1):
            print('Time Step ', k, ' out of ', self.Nt)
            t += self.dt

            un = dot(u, self.n)
            unPos = (un + abs(un))/TWO
            unNeg = (un - abs(un))/TWO

            # evolution equation for the Finger tensors and shear moduli
            LHSFT = invdt * inner(B_,C)*dx
            LHSFT -= inner(B_, div(outer(C, u)))*dx
            LHSFT += inner(unPos('+')*B_('+') + unNeg('+')*B_('-'), jump(C))*dS
            LHSFT += inner(un*B_, C)*ds
            LHSFT -= inner(grad(u)*B_ + B_*grad(u).T, C)*dx
            FTmat = fe.assemble(LHSFT)
            deformationsolver.set_operator(FTmat)

            for l in range(self.Nb - 1, -1, -1):
                # iteration over history blocks in reverse order
                # l = 0: da = dt (finest resolution)
                # l = 1: da = 2*dt
                # l = 2: da = 4*dt
                # ...
                da = 2**l*self.dt
                invda = fe.Constant(1./da)
                for m in range(self.Na - 1, -1, -1):
                    j = l*self.Na + m
                    RHSFT = invdt*inner(Bs[j], C)*dx
                    if j == 0:
                        # nearest younger Finger tensor of age a = 0 is Id
                        # nearest younger shear modulus of age a = 0 is GInf
                        RHSFT -= invda*inner(Bs[0] - self.Bs0, C)*dx
                    else:
                        RHSFT -= invda*inner(Bs[j] - Bs[j-1], C)*dx
                    FTvec = fe.assemble(RHSFT)
                    deformationsolver.solve (Bs[j].vector(), FTvec)

            self.model.step(self, k, u, un, unPos, unNeg, tau)

            # callback, for example to turn on/off pressure gradient
            self.callback(self, k)

            # solve Stokes problem
            LHSNS = invdt*rho*inner(u_, v)*dx \
                  + inner(two_muS*sym(grad(u_)), sym(grad(v)))*dx \
                  - p_*div(v)*dx - q*div(u_)*dx
            RHSNS = invdt*rho*inner(u0, v)*dx \
                  - inner(tau, sym(grad(v)))*dx \
                  + dot(avg(tau)*n('+'), jump(v))*dS + dot(tau*n, v)*ds
            # implement pressure boundary conditions for inlet/outlet:
            for idx,dp in self.dps.items():
                RHSNS -= dot(dp*n, v)*ds(idx)

            fe.solve(LHSNS == RHSNS, up, self.BClist, **kwargs)

            fe.assign(u, up.sub(0))
            fe.assign(p, up.sub(1))

            # export the current time step
            fe.assign(self.velocity, u)
            if normalize_pressure:
                fe.assign(self.pressure, self.fn.projectScalar(
                    p - fe.assemble(p*dx)/fe.assemble(fe.Constant(1.)*dx)))
            else:
                fe.assign(self.pressure, p)
            self.velocityFile << (self.velocity, t)
            self.pressureFile << (self.pressure, t)

            if self.stress:
                fe.assign(self.stress, self.fn.projectTensor(
                    two_muS*sym(grad(u)) + tau - self.pressure*self.I))
                self.stressFile << (self.stress, t)
            if self.strainrate:
                fe.assign(self.strainrate, self.fn.projectTensor(
                    TWO*sym(grad(u))))
                self.strainrateFile << (self.strainrate, t)
            if self.polystress:
                fe.assign(self.polystress, tau)
                self.polystressFile << (self.polystress, t)
            if self.N1:
                fe.assign(self.N1, self.fn.projectScalar(
                      two_muS*sym(grad(u))[0,0] + tau[0,0]
                    - two_muS*sym(grad(u))[1,1] - tau[1,1]))
                self.N1File << (self.N1, t)

            # prepare for next time step
            u0.assign(u)
            self.model.post_step (self, u)
            self.post_step_callback(self, t)

    def create_files(self, path='.', polystress=False, N1=False):
        self.velocityFile = fe.File(path + '/velocity.pvd')
        self.pressureFile = fe.File(path + '/pressure.pvd')
        self.stressFile = fe.File(path + '/stress.pvd')
        self.strainrateFile = fe.File(path + '/strainrate.pvd')
        if polystress:
            self.polystressFile = fe.File(path + '/polystress.pvd')
        if N1:
            self.N1File = fe.File(path + '/N1.pvd')
