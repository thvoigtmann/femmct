import numpy as np
import dolfin
from .aux import FunctionSpaces, DevNullFile, filtered_kwargs

def dirichlet_u (constant):
    return lambda up, boundaries, idx: dolfin.DirichletBC(up.sub(0), dolfin.Constant(constant), boundaries, idx)
def p_inlet (dp):
    return lambda up, boundaries, idx: dolfin.Constant(dp)
def p_outlet (dp):
    return lambda up, boundaries, idx: dolfin.Constant(dp)

# rho: density, muS: solvent viscosity
def default_parameters ():
    return {
      'rho': 1.0,
      'muS': 1.0,
    }


class StokesSolver(object):
    def __init__(self, mesh, **kwargs):
        self.I = dolfin.Constant(((1., 0.), (0., 1.)))

        self.fn = FunctionSpaces(mesh, **kwargs)

        kwargs2 = filtered_kwargs(kwargs, dolfin.Measure.__call__)
        self.dx = dolfin.Measure('dx', mesh)
        self.ds = dolfin.Measure('ds', mesh)(**kwargs2)

        self.n = dolfin.FacetNormal(mesh)

        self.BClist = []
        self.BCs = {}
        self.dps = {}
        if 'boundary_conditions' in kwargs:
            self.apply_bc (kwargs['boundary_conditions'], kwargs['subdomain_data'])

        self.callback = lambda obj, step: None

        self.velocityFile = DevNullFile()
        self.pressureFile = DevNullFile()
        self.stressFile = DevNullFile()
        self.strainrateFile = DevNullFile()
        self.polystressFile = DevNullFile()
        self.N1File = DevNullFile()


    def apply_bc (self, boundary_conditions, subdomain_data = None):
        for mark_idx,bcfunc in boundary_conditions.items():
            bc = bcfunc(self.fn.UP, subdomain_data, mark_idx)
            if isinstance(bc, dolfin.Constant):
                self.dps[mark_idx] = bc
            else:
                self.BCs[mark_idx] = bc
        self.BClist = list(self.BCs.values())

    # initialize makes new empty arrays for Finger and G tensors
    # ie clears all history-related information
    # Nt: time steps, Na: steps per age block, Nb: age blocks
    def initialize (self, model = None, T = 50., Nt = 500, Na = 16, Nb = 6, parameters = default_parameters()):
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
            print('(INFO) Using only', Nbmax, 'instead of', self.Nb, 'blocks of deformation fields')
            self.Nb = Nbmax

        # total number of deformation fields to store
        self.Nh = self.Na * self.Nb

        # time step
        self.dt = T/self.Nt

        # age on lin-log grid in multiples of dt
        #age = np.ones(self.Nh)
        #for l in range(self.Nb):
        #    for m in range(self.Na):
        #        age[l*Na + m] = (2**l - 1)*Na + 2**l*(m+1)

        self.Bs = [] # array of Finger tensor fields of increasing age
        # Bs[0] = youngest Finger tensor with age a = dt
        # Bs[Nh-1] = oldest Finger tensor with age a = (2**Nb - 1)*Na*dt
        #self.Gs = []
        # initialize Finger tensors:
        for i in range(self.Nh):
            self.Bs.append(dolfin.Function(self.fn.Tau, name='Finger tensor'))
            self.Bs[i].assign(dolfin.interpolate(self.I, self.fn.Tau))
            # IWB shear moduli
            # self.Gs.append(dolfin.Function(self.fn.P))
            # self.Gs[i].assign(dolfin.interpolate(dolfin.Constant(model.GInf*np.exp(-age[i]*self.dt/model.lambdaC)), self.fn.P))
        self.model.initialize(self)
        # for reference: store age a = 0 Finger tensor and shear modulus
        # nearest younger Finger tensor of age a = 0 is Id
        # nearest younger shear modulus of age a = 0 is GInf
        self.Bs0 = self.I
        #self.Gs0 = dolfin.Constant(model.GInf)

    # loop should initialize all instantaneous functions etc afresh
    def loop (self, **kwargs):
        n, dx, ds = self.n, self.dx, self.ds
        #Bs, Gs = self.Bs, self.Gs
        Bs = self.Bs

        # trial and test functions
        u_, p_ = dolfin.TrialFunctions(self.fn.UP)
        v, q = dolfin.TestFunctions(self.fn.UP)
        # functions to solve for
        u, p = dolfin.Function(self.fn.U), dolfin.Function(self.fn.P)

        up = dolfin.Function(self.fn.UP)
        tau = dolfin.Function(self.fn.Tau)
        u0 = dolfin.Function(self.fn.U)

        rho = self.parameters['rho']
        muS = self.parameters['muS']

        t = 0.
        velocity = dolfin.Function(self.fn.U, name = 'velocity')
        pressure = dolfin.Function(self.fn.P, name = 'pressure')
        stress = dolfin.Function(self.fn.Tau, name = 'total stress')
        strainrate = dolfin.Function(self.fn.Tau, name = 'strain rate')
        polystress = dolfin.Function(self.fn.Tau, name = 'polystress')
        N1 = dolfin.Function(self.fn.Tau, name = 'N1')

        self.velocityFile << (velocity, t)
        self.pressureFile << (pressure, t)
        self.stressFile << (stress, t)
        self.strainrateFile << (strainrate, t)
        self.polystressFile << (polystress, t)
        self.N1File << (N1, t)

        # to solve the evolution equation for the Finger tensors:
        B_ = dolfin.TrialFunction(self.fn.Tau)
        C = dolfin.TestFunction(self.fn.Tau)
        deformationsolver = dolfin.LUSolver()
        # to solve the evolution equation for the shear moduli (IWB):
        #G_ = dolfin.TrialFunction(self.fn.P)
        #H = dolfin.TestFunction(self.fn.P)
        #shearmodulussolver = dolfin.LUSolver('default')

        # time stepping
        for k in range(1, self.Nt + 1):
            print('Time Step ', k, ' out of ', self.Nt)
            t += self.dt

            un = dolfin.dot(u, self.n)
            unPos = (un + abs(un))/dolfin.Constant(2.)
            unNeg = (un - abs(un))/dolfin.Constant(2.)

            # evolution equation for the Finger tensors and shear moduli
            LHSFT = dolfin.Constant(1./self.dt) * dolfin.inner(B_,C)*dx
            LHSFT -= dolfin.inner(B_, dolfin.div(dolfin.outer(C, u)))*dx
            LHSFT += dolfin.inner(unPos('+')*B_('+') + unNeg('+')*B_('-'), dolfin.jump(C))*dolfin.dS
            LHSFT += dolfin.inner(un*B_, C)*ds
            LHSFT -= dolfin.inner(dolfin.grad(u)*B_ + B_*dolfin.grad(u).T, C)*dx
            FTmat = dolfin.assemble(LHSFT)
            deformationsolver.set_operator(FTmat)

            #lambdaInv = dolfin.Constant(1./model.lambdaC) + dolfin.Constant(np.sqrt(2.)/model.gammaC)*dolfin.sqrt(dolfin.inner(dolfin.sym(dolfin.grad(u)), dolfin.sym(dolfin.grad(u))))
            #LHSSM = dolfin.Constant(1./self.dt)*dolfin.inner(G_,H)*dx
            #LHSSM -= dolfin.inner(G_, dolfin.div(dolfin.outer(H, u)))*dx
            #LHSSM += dolfin.inner(unPos('+')*G_('+') + unNeg('+')*G_('-'), dolfin.jump(H))*dolfin.dS
            #LHSSM += dolfin.inner(un*G_, H)*ds
            #LHSSM += dolfin.inner(lambdaInv*G_, H)*dx

            #SMmat = dolfin.assemble(LHSSM)
            #shearmodulussolver.set_operator(SMmat)

            for l in range(self.Nb - 1, -1, -1):
                # iteration over history blocks in reverse order
                # l = 0: da = dt (finest resolution)
                # l = 1: da = 2*dt
                # l = 2: da = 4*dt
                # ...
                da = 2**l*self.dt
                for m in range(self.Na - 1, -1, -1):
                    j = l*self.Na + m
                    RHSFT = dolfin.Constant(1./self.dt)*dolfin.inner(Bs[j], C)*dx
                    #RHSSM = dolfin.Constant(1./self.dt)*dolfin.inner(Gs[j], H)*dx
                    if j == 0:
                        # nearest younger Finger tensor of age a = 0 is Id
                        # nearest younger shear modulus of age a = 0 is GInf
                        RHSFT -= dolfin.Constant(1./da)*dolfin.inner(Bs[0] - self.Bs0, C)*dx
                        #RHSSM -= dolfin.Constant(1./da)*dolfin.inner(Gs[0] - self.Gs0, H)*dx
                    else:
                        RHSFT -= dolfin.Constant(1./da)*dolfin.inner(Bs[j] - Bs[j-1], C)*dx
                        #RHSSM -= dolfin.Constant(1./da)*dolfin.inner(Gs[j] - Gs[j-1], H)*dx
                    FTvec = dolfin.assemble(RHSFT)
                    deformationsolver.solve (Bs[j].vector(), FTvec)
                    #SMvec = dolfin.assemble(RHSSM)
                    #shearmodulussolver.solve (Gs[j].vector(), SMvec)

            # stress integral, DG0 approximation in age a
            #for j in range(self.Nh):
            #    if j == 0:
            #        dolfin.assign(tau, self.fn.projectTensor(Gs[0]*(Bs[0] - self.Bs0)))
            #    else:
            #        dolfin.assign(tau, self.fn.projectTensor(tau + Gs[j]*(Bs[j] - Bs[j-1])))

            self.model.step(self, k, u, un, unPos, unNeg, tau)

            # callback, for example to turn on/off pressure gradient
            self.callback(self, k)

            # solve Stokes problem
            LHSNS = dolfin.Constant(rho/self.dt)*dolfin.inner(u_, v)*dx + dolfin.inner(dolfin.Constant(2.)*muS*dolfin.sym(dolfin.grad(u_)), dolfin.sym(dolfin.grad(v)))*dx - p_*dolfin.div(v)*dx - q*dolfin.div(u_)*dx
            RHSNS = dolfin.Constant(rho/self.dt)*dolfin.inner(u0, v)*dx - dolfin.inner(tau, dolfin.sym(dolfin.grad(v)))*dx + dolfin.dot(dolfin.avg(tau)*n('+'), dolfin.jump(v))*dolfin.dS + dolfin.dot(tau*n, v)*ds
            # TODO: assumes ds(1) and ds(2) to be inlet and outlet
            #RHSNS -= dolfin.dot(pIn*n, v)*ds(1) + dolfin.dot(pOut*n, v)*ds(2)
            for idx,dp in self.dps.items():
                RHSNS -= dolfin.dot(dp*n, v)*ds(idx)

            dolfin.solve(LHSNS == RHSNS, up, self.BClist, **kwargs)

            dolfin.assign(u, up.sub(0))
            dolfin.assign(p, up.sub(1))

            # export the current time step
            # TODO adjust to MCT code that does it differently here
            dolfin.assign(velocity, u)
            dolfin.assign(pressure, p)
            #dolfin.assign(pressure, self.fn.projectScalar(p - dolfin.Constant(dolfin.assemble(p*dx)/dolfin.assemble(dolfin.Constant(1.)*dx))))
            dolfin.assign(stress, self.fn.projectTensor(dolfin.Constant(2.*muS)*dolfin.sym(dolfin.grad(u)) + tau - pressure*self.I))
            dolfin.assign(strainrate, self.fn.projectTensor(dolfin.Constant(2.)*dolfin.sym(dolfin.grad(u))))
            #dolfin.assign(polystress, tau)
            #dolfin.assign(N1, self.fn.projectScalar(dolfin.Constant(2.*muS)*dolfin.sym(dolfin.grad(u))[0,0] + tau[0,0] - dolfin.Constant(2.*muS)*dolfin.sym(dolfin.grad(u))[1,1] - tau[1,1]))

            self.velocityFile << (velocity, t)
            self.pressureFile << (pressure, t)
            self.stressFile << (stress, t)
            self.strainrateFile << (strainrate, t)
            self.polystressFile << (polystress, t)
            self.N1File << (N1, t)

            # prepare for next time step
            u0.assign(u)
            self.model.post_step (self, u)

    def create_files(self, path='.', polystress=False, N1=False):
        self.velocityFile = dolfin.File(path + '/velocity.pvd')
        self.pressureFile = dolfin.File(path + '/pressure.pvd')
        self.stressFile = dolfin.File(path + '/stress.pvd')
        self.strainrateFile = dolfin.File(path + '/strainrate.pvd')
        if polystress:
            self.polystressFile = dolfin.File(path + '/polystress.pvd')
        if N1:
            self.N1File = dolfin.File(path + '/N1.pvd')
