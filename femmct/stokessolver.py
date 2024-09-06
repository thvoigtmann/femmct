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
    """Incompressible Stokes equation solver with deformation field history.

    This class provides a solver for the incompressible Stokes equation that
    interfaces with integral constitutive equations that are based on the
    deformation field history (given through the Finger tensors, evaluated
    on a logarithmic age grid)."""
    def __init__(self, mesh, **kwargs):
        """Initialization of the Stokes solver.

        Parameters
        ----------
        mesh : dolfin.Mesh
            Mesh for the numerical computation.
        **kwargs : dict, optional
            Extra arguments to set boundary conditions (see `apply_bc`)
            and to fine-tune dolfin settings.
        """
        self.I = fe.Constant(((1., 0.), (0., 1.)))
        #self.I = fe.Constant(np.diag(np.ones(mesh.geometric_dimension())))

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
        """Add boundary condition to the solver.

        Parameters
        ----------
        boundary_conditions : dict
            This should be a dict whose keys are the integer values
            used in `subdomain_data` to mark the relevant regions of the
            mesh, and whose entries are functions that return either a
            dolfin.Constant (interpreted as a pressure value to be set)
            or a suitable dolfin.DirichletBC object.
            See `femmct.dirichlet_u` or `femmct.p_inlet` or
            `femmct.p_outlet` for examples.
        subdomain_data : dolfin.MeshFunction, optional
            A `size_t` function containing zeros for non-boundary nodes,
            and integer values corresponding to the boundary-condition
            labels used in `boundary_conditions` on the boundary nodes.
        """
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
        """Initialize the Stokes solver.

        Parameters
        ----------
        model : femmct.ICEModel, default: None
            Class object that implements the integral constitutive equation.
            Defaults to `None`, which just solves the Newtonian background
            fluid. The `model` should be used to specify the additional
            non-Newtonian stresses.
        T : float, optional
            Maximum time for which to solve.
        Nt : int, default: 500
            Number of time steps (determines the time step together with `T`).
        Na : int, default: 16
            Number of time steps in each block of age information.
        Nb : int, default: 6
            Number of blocks of age information, where each block contains
            `Na` points and each additional block stores age information
            on twice the time step as the previous one.
        parameters : dict, optional
            Setting for numerical parameters in the Stokes equation.
            Should contain keys 'rho' and 'muS' for the density and the
            Newtonian background viscosity.

        Notes
        -----
        The parameters `Nt`, `Na`, and `Nb` might be adjusted depending
        on the specific model in case the implementation requires certain
        relations between them. Also, the number of history blocks `Nb`
        will be cut to those that are actually needed for the specified
        maximum time interval; any older history would not enter the
        equations anyway.
        """
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

    def loop (self, stress=True, strainrate=True, polystress=False, N1=False,
              normalize_pressure=True, **kwargs):
        """Time-step loop to solver Stokes equation with ICE.

        Parameters
        ----------
        stress : bool, default: True
            If True, calculate the stress field and write to file.
        strainrate : bool, default: True
            If True, calculate the strain rate field and write to file.
        polystress : bool, default: False
            If True, calculate the non-Newtonian "polymeric" stress
            and write separately to a file.
        N1 : bool, default: False
            If True, calculate the first normal-stress difference
            separately.
        normalize_pressure : bool, default: True
            If True, the pressure written to the output file will be
            normalized to 1/volume, since the overall stress magnitude
            does not play a role in the incompressible Stokes equation.
            This setting only affects the output, not the calculation.
        **kwargs : dict, optional
            Optional parameters passed to `dolfin.solve()`.

        Note
        ----
        The stress is given by the addition of the Newtonian stress,
        the pressure gradient, and the non-Newtonian stress. To get the
        stress contribution from the non-Newtonian constitutive law,
        set `polystress` to `True`. The first normal-stress difference
        is calculated separately when setting `N1` to `True`; since this
        performs the projection on the finite-element function space
        properly, this is better than evaluating the difference a posteriori
        from the polystress file.
        """
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
        """Create output files for the solutions.

        By default, no output files will be created. Calling this
        function, by default the velocity, pressure, total stress,
        and strainrate will be stored. The non-Newtonian stress and
        the first normal-stress difference can be separately stored.

        Parameters
        ----------
        path : str, default: '.'
            Base path where to write the files. Must be a directory,
            and in there, the various `pvd` files and their `vtk`/`vtu`
            companions will be created (one for each field and time step).
        polystress : bool, default: False
            If True, also create files for the non-Newtonian stress
            separately.
        N1 : bool, default: False
            If True, also create files for the first normal-stress difference
            separately.

        Note
        ----
        Creating a file does not yet imply that the corresponding quantity
        will be calculated, and vice versa. The calculation is controlled
        by the flags given to `solve()`.
        """
        self.velocityFile = fe.File(path + '/velocity.pvd')
        self.pressureFile = fe.File(path + '/pressure.pvd')
        self.stressFile = fe.File(path + '/stress.pvd')
        self.strainrateFile = fe.File(path + '/strainrate.pvd')
        if polystress:
            self.polystressFile = fe.File(path + '/polystress.pvd')
        if N1:
            self.N1File = fe.File(path + '/N1.pvd')
