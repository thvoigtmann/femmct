from dolfin import *
import numpy as np

parameters['ghost_mode'] = 'shared_facet'
parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['quadrature_degree'] = 4

# Problem parameters
length = 5. # channel length
height = 1. # channel height
T = 50. # final time
Nx = 50 # subintervals in horizontal direction
Ny = 50 # subintervals in vertical direction
Nt = 500 # subintervals in time
Na = 16 # subintervals in history (age) at the finest resolution
Nb = 6 # number of blocks in history (age) with constant resolution each
rho = 1. # density
muS = 1. # solvent viscosity
deltap = 2.5 # pressure difference between inlet and outlet
GInf = 1. # shear modulus at short time scales
lambdaC = 10. # constant time-scale parameter in the generalised Oldroyd-B model
gammaC = .1 # characteristic strain parameter in the generalised Oldroyd-B model


# Discard surplus deformation history
# We save all history from age = 0 to age = (2**Nb - 1)*Na*T/Nt
# History beyond age = T would never be used -> discard
Nbmax = int(np.ceil(np.log2(Nt/Na + 1.)))
if Nb > Nbmax:
    print('(INFO) Using only', Nbmax, 'instead of', Nb, 'blocks of deformation fields as these already cover the full simulated time span.')
    Nb = Nbmax

path = 'GOB_' + str(Nb) + 'x' + str(Na) + '_pressuredriven_dp-' + str(deltap) + '_GInf-' + str(GInf) +'_lambdaC-' + str(lambdaC) + '_muS-' + str(muS)
   
Nh = Na*Nb # total number of deformation fields used to capure the deformation history
Ntprime = (2**Nb - 1)*Na # maximum age in multiples of dt

# Compute interval widths in the lin-log grid in multiples of dt
stepsize = np.ones(Nh)
for l in range(1, Nb):
    stepsize[l*Na : (l + 1)*Na] = 2**l

# Compute absolute ages in the lin-log grid in multiples of dt
age = np.ones(Nh)
for l in range(Nb):
    for m in range(Na):
        age[l*Na + m] = (2**l - 1)*Na + 2**l*(m + 1)

# Geometric set up
mesh = RectangleMesh(Point(0., 0.), Point(length, height), Nx, Ny, diagonal = 'crossed')

class Inflow(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[0], 0.))

class Outflow(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[0], length))

class TopWall(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[1], height))

class BottomWall(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[1], 0.))

class PeriodicBoundary(SubDomain):
    # values on the inlet will be overwritten with values on the outlet
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[0], 0.))

    def map(self, x, y):
        y[0] = x[0] - length
        y[1] = x[1]
        
inflow = Inflow()
outflow = Outflow()
topWall = TopWall()
bottomWall = BottomWall()

boundaries = MeshFunction('size_t', mesh, 1)
boundaries.set_all(0)
inflow.mark(boundaries, 1)
outflow.mark(boundaries, 2)
topWall.mark(boundaries, 3)
bottomWall.mark(boundaries, 4)

ds = Measure('ds', mesh)(subdomain_data = boundaries)
dx = Measure('dx', mesh)
dt = T/Nt

# Normal vectors
n = FacetNormal(mesh)
ex = Constant((1., 0.))
ey = Constant((0., 1.))
I = Constant(((1., 0.), (0., 1.)))


# Function spaces
FEu = VectorElement('CG', mesh.ufl_cell(), 2) # velocity element
FEp = FiniteElement('DG', mesh.ufl_cell(), 0) # pressure element
FEtau = TensorElement('DG', mesh.ufl_cell(), 0) # stress element
UP = FunctionSpace(mesh, MixedElement([FEu, FEp]), constrained_domain = PeriodicBoundary())
U = UP.sub(0).collapse()
P = UP.sub(1).collapse()
Tau = FunctionSpace(mesh, FEtau, constrained_domain = PeriodicBoundary())

# Auxiliary functions for elementwise projections
def projectScalar(r):
    p = TrialFunction(P)
    q = TestFunction(P)
    LHS = inner(p, q)*dx
    RHS = inner(r, q)*dx
    solver = LocalSolver(LHS, RHS)
    solver.factorize()
    p = Function(P)
    solver.solve_local_rhs(p)
    return p

def projectTensor(r):
    tau = TrialFunction(Tau)
    sig = TestFunction(Tau)
    LHS = inner(tau, sig)*dx
    RHS = inner(r, sig)*dx
    solver = LocalSolver(LHS, RHS)
    solver.factorize()
    tau = Function(Tau)
    solver.solve_local_rhs(tau)
    return tau


# Boundary conditions
BCuTop = DirichletBC(UP.sub(0), Constant((0., 0.)), boundaries, 3)
BCuBottom = DirichletBC(UP.sub(0), Constant((0., 0.)), boundaries, 4)
BCs = [BCuTop, BCuBottom]

pIn = Constant(deltap)
pOut = Constant(0.)

# Functions
u = Function(U)
p = Function(P)
up = Function(UP)
tau = Function(Tau)
u0 = Function(U)
tau0 = Function(Tau)

# Initial conditions
# (the flow starts at rest with u(t=0), p(t=0), B(t=0,a) = id  for all ages a >= 0)
Bs = [] # array of Finger tensor fields of increasing age:
# Bs[0] = youngest Finger tensor with age a = dt
# Bs[1] = second-youngest Finger tensor with age a = 2*dt (or a = 3*dt if Na = 1)
# ...
# Bs[Nh - 1] = oldest Finger tensor with age a = (2**Nb - 1)*Na*dt

Gs = []
lambdaInvs = []

for i in range(Nh):
    # Initialise all Finger tensors as identity tensors
    Bs.append(Function(Tau, name='Finger tensor'))
    Bs[i].assign(interpolate(I, Tau))
    
    # Initialise shear moduli
    # For Du(t) = 0 (t <= 0), the generalised Oldroyd-B model simplifies to 1/lambda(t-a) = 1/lambdaC and hence G(t=0, a) = GInf*exp(-a/lambdaC)
    Gs.append(Function(P))
    Gs[i].assign(interpolate(Constant(GInf*exp(-age[i]*dt/lambdaC)), P))

for i in range(Ntprime):
    lambdaInvs.append(Function(P))
    lambdaInvs[i].assign(interpolate(Constant(1./lambdaC), P))
    

u_, p_ = TrialFunctions(UP)
v, q = TestFunctions(UP)

# Data export
velocityFile = File(path + '/velocity.pvd')
pressureFile = File(path + '/pressure.pvd')
stressFile = File(path + '/stress.pvd')
strainrateFile = File(path + '/strainrate.pvd')

t = 0.

velocity = Function(U, name = 'velocity')
pressure = Function(P, name = 'pressure')
stress = Function(Tau, name = 'total stress')
strainrate = Function(Tau, name = 'strain rate')

velocityFile << (velocity, t)
pressureFile << (pressure, t)
stressFile << (stress, t)
strainrateFile << (strainrate, t)

# Evolution equation for Finger tensors (w/- backward Euler in time and age)
B_ = TrialFunction(Tau)
C = TestFunction(Tau)
deformationsolver = LUSolver()

# Evolution equation for shear moduli (w/- backward Euler in time and age)
G_ = TrialFunction(P)
H = TestFunction(P)
shearmodulussolver = LUSolver('default')

# Time stepping
for k in range(1, Nt + 1):
    print('Time Step ', k, ' out of ', Nt)
    t += dt
    
    # Evolution equation for the Finger tensors & shear moduli
    un = dot(u, n)
    unPos = (un + abs(un))/Constant(2.)
    unNeg = (un - abs(un))/Constant(2.)
    
    LHSFT = Constant(1./dt)*inner(B_, C)*dx
    LHSFT += -inner(B_, div(outer(C, u)))*dx
    LHSFT += inner(unPos('+')*B_('+') + unNeg('+')*B_('-'), jump(C))*dS
    LHSFT += inner(un*B_, C)*ds
    LHSFT -= inner(grad(u)*B_ + B_*grad(u).T, C)*dx
    FT_mat = assemble(LHSFT)
    deformationsolver.set_operator(FT_mat)
    
    for l in range(Nb - 1, -1, -1):
      # Iteration over history blocks (in reverse order)
      # l = 0: da = dt (finest resolution)
      # l = 1: da = 2*dt
      # l = 2: da = 4*dt
      # ...
      da = 2**l*dt
      
      for m in range(Na - 1, -1, -1):
          j = l*Na + m
          RHSFT = Constant(1./dt)*inner(Bs[j], C)*dx
          RHSSM = Constant(1./dt)*inner(Gs[j], H)*dx
          
          if j == 0:
              # The nearest younger Finger tensor of age a = 0 is the identity tensor
              RHSFT -= Constant(1./da)*inner(Bs[0] - I, C)*dx
              # The nearest younger shear modulus of age a = 0 is GInf
              RHSSM -= Constant(1./da)*inner(Gs[0] - Constant(GInf), H)*dx
          else:
              RHSFT -= Constant(1./da)*inner(Bs[j] - Bs[j - 1], C)*dx
              RHSSM -= Constant(1./da)*inner(Gs[j] - Gs[j - 1], H)*dx
          
          FT_vec = assemble(RHSFT)
          deformationsolver.solve(Bs[j].vector(), FT_vec)
          
          LHSSM = Constant(1./dt)*inner(G_, H)*dx
          LHSSM += -inner(G_, div(outer(H, u)))*dx
          LHSSM += inner(unPos('+')*G_('+') + unNeg('+')*G_('-'), jump(H))*dS
          LHSSM += inner(un*G_, H)*ds
          LHSSM += inner(lambdaInvs[(2**l - 1)*Na + 2**l*(m + 1) - 1]*G_, H)*dx
          SM_mat = assemble(LHSSM)
          shearmodulussolver.set_operator(SM_mat)
          SM_vec = assemble(RHSSM)
          shearmodulussolver.solve(Gs[j].vector(), SM_vec)
          
    # Move all inverse time scales back by one time step
    for j in range(Ntprime - 1, -1, -1):
      if j == 0:
          lambdaInvs[0].assign(projectScalar(Constant(1./lambdaC) + Constant(sqrt(2.)/gammaC)*sqrt(inner(sym(grad(u)), sym(grad(u))))))
      else:
          lambdaInvs[j].assign(lambdaInvs[j - 1])
        
    # Stress integral (approximated with DG(0) in age a)
    for j in range(Nh):
        if j == 0:
            # The Finger tensor of age a = 0 is the identity tensor and the shear modulus G = GInf
            assign(tau, projectTensor(Gs[0]*(Bs[0] - I)))
        else:
            assign(tau, projectTensor(tau + Gs[j]*(Bs[j] - Bs[j - 1])))
    
    # Turn off the pressure gradient after half of the simulation time has elapsed
    if k == int(Nt/2):
        pIn = Constant(0.)
    
    # Solve the Stokes problem    
    LHSNS = Constant(rho/dt)*inner(u_, v)*dx + inner(Constant(2.)*muS*sym(grad(u_)), sym(grad(v)))*dx - p_*div(v)*dx - q*div(u_)*dx
    RHSNS = Constant(rho/dt)*inner(u0, v)*dx - inner(tau, sym(grad(v)))*dx + dot(avg(tau)*n('+'), jump(v))*dS + dot(tau*n, v)*ds
    RHSNS -= dot(pIn*n, v)*ds(1) + dot(pOut*n, v)*ds(2)

    solve(LHSNS == RHSNS, up, BCs)
    
    assign(u, up.sub(0))
    assign(p, up.sub(1))
    
    # Export this current time step
    assign(velocity, u)
    assign(pressure, p)
    assign(stress, projectTensor(Constant(2.*muS)*sym(grad(u)) + tau - pressure*I))
    assign(strainrate, projectTensor(Constant(2.)*sym(grad(u))))
    
    velocityFile << (velocity, t)
    pressureFile << (pressure, t)
    stressFile << (stress, t)
    strainrateFile << (strainrate, t)
    
    u0.assign(u)