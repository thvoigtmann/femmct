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
Na = 8 # subintervals in history (age) at the finest resolution
Nb = 6 # number of blocks in history (age) with constant resolution each
rho = 1. # density
muS = 1. # solvent viscosity
deltap = 2.5 # pressure difference between inlet and outlet
GInf = 1. # shear modulus at short time scales
lambdaC = 1. # constant time-scale parameter in the MCT model
gammaC = .1 # characteristic strain parameter in the MCT model
v1 = 0.
v2 = 6.
absTol = 1E-9 # absolute tolerance for nonlinear iterations
relTol = 1E-6 # relative tolerance for nonlinear iterations
maxIter = 25 # maximum number of nonlinear iterations
outerAdvection = True # use an advected derivative in front of the MCT integral if True
innerAdvection = True # use an advected derivative inside the MCT integral if True

# Parts of the following code assume that the fine end of the lin-log mesh resolves intervals of width dt, 2dt, 4dt, ..., (2**(Nb-1))dt exactly. We adjust the user input for Na to make sure this condition is met.
# The smallest power of 2 such that 2**l >= Na:
if Nb > 1:
    l = int(np.ceil(np.log2(Na)))
    for k in range(l, Nb):
        # The number of blocks that fully fit into 2**(Nb - 1) when Na = 2**k
        L = int(np.log2(1 + 2**(Nb - 1 - k))) - 1
        # There may be a gap left between the full blocks 0, ..., L and the big interval 2**(Nb-1)
        # We must be able to fill this gap with an integer number of steps of length 2**(L+1) or else the meshes don't match as required
        if np.mod(2**(Nb - 1) - (2**(L + 1) - 1)*2**k, 2**(L + 1)) == 0:
            # The meshes match
            l = k
            break
    
    if Na != 2**l:
        print('(INFO) Adjusting the number of subintervals in each block from', Na, 'to', 2**l, '.')
        Na = 2**l

# Discard surplus deformation history
# We save all history from age = 0 to age = (2**Nb - 1)*Na*T/Nt
# History beyond age = T would never be used -> discard
Nbmax = int(np.ceil(np.log2(Nt/Na + 1.)))
if Nb > Nbmax:
    print('(INFO) Using only', Nbmax, 'instead of', Nb, 'blocks of deformation fields as these already cover the full simulated time span.')
    Nb = Nbmax

path = 'MCT_'
if innerAdvection:
    path += 'innerAdv_'
if outerAdvection:
    path += 'outerAdv_'
path += str(Nb) + 'x' + str(Na) + '_pressuredriven_dp-' + str(deltap) + '_GInf-' + str(GInf) +'_v1-' + str(v1) + '_v2-' + str(v2)
   
Nh = Na*Nb # total number of deformation fields used to capture the deformation history
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

# MCT memory kernel (F12 like)
def memory_kernel(phi):
    return Constant(v1)*phi + Constant(v2)*phi*phi

def diff_memory_kernel(phi):
    return Constant(v1) + Constant(2.*v2)*phi

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

hs = []
phis = []
us = []

for i in range(Ntprime):
    # Initialise velocity fields as zero
    us.append(Function(U, name='Velocity'))

for i in range(Nh):
    # Initialise all Finger tensors as identity tensors
    Bs.append(Function(Tau, name='Finger tensor'))
    Bs[i].assign(interpolate(I, Tau))
    
    # Initialse shear factors
    hs.append(Function(P))
    hs[i].assign(projectScalar(Constant(gammaC**2.)/(Constant(gammaC**2.) + tr(Bs[i]) - Constant(2.))))
    
    # Initialise correlators as exponential functions (= analytical solution with vanishing memory kernel)
    phis.append([])
    
    l = int(np.floor(i/Na)) # block index (between 0 and Nb - 1)
    m = i - l*Na # index within block (between 0 and Na - 1, such that i = l*Nb + m)
    a = ((2**l - 1)*Na + 2**l*(m + 1))*dt # the age of all correlators on this diagonal
    
    for j in range((2**Nb - 2**l)*Na - 2**l*(m + 1) + 1):
        # this is the length of this constant-age diagonal in the lin-log grid
        # j = 0 corresponds to the current time t, larger j's to past times t - j*dt
        phis[i].append(Function(P))
        phis[i][j].assign(interpolate(Constant(exp(-a/lambdaC)), P))

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

phi = Function(P)
psi = TestFunction(P)
FMCT = Function(P)
F0MCT = Function(P)
HM = Function(P)
DFMCT = Function(P)
dphi = Function(P)
dphi_ = TrialFunction(P)
correlatorsolver = LUSolver()

# Time stepping
for k in range(1, Nt + 1):
    print('Time Step ', k, ' out of ', Nt)
    t += dt
    
    # Evolution equation for the Finger tensors
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
      # iteration over history blocks (in reverse order)
      # l = 0: da = dt (finest resolution)
      # l = 1: da = 2*dt
      # l = 2: da = 4*dt
      # ...
      da = 2**l*dt
      
      for m in range(Na - 1, -1, -1):
          j = l*Na + m
          RHSFT = Constant(1./dt)*inner(Bs[j], C)*dx
          
          if j == 0:
              # The nearest younger Finger tensor of age a = 0 is the identity tensor
              RHSFT -= Constant(1./da)*inner(Bs[0] - I, C)*dx
          else:
              RHSFT -= Constant(1./da)*inner(Bs[j] - Bs[j - 1], C)*dx
          
          FT_vec = assemble(RHSFT)
          deformationsolver.solve(Bs[j].vector(), FT_vec)
          
          # Compute corresponding shear factor
          hs[j].assign(projectScalar(Constant(gammaC**2.)/(Constant(gammaC**2.) + tr(Bs[j]) - Constant(2.))))
    
    
    for i in range(Nh - 1):
        l = int(np.floor(i/Na)) # block index (between 0 and Nb - 1)
        m = i - l*Na # index within block (between 0 and Na - 1, such that i = l*Nb + m)
        
        for j in range((2**Nb - 2**l)*Na - 2**l*(m + 1), 0, -1):

            # Move all correlators back by one time step
            phis[i][j].assign(phis[i][j-1])
    
    for l in range(Nb):
        
        da = 2**l*dt
        
        for m in range(Na):
            # Compute correlators at the current time
            # We proceed downwards (increasing age)
            
            j = l*Na + m
            
            # We distinguish three types of terms in the discretised MCT integral:
            #  (1) terms in which the current unknown phi is referenced BOTH in the horizontal t integration AND the vertical t' integration
            #  (2) terms in which the current unknown phi is referenced EITHER in the horizontal t integration OR the vertical t' integration
            #  (3) terms in which the current unknown phi is referenced NEITHER in the horizontal t integration NOR the vertical t' integration
            # We therefore split the integration domain into intervals of type (1), type (2) and type (3). 
            
            if j == 0:
                # Type (1) only arises in the integral for the youngest correlator phi[0][0]
                
                # We use the immediate past iterate for the initial guess
                phi.assign(phis[0][0])
                
                print('   Computing phi[ 0 ]...')
                print('      Iter |    ||dphi||   |   ||phi||')
                
                # Newton's method to solve for phi
                for ni in range(1, maxIter + 1):
                    # The nearest younger correlator of age a = 0 is 1
                    RHSMCT = Constant(lambdaC/dt)*(phi - Constant(1.))*psi*dx
                    if outerAdvection:
                        RHSMCT -= Constant(lambdaC)*inner(phi, div(outer(psi, u)))*dx
                        RHSMCT += Constant(lambdaC)*inner(unPos('+')*phi('+') + unNeg('+')*phi('-'), jump(psi))*dS
                        RHSMCT += Constant(lambdaC)*inner(un*phi, psi)*ds
                    
                    RHSMCT += phi*psi*dx
                    
                    RHSMCT += hs[0]*hs[0]*memory_kernel(phi)*(phi - Constant(1.))*psi*dx
                    if innerAdvection:
                        RHSMCT -= Constant(dt)*inner(phi, div(outer(hs[0]*hs[0]*memory_kernel(phi)*psi, u)))*dx
                        RHSMCT += Constant(dt)*inner(unPos('+')*phi('+') + unNeg('+')*phi('-'), jump(hs[0]*hs[0]*memory_kernel(phi)*psi))*dS
                        RHSMCT += Constant(dt)*inner(un*phi, hs[0]*hs[0]*memory_kernel(phi)*psi)*ds
                    
                    LHSMCT = Constant(lambdaC/dt)*dphi_*psi*dx
                    if outerAdvection:
                        LHSMCT -= Constant(lambdaC)*inner(dphi_, div(outer(psi, u)))*dx
                        LHSMCT += Constant(lambdaC)*inner(unPos('+')*dphi_('+') + unNeg('+')*dphi_('-'), jump(psi))*dS
                        LHSMCT += Constant(lambdaC)*inner(un*dphi_, psi)*ds
                    
                    LHSMCT += dphi_*psi*dx
                    
                    LHSMCT += hs[0]*hs[0]*(memory_kernel(phi) + diff_memory_kernel(phi)*(phi - Constant(1.)))*dphi_*psi*dx
                    if innerAdvection:
                        LHSMCT -= Constant(dt)*inner(dphi_, div(outer(hs[0]*hs[0]*memory_kernel(phi)*psi, u)))*dx
                        LHSMCT += Constant(dt)*inner(unPos('+')*dphi_('+') + unNeg('+')*dphi_('-'), jump(hs[0]*hs[0]*memory_kernel(phi)*psi))*dS
                        LHSMCT += Constant(dt)*inner(un*dphi_, hs[0]*hs[0]*memory_kernel(phi)*psi)*ds
                        LHSMCT -= Constant(dt)*inner(phi, div(outer(hs[0]*hs[0]*diff_memory_kernel(phi)*dphi_*psi, u)))*dx
                        LHSMCT += Constant(dt)*inner(unPos('+')*phi('+') + unNeg('+')*phi('-'), jump(hs[0]*hs[0]*diff_memory_kernel(phi)*dphi_*psi))*dS
                        LHSMCT += Constant(dt)*inner(un*phi, hs[0]*hs[0]*diff_memory_kernel(phi)*dphi_*psi)*ds
                    
                    MCT_mat = assemble(LHSMCT)
                    MCT_vec = assemble(-RHSMCT)
                    
                    correlatorsolver.set_operator(MCT_mat)
                    correlatorsolver.solve(dphi.vector(), MCT_vec)
                    
                    phi.assign(phi + dphi)
                    
                    dphiL2 = np.sqrt(assemble(dphi*dphi*dx)) 
                    phiL2 = np.sqrt(assemble(phi*phi*dx))
                    
                    # Print convergence history
                    print(f'{ni:10.0f}', '|', f'{dphiL2:13.2e}', '|', f'{phiL2:13.2e}')
                    
                    if dphiL2 <= absTol or dphiL2 <= relTol*phiL2:
                        # Newton's method has converged
                        break
                    elif ni == maxIter:
                        # Newton's method failed to converge
                        print('(WARNING) No convergence after', maxIter, 'iterations.')                        
            else:
                # Older correlators contain a term of type (2), possibly terms of type (3), and another term of type (2).
                # We separate off the two (distinct) terms of type (2) and then add terms of type (3), if there are any.
                
                # We use the immediate past iterate for the initial guess
                phi.assign(phis[j][0])
                
                print('   Computing phi[', j ,']...')
                print('      Iter |    ||dphi||   |   ||phi||')
                
                # The terms of type (2) appear at the beginning and end of the MCT integral. Near phi, the time step is (2**l)*dt. Near the boundary where phi = 1, the time steps are finer (dt, 2dt, 4dt, ...).
                # The time step (2**l)*dt from the coarse end covers the same time span as all time steps up to block L, plus step M from the fine end.
                L = int(np.ceil(np.log2(1 + 2**l/Na))) - 1
                M = int((2**l - (2**L - 1)*Na)/(2**L)) - 1
                J = L*Na + M
                
                # Terms of type (3): steps J, ..., j-J 
                F0MCT.assign(projectScalar(Constant(lambdaC/da)*(phis[j][0] - phis[j - 1][1])))
                
                if innerAdvection:
                    # Reset advection terms of type(3)
                    F0MCTadv = Constant(0.)*psi*dx
                
                tVert = age[J:j]-age[J] # lin-log grid points downwards counting from index J to j-1 in multiples of dt
                if J > 0:
                    tHorz = age[j-1]-age[j-1:J-1:-1] # lin-log grid points leftwards counting from index j-1 in multiples of dt
                else:
                    tHorz = age[j-1]-age[j-1::-1] # NB: age[j-1:-1:-1] would always be empty so we have to index the case J = 0 differently
                
                for i in range(J + 1, j):
                    # We iterate over vertical (age) intervals and have to find the corresponding projected grid points on the horizontal (time) axis
                    # On the vertical age axis, the i-th interval begins at tVert[i - (J + 1)] and ends at tVert[i - J].
                    # On the horizontal time axis, we will identify
                    #   * the largest grid point <= tVert[i - J] in leftBound together with a weight for interpolation between this grid point and its right neighbour (w*point + (1-w)*neighbour)
                    #   * the smallest grid point >= tVert[i - (J + 1)] in rightBound together with a weight for interpolation between this grid point and its left neighbour (w*point + (1-w)*neighbour)
                    leftTime = next(left for left in tHorz if left >= tVert[i - J])
                    leftBound = j - 1 - np.argmax(tHorz == leftTime) # NB: this is the index for the age array
                    leftWeight = 1 - ((age[j] - age[i]) - age[leftBound])/(age[leftBound + 1] - age[leftBound])
                    
                    rightTime = next(right for right in np.flip(tHorz) if right <= tVert[i - (J + 1)])
                    rightBound = j - 1 - np.argmax(tHorz == rightTime) # NB: this is the index for the age array
                    rightWeight = 1 - ((age[j] - age[i - 1]) - age[rightBound])/(age[rightBound - 1] - age[rightBound])
                    
                    F0MCT.assign(F0MCT + projectScalar(hs[j]*hs[i]*memory_kernel(phis[i][0])*((Constant(rightWeight)*phis[rightBound][int(age[j] - age[rightBound])] + Constant(1. - rightWeight)*phis[rightBound - 1][int(age[j] - age[rightBound - 1])]) - (Constant(leftWeight)*phis[leftBound][int(age[j] - age[leftBound])] + Constant(1. - leftWeight)*phis[leftBound + 1][int(age[j] - age[leftBound + 1])]))))
                    
                    if innerAdvection:
                        # We iterate over horizontal (time) intervals from right to left and have to find the corresponding projected grid points on the vertical (age) axis
                        # On the horizontal time axis, the i-th interval begins at tHorz[i - (J + 1)] and ends at tHorz[i - J].
                        # On the vertical age axis, we will identify
                        #   * the largest grid point <= tHorz[i - J] in bottomBound
                        #   * the largest grid point < tHorz[i - (J + 1)] in topBound
                        bottomTime = next(bottom for bottom in tVert if bottom >= tHorz[i - J])
                        bottomBound = np.argmax(tVert == bottomTime)
                        
                        topTime = next(top for top in tVert if top > tHorz[i - (J + 1)])
                        topBound = np.argmax(tVert == topTime)
                        
                        HM.assign(Function(P))
                        
                        for k in range(topBound, bottomBound + 1):
                            # each h*m contribution spans the time interval from tVert[k - 1] to tVert[k], intersected with the interval from tHorz[i - J] to tHorz[i - (J + 1)]
                            HM.assign(HM + projectScalar(Constant(np.min([tVert[k], tHorz[i - J]]) - np.max([tVert[k - 1], tHorz[i - (J + 1)]]))*hs[J + k]*memory_kernel(phis[J + k][0])))
                        
                        uOld = us[int(age[j] - age[j - i + J] - 1)]
                        uOldn = dot(uOld, n)
                        uOldnPos = (uOldn + abs(uOldn))/Constant(2.)
                        uOldnNeg = (uOldn - abs(uOldn))/Constant(2.)
                        
                        F0MCTadv -= Constant(dt)*inner(phis[j - i + J][int(age[j] - age[j - i + J])], div(outer(hs[j]*HM*psi, uOld)))*dx
                        F0MCTadv += Constant(dt)*inner(uOldnPos('+')*phis[j - i + J][int(age[j] - age[j - i + J])]('+') + uOldnNeg('+')*phis[j - i + J][int(age[j] - age[j - i + J])]('-'), jump(hs[j]*HM*psi))*dS
                        F0MCTadv += Constant(dt)*inner(uOldn*phis[j - i + J][int(age[j] - age[j - i + J])], hs[j]*HM*psi)*ds
                
                # Newton's method to solve for phi
                for ni in range(1, maxIter + 1):
                    # Terms of type (2): 0, ..., J
                    RHSMCT = Constant(lambdaC/dt)*(phi - phis[j][0])*psi*dx
                    if outerAdvection:
                        RHSMCT -= Constant(lambdaC)*inner(phi, div(outer(psi, u)))*dx
                        RHSMCT += Constant(lambdaC)*inner(unPos('+')*phi('+') + unNeg('+')*phi('-'), jump(psi))*dS
                        RHSMCT += Constant(lambdaC)*inner(un*phi, psi)*ds
                    
                    RHSMCT += phi*psi*dx
                    
                    RHSMCT += hs[j]*hs[j]*memory_kernel(phi)*(phis[J][int(age[j] - age[J])] - Constant(1.))*psi*dx
                    if innerAdvection:
                        for i in range(J + 1):
                            uOld = us[int(age[j] - age[i] - 1)]
                            uOldn = dot(uOld, n)
                            uOldnPos = (uOldn + abs(uOldn))/Constant(2.)
                            uOldnNeg = (uOldn - abs(uOldn))/Constant(2.)
                            
                            RHSMCT -= Constant(stepsize[i]*dt)*inner(phis[i][int(age[j] - age[i])], div(outer(hs[j]*hs[j]*memory_kernel(phi)*psi, uOld)))*dx
                            RHSMCT += Constant(stepsize[i]*dt)*inner(uOldnPos('+')*phis[i][int(age[j] - age[i])]('+') + uOldnNeg('+')*phis[i][int(age[j] - age[i])]('-'), jump(hs[j]*hs[j]*memory_kernel(phi)*psi))*dS
                            RHSMCT += Constant(stepsize[i]*dt)*inner(uOldn*phis[i][int(age[j] - age[i])], hs[j]*hs[j]*memory_kernel(phi)*psi)*ds
                    
                    # Terms of type (2): j, ..., j-J
                    HM.assign(Function(P))
                    for i in range(J + 1):
                        HM.assign(HM + projectScalar(Constant(stepsize[i]/np.sum(stepsize[0 : J + 1]))*hs[i]*memory_kernel(phis[i][0])))
                    RHSMCT += hs[j]*HM*(phi - phis[j-1][int(age[j] - age[j - 1])])*psi*dx
                    if innerAdvection:
                        RHSMCT -= Constant(np.sum(stepsize[0 : J + 1])*dt)*inner(phi, div(outer(hs[j]*HM*psi, u)))*dx
                        RHSMCT += Constant(np.sum(stepsize[0 : J + 1])*dt)*inner(unPos('+')*phi('+') + unNeg('+')*phi('-'), jump(hs[j]*HM*psi))*dS
                        RHSMCT += Constant(np.sum(stepsize[0 : J + 1])*dt)*inner(un*phi, hs[j]*HM*psi)*ds
                    
                    # Terms of type (3): J, ..., j-J 
                    RHSMCT += F0MCT*psi*dx
                    if innerAdvection:
                        RHSMCT += F0MCTadv
                    
                    LHSMCT = Constant(lambdaC/dt)*dphi_*psi*dx
                    if outerAdvection:
                        LHSMCT -= Constant(lambdaC)*inner(dphi_, div(outer(psi, u)))*dx
                        LHSMCT += Constant(lambdaC)*inner(unPos('+')*dphi_('+') + unNeg('+')*dphi_('-'), jump(psi))*dS
                        LHSMCT += Constant(lambdaC)*inner(un*dphi_, psi)*ds
                    
                    LHSMCT += dphi_*psi*dx
                    
                    LHSMCT += hs[j]*hs[j]*diff_memory_kernel(phi)*(phis[J][int(age[j] - age[J])] - Constant(1.))*dphi_*psi*dx
                    if innerAdvection:
                        # u . grad terms from bottom end
                        for i in range(J + 1):
                            uOld = us[int(age[j] - age[i] - 1)]
                            uOldn = dot(uOld, n)
                            uOldnPos = (uOldn + abs(uOldn))/Constant(2.)
                            uOldnNeg = (uOldn - abs(uOldn))/Constant(2.)
                            
                            LHSMCT -= Constant(stepsize[i]*dt)*inner(phis[i][int(age[j] - age[i])], div(outer(hs[j]*hs[j]*diff_memory_kernel(phi)*dphi_*psi, uOld)))*dx
                            LHSMCT += Constant(stepsize[i]*dt)*inner(uOldnPos('+')*phis[i][int(age[j] - age[i])]('+') + uOldnNeg('+')*phis[i][int(age[j] - age[i])]('-'), jump(hs[j]*hs[j]*diff_memory_kernel(phi)*dphi_*psi))*dS
                            LHSMCT += Constant(stepsize[i]*dt)*inner(uOldn*phis[i][int(age[j] - age[i])], hs[j]*hs[j]*diff_memory_kernel(phi)*dphi_*psi)*ds
                    
                    LHSMCT += hs[j]*HM*dphi_*psi*dx
                    if innerAdvection:
                        LHSMCT -= Constant(np.sum(stepsize[0 : J + 1])*dt)*inner(dphi_, div(outer(hs[j]*HM*psi, u)))*dx
                        LHSMCT += Constant(np.sum(stepsize[0 : J + 1])*dt)*inner(unPos('+')*dphi_('+') + unNeg('+')*dphi_('-'), jump(hs[j]*HM*psi))*dS
                        LHSMCT += Constant(np.sum(stepsize[0 : J + 1])*dt)*inner(un*dphi_, hs[j]*HM*psi)*ds
                    
                    MCT_mat = assemble(LHSMCT)
                    MCT_vec = assemble(-RHSMCT)
                    
                    correlatorsolver.set_operator(MCT_mat)
                    correlatorsolver.solve(dphi.vector(), MCT_vec)
                    
                    phi.assign(phi + dphi)
                    
                    dphiL2 = np.sqrt(assemble(dphi*dphi*dx)) 
                    phiL2 = np.sqrt(assemble(phi*phi*dx))
                    
                    # Print convergence history
                    print(   f'{ni:10.0f}', '|', f'{dphiL2:13.2e}', '|', f'{phiL2:13.2e}')
                    
                    if dphiL2 <= absTol or dphiL2 <= relTol*phiL2:
                        # Newton's method has converged
                        break
                    elif ni == maxIter:
                        # Newton's method failed to converge
                        print('(WARNING) No convergence after', maxIter, 'iterations.')                
           
            # Update this correlator
            phis[j][0].assign(phi)
        
    # Stress integral (approximated with DG(0) in age a)
    for j in range(Nh):
        if j == 0:
            # The Finger tensor of age a = 0 is the identity tensor and the correlator phi = 1
            assign(tau, projectTensor(Constant(GInf)*phis[0][0]**Constant(2.)*(Bs[0] - I)))
        else:
            assign(tau, projectTensor(tau + Constant(GInf)*phis[j][0]**Constant(2.)*(Bs[j] - Bs[j - 1])))
    
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
    for i in range(Ntprime - 1, 0, -1):
        us[i].assign(us[i - 1])
    us[0].assign(u)
