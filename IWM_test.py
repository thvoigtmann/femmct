import dolfin
import femmct

path = 'iwmtest'

# make mesh

# channel length and height, and number of elements per direction
length = 5.
height = 1.
Nx = 50
Ny = 50

# initial pressure gradient
deltap = 2.5

mesh = dolfin.RectangleMesh(dolfin.Point(0.,0.), dolfin.Point(length, height), Nx, Ny, diagonal='crossed')

class InFlow(dolfin.SubDomain):
  def inside(self, x, on_boundary):
    return (on_boundary and dolfin.near(x[0],0.))
class OutFlow(dolfin.SubDomain):
  def inside(self, x, on_boundary):
    return (on_boundary and dolfin.near(x[0],length))
class TopWall(dolfin.SubDomain):
  def inside(self, x, on_boundary):
    return (on_boundary and dolfin.near(x[1],height))
class BottomWall(dolfin.SubDomain):
  def inside(self, x, on_boundary):
    return (on_boundary and dolfin.near(x[1],0.))

# values on the inlet shall be overwritten with values on the outlet
class PeriodicBoundary(dolfin.SubDomain):
  def inside(self, x, on_boundary):
    return (on_boundary and dolfin.near(x[0],0.))
  def map(self, x, y):
    y[0] = x[0] - length
    y[1] = x[1]

inflow = InFlow()
outflow = OutFlow()
topwall = TopWall()
bottomwall = BottomWall()

# make a boundaries function that has integer indices for the boundaries
# that we have defined
boundaries = dolfin.MeshFunction('size_t', mesh, 1)
boundaries.set_all(0)
inflow.mark(boundaries, 1)
outflow.mark(boundaries, 2)
topwall.mark(boundaries, 3)
bottomwall.mark(boundaries, 4)

boundary_conditions = {
  1: femmct.p_inlet (deltap),
  2: femmct.p_outlet (0.),
  3: femmct.dirichlet_u ((0.,0.)),
  4: femmct.dirichlet_u ((0.,0.))
}

solver = femmct.StokesSolver(mesh, constrained_domain = PeriodicBoundary(), subdomain_data = boundaries, boundary_conditions = boundary_conditions)


def switchoff_dp (solver_obj, step):
    if step == int(solver_obj.Nt/2):
      solver_obj.apply_bc ({1: femmct.p_inlet (0.)})

solver.callback = switchoff_dp

model = femmct.IntegralWhiteMetznerModel (Ginf=1.0, lambdaC=10., gammaC=0.1)
#solver.initialize (model = model, T = 50., Nt = 500, Na = 16, Nb = 6)
solver.initialize (model = model, T = 5., Nt = 50, Na = 16, Nb = 6)

solver.create_files(path)

solver.loop()
