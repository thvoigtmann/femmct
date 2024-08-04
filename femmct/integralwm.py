import numpy as np
import dolfin

class IntegralWhiteMetznerModel(object):
    def __init__(self, Ginf=1.0, lambdaC = 10., gammaC = .1):
        self.GInf = Ginf
        self.lambdaC = lambdaC
        self.gammaC = gammaC
    def adjust (self, Nt, Na, Nb):
        return Nt, Na, Nb
    def initialize (self, solver):
        # age on lin-log grid in multiples of dt
        age = np.ones(solver.Nh)
        for l in range(solver.Nb):
            for m in range(solver.Na):
                age[l*solver.Na + m] = (2**l - 1)*solver.Na + 2**l*(m+1)
        self.Gs = []
        for i in range(solver.Nh):
            self.Gs.append(dolfin.Function(solver.fn.P))
            self.Gs[i].assign(dolfin.interpolate(dolfin.Constant(self.GInf*np.exp(-age[i]*solver.dt/self.lambdaC)), solver.fn.P))
        self.Gs0 = dolfin.Constant(self.GInf)

        self.shearmodulussolver = dolfin.LUSolver('default')
        self.G_ = dolfin.TrialFunction(solver.fn.P)
        self.H = dolfin.TestFunction(solver.fn.P)

    def step (self, solver, k, u, un, unPos, unNeg, tau):
        dx, ds, dt = solver.dx, solver.ds, solver.dt
        G_, H = self.G_, self.H
        Gs = self.Gs
        lambdaInv = dolfin.Constant(1./self.lambdaC) + dolfin.Constant(np.sqrt(2.)/self.gammaC)*dolfin.sqrt(dolfin.inner(dolfin.sym(dolfin.grad(u)), dolfin.sym(dolfin.grad(u))))
        LHSSM = dolfin.Constant(1./dt)*dolfin.inner(G_,H)*dx
        LHSSM -= dolfin.inner(G_, dolfin.div(dolfin.outer(H, u)))*dx
        LHSSM += dolfin.inner(unPos('+')*G_('+') + unNeg('+')*G_('-'), dolfin.jump(H))*dolfin.dS
        LHSSM += dolfin.inner(un*G_, H)*ds
        LHSSM += dolfin.inner(lambdaInv*G_, H)*dx
        SMmat = dolfin.assemble(LHSSM)
        self.shearmodulussolver.set_operator(SMmat)

        for l in range(solver.Nb - 1, -1, -1):
          da = 2**l*dt
          for m in range(solver.Na - 1, -1, -1):
            j = l*solver.Na + m
            RHSSM = dolfin.Constant(1./dt)*dolfin.inner(Gs[j], H)*dx
            if j == 0:
                RHSSM -= dolfin.Constant(1./da)*dolfin.inner(Gs[0] - self.Gs0, H)*dx
            else:
                RHSSM -= dolfin.Constant(1./da)*dolfin.inner(Gs[j] - Gs[j-1], H)*dx
            SMvec = dolfin.assemble(RHSSM)
            self.shearmodulussolver.solve (Gs[j].vector(), SMvec)

        # stress integral, DG0 approximation in age a
        for j in range(solver.Nh):
            if j == 0:
                dolfin.assign(tau, solver.fn.projectTensor(Gs[0]*(solver.Bs[0] - solver.Bs0)))
            else:
                dolfin.assign(tau, solver.fn.projectTensor(tau + Gs[j]*(solver.Bs[j] - solver.Bs[j-1])))

    def post_step (self, solver, u):
        return
