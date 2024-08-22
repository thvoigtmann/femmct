import numpy as np
import dolfin as fe
from dolfin import tr, inner, outer, div, dot, jump, dS

def default_parameters ():
    return {
      'absTol': 1e-9,
      'relTol': 1e-6,
      'maxIter': 25,
      'outerAdvection': True,
      'innerAdvection': True
    }

class F12Model(object):
    def __init__(self, v1=0., v2=4.1, Ginf=1.0, lambdaC=1.0, gammaC=0.1,
                 parameters = default_parameters()):
        self.GInf = fe.Constant(Ginf)
        self.lambdaC = lambdaC
        self.gammaC = gammaC
        self.v1 = fe.Constant(v1)
        self.v2 = fe.Constant(v2)
        self.dv2 = fe.Constant(2.*v2)
        self.param = parameters

    def memory_kernel(self, phi):
        return self.v1*phi + self.v2*phi*phi
    def diff_memory_kernel(self, phi):
        return self.v1 + self.dv2*phi

    def adjust (self, Nt, Na, Nb):
        # Parts of the following code assume that the fine end of the lin-log
        # mesh resolves intervals of width dt, 2dt, 4dt, ..., (2**(Nb-1))dt
        # exactly. We adjust the user input for Na to make sure this condition
        # is met.
        # The smallest power of 2 such that 2**l >= Na:
        if Nb > 1:
            l = int(np.ceil(np.log2(Na)))
            for k in range(l, Nb):
                # number of blocks that fully fit into 2**(Nb-1) when Na = 2**k
                L = int(np.log2(1 + 2**(Nb - 1 - k))) - 1
                # There may be a gap left between the full blocks 0, ..., L
                # and the big interval 2**(Nb-1) - we must be able to fill this
                # gap with an integer number of steps of length 2**(L+1) or
                # else the meshes don't match as required
                if np.mod(2**(Nb-1) - (2**(L+1) - 1)*2**k, 2**(L+1)) == 0:
                    # meshes match
                    l = k
                    break
    
            if Na != 2**l:
                print('(INFO) Adjusting the number of subintervals in each block from', Na, 'to', 2**l, '.')
                Na = 2**l
        Ntprime = int((2**Nb - 1)*Na)
        if Ntprime != Nt:
            print('(INFO) Adjusting Nt from',Nt,'to',Ntprime)
        return Ntprime, Na, Nb

    def initialize (self, solver):
        # stepsizes on the lin-log grid in multiples of dt
        self.stepsize = np.ones(solver.Nh)
        for l in range(1, solver.Nb):
            self.stepsize[l*solver.Na : (l+1)*solver.Na] = 2**l

        # age on lin-log grid in multiples of dt
        self.age = np.ones(solver.Nh)
        for l in range(solver.Nb):
            for m in range(solver.Na):
                self.age[l*solver.Na + m] = (2**l - 1)*solver.Na + 2**l*(m+1)

        self.hs = []
        self.phis = []

        if self.param['innerAdvection']:
            self.us = []
            for i in range(solver.Nt):
                self.us.append(fe.Function(solver.fn.U, name='Velocity'))

        gamma_c_sq = fe.Constant(self.gammaC**2.)
        for i in range(solver.Nh):
            # initialise shear factors
            self.hs.append(fe.Function(solver.fn.P))
            self.hs[i].assign(solver.fn.projectScalar(gamma_c_sq/(gamma_c_sq + tr(solver.Bs[i]) - fe.Constant(2.))))
            # initialize correlators using exp-shorttime expansion
            self.phis.append([])
            l = int(np.floor(i/solver.Na))
            m = i - l*solver.Na
            a = ((2**l - 1)*solver.Na + 2**l*(m + 1))*solver.dt # age of correlators on diag.
            for j in range((2**solver.Nb - 2**l)*solver.Na - 2**l*(m + 1) + 1):
                # this is the length of the constant-age diagonal
                # on the lin-log grid
                # j = 0 corresponds to the current time t,
                # larger j to past times t - j*dt
                self.phis[i].append(fe.Function(solver.fn.P))
                self.phis[i][j].assign(fe.interpolate(fe.Constant(np.exp(-a/self.lambdaC)), solver.fn.P))
        self.phi0 = fe.Constant(1.)

        self.phi = fe.Function(solver.fn.P)
        self.psi = fe.TestFunction(solver.fn.P)
        self.FMCT = fe.Function(solver.fn.P)
        self.F0MCT = fe.Function(solver.fn.P)
        self.HM = fe.Function(solver.fn.P)
        self.DFMCT = fe.Function(solver.fn.P)
        self.dphi = fe.Function(solver.fn.P)
        self.dphi_ = fe.TrialFunction(solver.fn.P)
        self.correlatorsolver = fe.LUSolver()

    def step (self, solver, k, u, un, unPos, unNeg, tau):
        dx, ds, dt = solver.dx, solver.ds, solver.dt
        gamma_c_sq = fe.Constant(self.gammaC**2.)
        hs = self.hs
        phis = self.phis
        phi, psi = self.phi, self.psi
        dphi, dphi_ = self.dphi, self.dphi_
        maxIter = self.param['maxIter']
        absTol = self.param['absTol']
        relTol = self.param['relTol']
        outerAdvection = self.param['outerAdvection']
        innerAdvection = self.param['innerAdvection']

        for l in range(solver.Nb - 1, -1, -1):
            da = 2**l*dt
            for m in range(solver.Na -1, -1, -1):
                j = l*solver.Na + m
                hs[j].assign(solver.fn.projectScalar(gamma_c_sq/(gamma_c_sq + tr(solver.Bs[j]) - fe.Constant(2.))))

        for i in range(solver.Nh - 1):
            l = int(np.floor(i/solver.Na)) # block index between 0 and Nb-1
            m = i - l*solver.Na # index within block between 0 and Na-1
            for j in range((2**solver.Nb - 2**l)*solver.Na - 2**l*(m + 1), 0, -1):
                # move all correlators back by one time step
                phis[i][j].assign(phis[i][j-1])

        for l in range(solver.Nb):
            da = 2**l*dt
            for m in range(solver.Na):
                # calculate correlators at the current time
                # proceedings downwards (increasing age)
                j = l*solver.Na + m
                # distinguish three types of terms in the disc. MCT eqs.
                # (1) terms where current unknown phi is referenced in t AND t'
                # (2) unknown phi referenced EITHER in t OR in t'
                # (3) unknown phi not referenced
                if j == 0:
                    # type (1) only arises in integral for youngest correlator
                    # which is phi[0][0]
                    phi.assign(phis[0][0])
                    print ('   Computing phi[ 0 ]...')
                    print ('      Iter |    ||dphi||   |   ||phi||')
                    # Newton's method to solve for phi
                    for ni in range(1, maxIter + 1):
                        # nearest younger correlator of age a = 0 is 1
                        # TODO without any advection:
                        #FMCT.assign(projectScalar(Constant(lambdaC/dt)*(phi - Constant(1.)) + phi + hs[0]*hs[0]*memory_kernel(phi)*(phi - Constant(1.))))
                        #DFMCT.assign(projectScalar(Constant(lambdaC/dt + 1.) + hs[0]*hs[0]*memory_kernel(phi) + hs[0]*hs[0]*diff_memory_kernel(phi)*(phi - Constant(1.))))
                        #dphi.assign(projectScalar(-FMCT/DFMCT))
                        #
                        # code with MCT advection:
                        RHSMCT = fe.Constant(self.lambdaC/dt)*(phi - self.phi0)*psi*dx
                        if outerAdvection:
                            RHSMCT -= fe.Constant(self.lambdaC)*inner(phi, div(outer(psi, u)))*dx
                            RHSMCT += fe.Constant(self.lambdaC)*inner(unPos('+')*phi('+') + unNeg('+')*phi('-'), fe.jump(psi))*dS
                            RHSMCT += fe.Constant(self.lambdaC)*inner(un*phi, psi)*ds
                        RHSMCT += phi*psi*dx
                        RHSMCT += hs[0]*hs[0]*self.memory_kernel(phi)*(phi - self.phi0)*psi*dx
                        if innerAdvection:
                            RHSMCT -= fe.Constant(dt)*inner(phi, div(outer(hs[0]*hs[0]*self.memory_kernel(phi)*psi, u)))*dx
                            RHSMCT += fe.Constant(dt)*inner(unPos('+')*phi('+') + unNeg('+')*phi('-'), jump(hs[0]*hs[0]*self.memory_kernel(phi)*psi))*dS
                            RHSMCT += fe.Constant(dt)*inner(un*phi, hs[0]*hs[0]*self.memory_kernel(phi)*psi)*ds

                        LHSMCT = fe.Constant(self.lambdaC/dt)*dphi_*psi*dx
                        if outerAdvection:
                            LHSMCT -= fe.Constant(self.lambdaC)*inner(dphi_, div(outer(psi, u)))*dx
                            LHSMCT += fe.Constant(self.lambdaC)*inner(unPos('+')*dphi_('+') + unNeg('+')*dphi_('-'), jump(psi))*dS
                            LHSMCT += fe.Constant(self.lambdaC)*inner(un*dphi_, psi)*ds
                        LHSMCT += dphi_*psi*dx
                        LHSMCT += hs[0]*hs[0]*(self.memory_kernel(phi) + self.diff_memory_kernel(phi)*(phi - self.phi0))*dphi_*psi*dx
                        if innerAdvection:
                            LHSMCT -= fe.Constant(dt)*inner(dphi_, div(outer(hs[0]*hs[0]*self.memory_kernel(phi)*psi, u)))*dx
                            LHSMCT += fe.Constant(dt)*inner(unPos('+')*dphi_('+') + unNeg('+')*dphi_('-'), jump(hs[0]*hs[0]*self.memory_kernel(phi)*psi))*dS
                            LHSMCT += fe.Constant(dt)*inner(un*dphi_, hs[0]*hs[0]*self.memory_kernel(phi)*psi)*ds
                            LHSMCT -= fe.Constant(dt)*inner(phi, div(outer(hs[0]*hs[0]*self.diff_memory_kernel(phi)*dphi_*psi, u)))*dx
                            LHSMCT += fe.Constant(dt)*inner(unPos('+')*phi('+') + unNeg('+')*phi('-'), jump(hs[0]*hs[0]*self.diff_memory_kernel(phi)*dphi_*psi))*dS
                            LHSMCT += fe.Constant(dt)*inner(un*phi, hs[0]*hs[0]*self.diff_memory_kernel(phi)*dphi_*psi)*ds
                        MCTmat = fe.assemble(LHSMCT)
                        MCTvec = fe.assemble(-RHSMCT)
                        self.correlatorsolver.set_operator(MCTmat)
                        self.correlatorsolver.solve(dphi.vector(), MCTvec)
                        # end of MCT advection code
                        phi.assign(phi + dphi)

                        dphiL2 = np.sqrt(fe.assemble(dphi*dphi*dx)) 
                        phiL2 = np.sqrt(fe.assemble(phi*phi*dx))

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
                    L = int(np.ceil(np.log2(1 + 2**l/solver.Na))) - 1
                    M = int((2**l - (2**L - 1)*solver.Na)/(2**L)) - 1
                    J = L*solver.Na + M

                    # Terms of type (3): steps J, ..., j-J 
                    self.F0MCT.assign(solver.fn.projectScalar(fe.Constant(self.lambdaC/da)*(phis[j][0] - phis[j - 1][1])))

                    if innerAdvection:
                        # Reset advection terms of type(3)
                        F0MCTadv = fe.Constant(0.)*psi*dx

                    tVert = self.age[J:j]-self.age[J] # lin-log grid points downwards counting from index J to j-1 in multiples of dt
                    if J > 0:
                        tHorz = self.age[j-1]-self.age[j-1:J-1:-1] # lin-log grid points leftwards counting from index j-1 in multiples of dt
                    else:
                        tHorz = self.age[j-1]-self.age[j-1::-1] # NB: age[j-1:-1:-1] would always be empty so we have to index the case J = 0 differently

                    for i in range(J + 1, j):
                        # We iterate over vertical (age) intervals and have to find the corresponding projected grid points on the horizontal (time) axis
                        # On the vertical age axis, the i-th interval begins at tVert[i - (J + 1)] and ends at tVert[i - J].
                        # On the horizontal time axis, we will identify
                        #   * the largest grid point <= tVert[i - J] in leftBound together with a weight for interpolation between this grid point and its right neighbour (w*point + (1-w)*neighbour)
                        #   * the smallest grid point >= tVert[i - (J + 1)] in rightBound together with a weight for interpolation between this grid point and its left neighbour (w*point + (1-w)*neighbour)
                        leftTime = next(left for left in tHorz if left >= tVert[i - J])
                        leftBound = j - 1 - np.argmax(tHorz == leftTime) # NB: this is the index for the age array
                        leftWeight = 1 - ((self.age[j] - self.age[i]) - self.age[leftBound])/(self.age[leftBound + 1] - self.age[leftBound])

                        rightTime = next(right for right in np.flip(tHorz) if right <= tVert[i - (J + 1)])
                        rightBound = j - 1 - np.argmax(tHorz == rightTime) # NB: this is the index for the age array
                        rightWeight = 1 - ((self.age[j] - self.age[i - 1]) - self.age[rightBound])/(self.age[rightBound - 1] - self.age[rightBound])

                        self.F0MCT.assign(self.F0MCT + solver.fn.projectScalar(hs[j]*hs[i]*self.memory_kernel(phis[i][0])*((fe.Constant(rightWeight)*phis[rightBound][int(self.age[j] - self.age[rightBound])] + fe.Constant(1. - rightWeight)*phis[rightBound - 1][int(self.age[j] - self.age[rightBound - 1])]) - (fe.Constant(leftWeight)*phis[leftBound][int(self.age[j] - self.age[leftBound])] + fe.Constant(1. - leftWeight)*phis[leftBound + 1][int(self.age[j] - self.age[leftBound + 1])]))))

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

                            self.HM.assign(fe.Function(solver.fn.P))

                            for k in range(topBound, bottomBound + 1):
                                # each h*m contribution spans the time interval from tVert[k - 1] to tVert[k], intersected with the interval from tHorz[i - J] to tHorz[i - (J + 1)]
                                self.HM.assign(self.HM + solver.fn.projectScalar(fe.Constant(np.min([tVert[k], tHorz[i - J]]) - np.max([tVert[k - 1], tHorz[i - (J + 1)]]))*hs[J + k]*self.memory_kernel(phis[J + k][0])))

                            uOld = self.us[int(self.age[j] - self.age[j - i + J] - 1)]
                            uOldn = dot(uOld, solver.n)
                            uOldnPos = (uOldn + abs(uOldn))/fe.Constant(2.)
                            uOldnNeg = (uOldn - abs(uOldn))/fe.Constant(2.)

                            F0MCTadv -= fe.Constant(dt)*inner(phis[j - i + J][int(self.age[j] - self.age[j - i + J])], div(outer(hs[j]*self.HM*psi, uOld)))*dx
                            F0MCTadv += fe.Constant(dt)*inner(uOldnPos('+')*phis[j - i + J][int(self.age[j] - self.age[j - i + J])]('+') + uOldnNeg('+')*phis[j - i + J][int(self.age[j] - self.age[j - i + J])]('-'), jump(hs[j]*self.HM*psi))*dS
                            F0MCTadv += fe.Constant(dt)*inner(uOldn*phis[j - i + J][int(self.age[j] - self.age[j - i + J])], hs[j]*self.HM*psi)*ds

                    # Newton's method to solve for phi
                    for ni in range(1, maxIter + 1):
                        # Terms of type (2): 0, ..., J
                        # no MCT advection:
                        #FMCT.assign(projectScalar(Constant(lambdaC/dt)*(phi - phis[j][0]) + phi + hs[j]*hs[j]*memory_kernel(phi)*(phis[J][int(age[j] - age[J])] - Constant(1.))))
                        #DFMCT.assign(projectScalar(Constant(lambdaC/dt + 1.) + hs[j]*hs[j]*diff_memory_kernel(phi)*(phis[J][int(age[j] - age[J])] - Constant(1.))))

                        # MCT advection:
                        RHSMCT = fe.Constant(self.lambdaC/dt)*(phi - phis[j][0])*psi*dx
                        if outerAdvection:
                            RHSMCT -= fe.Constant(self.lambdaC)*inner(phi, div(outer(psi, u)))*dx
                            RHSMCT += fe.Constant(self.lambdaC)*inner(unPos('+')*phi('+') + unNeg('+')*phi('-'), jump(psi))*dS
                            RHSMCT += fe.Constant(self.lambdaC)*inner(un*phi, psi)*ds

                        RHSMCT += phi*psi*dx

                        RHSMCT += hs[j]*hs[j]*self.memory_kernel(phi)*(phis[J][int(self.age[j] - self.age[J])] - fe.Constant(1.))*psi*dx
                        if innerAdvection:
                            for i in range(J + 1):
                                uOld = self.us[int(self.age[j] - self.age[i] - 1)]
                                uOldn = dot(uOld, solver.n)
                                uOldnPos = (uOldn + abs(uOldn))/fe.Constant(2.)
                                uOldnNeg = (uOldn - abs(uOldn))/fe.Constant(2.)
                                
                                RHSMCT -= fe.Constant(self.stepsize[i]*dt)*inner(phis[i][int(self.age[j] - self.age[i])], div(outer(hs[j]*hs[j]*self.memory_kernel(phi)*psi, uOld)))*dx
                                RHSMCT += fe.Constant(self.stepsize[i]*dt)*inner(uOldnPos('+')*phis[i][int(self.age[j] - self.age[i])]('+') + uOldnNeg('+')*phis[i][int(self.age[j] - self.age[i])]('-'), jump(hs[j]*hs[j]*self.memory_kernel(phi)*psi))*dS
                                RHSMCT += fe.Constant(self.stepsize[i]*dt)*inner(uOldn*phis[i][int(self.age[j] - self.age[i])], hs[j]*hs[j]*self.memory_kernel(phi)*psi)*ds
                        # end MCT advection

                        # Terms of type (2): j, ..., j-J
                        self.HM.assign(fe.Function(solver.fn.P))
                        for i in range(J + 1):
                            self.HM.assign(self.HM + solver.fn.projectScalar(fe.Constant(self.stepsize[i]/np.sum(self.stepsize[0 : J + 1]))*hs[i]*self.memory_kernel(phis[i][0])))
                        # no MCT advection:
                        #FMCT.assign(FMCT + projectScalar(hs[j]*HM*(phi - phis[j-1][int(age[j] - age[j - 1])])))
                        #DFMCT.assign(DFMCT + projectScalar(hs[j]*HM))
                        #dphi.assign(projectScalar(-(FMCT + F0MCT)/DFMCT))
                        # MCT advection:
                        RHSMCT += hs[j]*self.HM*(phi - phis[j-1][int(self.age[j] - self.age[j - 1])])*psi*dx
                        if innerAdvection:
                            RHSMCT -= fe.Constant(np.sum(self.stepsize[0 : J + 1])*dt)*inner(phi, div(outer(hs[j]*self.HM*psi, u)))*dx
                            RHSMCT += fe.Constant(np.sum(self.stepsize[0 : J + 1])*dt)*inner(unPos('+')*phi('+') + unNeg('+')*phi('-'), jump(hs[j]*self.HM*psi))*dS
                            RHSMCT += fe.Constant(np.sum(self.stepsize[0 : J + 1])*dt)*inner(un*phi, hs[j]*self.HM*psi)*ds

                        # Terms of type (3): J, ..., j-J 
                        RHSMCT += self.F0MCT*psi*dx
                        if innerAdvection:
                            RHSMCT += F0MCTadv

                        LHSMCT = fe.Constant(self.lambdaC/dt)*dphi_*psi*dx
                        if outerAdvection:
                            LHSMCT -= fe.Constant(self.lambdaC)*inner(dphi_, div(outer(psi, u)))*dx
                            LHSMCT += fe.Constant(self.lambdaC)*inner(unPos('+')*dphi_('+') + unNeg('+')*dphi_('-'), jump(psi))*dS
                            LHSMCT += fe.Constant(self.lambdaC)*inner(un*dphi_, psi)*ds

                        LHSMCT += dphi_*psi*dx

                        LHSMCT += hs[j]*hs[j]*self.diff_memory_kernel(phi)*(phis[J][int(self.age[j] - self.age[J])] - fe.Constant(1.))*dphi_*psi*dx
                        if innerAdvection:
                            # u . grad terms from bottom end
                            for i in range(J + 1):
                                uOld = self.us[int(self.age[j] - self.age[i] - 1)]
                                uOldn = dot(uOld, solver.n)
                                uOldnPos = (uOldn + abs(uOldn))/fe.Constant(2.)
                                uOldnNeg = (uOldn - abs(uOldn))/fe.Constant(2.)

                                LHSMCT -= fe.Constant(self.stepsize[i]*dt)*inner(phis[i][int(self.age[j] - self.age[i])], div(outer(hs[j]*hs[j]*self.diff_memory_kernel(phi)*dphi_*psi, uOld)))*dx
                                LHSMCT += fe.Constant(self.stepsize[i]*dt)*inner(uOldnPos('+')*phis[i][int(self.age[j] - self.age[i])]('+') + uOldnNeg('+')*phis[i][int(self.age[j] - self.age[i])]('-'), jump(hs[j]*hs[j]*self.diff_memory_kernel(phi)*dphi_*psi))*dS
                                LHSMCT += fe.Constant(self.stepsize[i]*dt)*inner(uOldn*phis[i][int(self.age[j] - self.age[i])], hs[j]*hs[j]*self.diff_memory_kernel(phi)*dphi_*psi)*ds
                        
                        LHSMCT += hs[j]*self.HM*dphi_*psi*dx
                        if innerAdvection:
                            LHSMCT -= fe.Constant(np.sum(self.stepsize[0 : J + 1])*dt)*inner(dphi_, div(outer(hs[j]*self.HM*psi, u)))*dx
                            LHSMCT += fe.Constant(np.sum(self.stepsize[0 : J + 1])*dt)*inner(unPos('+')*dphi_('+') + unNeg('+')*dphi_('-'), jump(hs[j]*self.HM*psi))*dS
                            LHSMCT += fe.Constant(np.sum(self.stepsize[0 : J + 1])*dt)*inner(un*dphi_, hs[j]*self.HM*psi)*ds

                        MCT_mat = fe.assemble(LHSMCT)
                        MCT_vec = fe.assemble(-RHSMCT)

                        self.correlatorsolver.set_operator(MCT_mat)
                        self.correlatorsolver.solve(dphi.vector(), MCT_vec)
                        # end MCT advection

                        phi.assign(phi + dphi)

                        dphiL2 = np.sqrt(fe.assemble(dphi*dphi*dx)) 
                        phiL2 = np.sqrt(fe.assemble(phi*phi*dx))

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
        # stress integral, DG0 approximation in age a
        for j in range(solver.Nh):
            if j == 0:
                fe.assign(tau, solver.fn.projectTensor(self.GInf*phis[0][0]**fe.Constant(2.)*(solver.Bs[0] - solver.Bs0)))
            else:
                fe.assign(tau, solver.fn.projectTensor(tau + self.GInf*phis[j][0]**fe.Constant(2.)*(solver.Bs[j] - solver.Bs[j-1])))

    def post_step(self, solver, u):
        if self.us:
            for i in range(solver.Nt - 1, 0, -1):
                self.us[i].assign(self.us[i - 1])
            self.us[0].assign(u)
