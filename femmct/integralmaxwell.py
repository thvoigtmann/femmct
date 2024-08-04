import dolfin

class IntegralWhiteMetznerModel(object):
    def __init__(self, Ginf=1.0, lambdaC = 10., gammaC = .1):
        self.GInf = Ginf
        self.lambdaC = lambdaC
        self.gammaC = gammaC
