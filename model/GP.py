import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, ConstantKernel, MaternKernel, CosineKernel, CylindricalKernel, PeriodicKernel, SpectralMixtureKernel, SpectralDeltaKernel, RQKernel, PolynomialKernelGrad, PolynomialKernel, LinearKernel, PiecewisePolynomialKernel, ArcKernel, RBFKernelGrad
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ExactGP
from gpytorch.constraints import Interval


class GPRegressionModel(ExactGP):
    def __init__(self, kernel, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        
        # alpha_constraint = Interval(0.0, alpha)
        
        self.covar_module = ScaleKernel(PeriodicKernel())   # PeriodicKernel()
        self.shift_module = ConstantKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x) + self.shift_module(x)
        return MultivariateNormal(mean_x, covar_x)
    

