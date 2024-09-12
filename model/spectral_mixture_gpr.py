import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, ConstantKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ExactGP


class SpectralMixtureGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, device):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean().to(device)
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=1, batch_size=1, ard_num_dims=384).to(device)
        self.covar_module.initialize_from_data(train_x, train_y)
        self.shift_module = ConstantKernel()

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x) + self.shift_module(x)
        import pdb; pdb.set_trace()
        return MultivariateNormal(mean_x, covar_x)