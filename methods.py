# Gaussian process
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal

# Define the GP model
class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean = ConstantMean()
        self.covariance = ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[1]))
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean(x)
        covar_x = self.covariance(x)
        return MultivariateNormal(mean_x, covar_x)