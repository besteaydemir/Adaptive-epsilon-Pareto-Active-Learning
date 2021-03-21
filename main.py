import numpy as np
import gpflow as gpf

from AdaptiveEpsilonPAL import AdaptiveEpsilonPAL
from OptimizationProblem import OptimizationProblem
from GaussianProcessModel import GaussianProcessModel
from Hypercube import Hypercube


# Generate the function lists with synthetic data sets
func1 = lambda x: 2 * np.sin(np.pi * x[:,0]) * np.sin(np.pi * x[:,1]) + 4 * np.sin(2 * np.pi * x[:,0]) * np.sin(2 * np.pi * x[:,1])
func2 = lambda x: 2 * np.sin(np.pi * x[:,0]) * np.sin(np.pi * x[:,1]) - 6 * np.sin(2 * np.pi * x[:,0]) * np.sin(2 * np.pi * x[:,1])
func_list = [func1, func2]
problem_model = OptimizationProblem(func_list)


# Generate data
data = np.random.randn(40, 2)  # Can be generated with opt problem instance for syn. data
y = problem_model.observe(data, std=2)


# Specify kernel and mean function for GP prior
kernel_list = [gpf.kernels.SquaredExponential() for _ in range(2)]
gp = GaussianProcessModel(data, y, multi=False, m=2, kernel_list=kernel_list, verbose=True)


# Adaptive Epsilon PAL algorithm
pareto_set, pareto_set_cells = AdaptiveEpsilonPAL(problem_model, epsilon=0.10, delta=0.05, gp=gp,
                                                  initial_hypercube=Hypercube(1, (0.5, 0.5))).algorithm()

print(pareto_set, pareto_set_cells)
