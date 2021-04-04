import numpy as np
import gpflow as gpf

from AdaptiveEpsilonPAL import AdaptiveEpsilonPAL
from OptimizationProblem import OptimizationProblem
from GaussianProcessModel import GaussianProcessModel
from Hypercube import Hypercube

from utils_plot import plot_func_list, plot_pareto_front
from utils import printl

from paretoset import paretoset
import pandas as pd

np.random.seed(7)
# Generate the function lists
func1 = lambda x: (2 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) + 4 * np.sin(2 * np.pi * x[:, 0]) * np.sin(
    2 * np.pi * x[:, 1]))/4
func2 = lambda x: (2 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) - 6 * np.sin(2 * np.pi * x[:, 0]) * np.sin(
    2 * np.pi * x[:, 1]))/6
# func1 = lambda x: x[:, 0] ** 2 - (x[:, 1])
# func2 = lambda x: (x[:, 1])**3 + x[:, 0] ** 2
func_list = [func1, func2]
problem_model = OptimizationProblem(func_list)


# Visualize the functions (two functions)
title1 = "$2sin(\pi x_1)sin(\pi x_2) + 4sin(2 \pi x_1)sin(2 \pi x_2)$"
title2 = "$2sin(\pi x_1)sin(\pi x_2) - 6sin(2 \pi x_1)sin(2 \pi x_2)$"
func_val1, func_val2 = plot_func_list(func_list, (-1, 1), (-1, 1), title1, title2)

# Plot pareto front (two functions)
hotels = pd.DataFrame({"price": func_val1, "distance_to_beach": func_val2})
mask = paretoset(hotels, sense=["max", "max"])
plot_pareto_front(func_val1, func_val2, mask)


# Generate synthetic data
data = np.random.uniform(low=-1, high=1, size=(40, 2))  # Can be generated with opt problem instance for syn. data
y = problem_model.observe(data, std=0)


# Specify kernel and mean function for GP prior
#kernel_list = [gpf.kernels.Periodic(gpf.kernels.SquaredExponential()) for _ in range(2)] # lengthscales=[0.5, 0.5]
kernel_list = [gpf.kernels.Periodic(gpf.kernels.SquaredExponential(lengthscales=[0.1, 0.1])) for _ in range(2)] # lengthscales=[0.5, 0.5]
gp = GaussianProcessModel(data, y, multi=False, periodic=True, m=2, kernel_list=kernel_list, verbose=True)


# Adaptive Epsilon PAL algorithm
pareto_set, pareto_set_cells = AdaptiveEpsilonPAL(problem_model, epsilon=5, delta=0.10, gp=gp,
                                                  initial_hypercube=Hypercube(2, (0, 0))).algorithm()

# Print nodes in the Pareto set
printl(pareto_set)

# Get the center of each node in the Pareto set and plot after observing
pareto_nodes_center = [node.get_center() for node in pareto_set]

# Print the cell centers of the the Pareto node cells
# print([[cell.get_center() for cell in cells] for cells in pareto_set_cells])
# print(np.array([[cell.get_center() for cell in cells] for cells in pareto_set_cells]))
# print(np.array(pareto_nodes_center))



#data_alg = np.array([[-0.625, -0.375], [0.1875, 0.8125], [0.1875, 0.9375], [0.375, 0.625], [-0.9375, -0.6875], [-0.9375, -0.5625], [-0.6875, -0.6875], [-0.6875, -0.5625]])
#data_alg2 = np.array([[-0.6875, -0.5625], [-0.1875, -0.8125], [-0.3125, -0.3125], [0.4375, 0.6875], [-0.5625, -0.5625], [0.6875, 0.4375], [0.5625, 0.5625],[0.3125, 0.3125], [0.4375, 0.3125],[0.5625, 0.4375], [-0.4375, -0.3125], [0.4375, 0.5625], [-0.4375, -0.5625], [0.4375, 0.4375], [-0.4375, -0.6875], [-0.3125, -0.6875],[-0.6875, -0.6875], [0.6875, 0.5625], [0.6875, 0.3125], [0.6875, 0.6875], [-0.6875, -0.3125], [0.3125, 0.6875], [-0.5625, -0.6875], [0.5625, 0.3125], [-0.5625, -0.4375],[-0.6875, -0.4375],[0.3125, 0.5625], [-0.4375, -0.4375], [-0.3125, -0.4375], [-0.3125, -0.5625], [-0.5625, -0.3125]])


#print(np.squeeze(np.array(pareto_nodes_center)).shape)

# Plot Pareto set
a= np.squeeze(np.array(pareto_nodes_center)).reshape(-1, 2)
print(a.shape)
y = problem_model.observe(a, std=0)
plot_pareto_front(func_val1, func_val2, mask, y[:,0], y[:,1])