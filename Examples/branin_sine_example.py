import numpy as np
import gpflow as gpf

from AdaptiveEpsilonPAL import AdaptiveEpsilonPAL
from OptimizationProblem import OptimizationProblem
from GaussianProcessModel import GaussianProcessModel
from Hypercube import Hypercube
from utils import printl

from utils_plot import plot_func_list, plot_pareto_front

from paretoset import paretoset
import pandas as pd


# Generate the function lists
# def branin(x):
#     x1 = (15*x[:, 0] - 5)
#     x2 = (15*x[:, 1])
func2 = lambda x: -(((15*x[:, 1]) - (5.1 * (15*x[:, 0] - 5)**2 / (4*np.pi**2)) + (5 * (15*x[:, 0] - 5) / np.pi - 6))**2 + (10 - 10 / (8 * np.pi)) * np.cos((15*x[:, 0] - 5)) - 44.81) / 51.95

func1 = lambda x: 2 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) + 4 * np.sin(2 * np.pi * x[:, 0]) * np.sin(
    2 * np.pi * x[:, 1])/(81/16)

#func1 = lambda x: -((4 - 2.1 * (x[:,0])**2 + (x[:,0])**4 / 3)* (x[:,0])**2 + x[:,0] * x[:,1] + (-4 + 4*x[:,1]**2)*x[:,1]**2)

func_list = [func1, func2]
problem_model = OptimizationProblem(func_list)


# Visualize the functions (two functions)
title1 = "$2sin(\pi x_1)sin(\pi x_2) + 4sin(2 \pi x_1)sin(2 \pi x_2)$"
#title1 = "Six-Hump Camel Back (Neg)"
title2 = "Rescaled Branin-Hoo (Neg)"
func_val1, func_val2 = plot_func_list(func_list, (0, 1), (0, 1), title1, title2)

# Plot pareto front (two functions)
hotels = pd.DataFrame({"price": func_val1, "distance_to_beach": func_val2})
mask = paretoset(hotels, sense=["max", "max"])
plot_pareto_front(func_val1, func_val2, mask)


# Generate synthetic data
data = np.random.uniform(low=0, high=1, size=(40, 2))  # Can be generated with opt problem instance for syn. data
y = problem_model.observe(data, std=0.1)


# Specify kernel and mean function for GP prior
kernel_list = [(gpf.kernels.SquaredExponential(lengthscales=[0.1, 0.5])) for _ in range(2)] # lengthscales=[0.5, 0.5]
gp = GaussianProcessModel(data, y, periodic=False, multi=False, m=2, kernel_list=kernel_list, verbose=True)


# Adaptive Epsilon PAL algorithm
# Adaptive Epsilon PAL algorithm
epsilon = 0.1
alg_object = AdaptiveEpsilonPAL(problem_model, epsilon=epsilon, delta=0.1, gp=gp,
                                                  initial_hypercube=Hypercube(2, (0, 0)))
pareto_set, pareto_set_cells = alg_object.algorithm()
hmax = alg_object.hmax
time_elapsed = alg_object.time_elapsed
tau_eval = alg_object.tau
t_eval = alg_object.t

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


func_val1, func_val2 = plot_func_list(func_list, (-1, 1), (-1, 1), title1, title2, h=hmax-3)

# Plot pareto front (two functions)
hotels = pd.DataFrame({"price": func_val1, "distance_to_beach": func_val2})
mask = paretoset(hotels, sense=["max", "max"])
plot_pareto_front(func_val1, func_val2, mask)

a= np.squeeze(np.array(pareto_nodes_center)).reshape(-1, 2)
print(a.shape)
y = problem_model.observe(a, std=0)

# Error metric
p_set = np.hstack((func_val1, func_val2))

c = 0
for row in p_set:
    a = y-row
    b = np.linalg.norm(a, axis=1)
    # print("b")
    # print(b)
    c += np.min(b)

title = "$\epsilon = $" + '%.2f' % epsilon + ", Error = " + '%.3f' % (c / p_set.shape[0]) + r'$, \tau $ :' + str(tau_eval) + ", Time(s) :" + '%.3f' % time_elapsed
plot_pareto_front(func_val1, func_val2, mask, y[:,0], y[:,1], title=title, plotfront = True)
plot_pareto_front(func_val1, func_val2, mask, y[:,0], y[:,1], title=title, plotfront = False)