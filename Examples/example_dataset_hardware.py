import numpy as np
import gpflow as gpf
from sklearn import preprocessing
from AdaptiveEpsilonPAL import AdaptiveEpsilonPAL
from OptimizationProblem import OptimizationProblem
from GaussianProcessModel import GaussianProcessModel
from Hypercube import Hypercube
from utils import printl

from utils_plot import plot_func_list, plot_pareto_front

from paretoset import paretoset
import pandas as pd

np.random.seed(10)
# Load the dataset into a data frame
data = pd.read_csv("data.txt", sep=';', header = None).to_numpy()

# Standardize the design space and the objectives
scaler = preprocessing.MinMaxScaler()
data[:, :3] = scaler.fit_transform(data[:, :3])
data[:, 3:] = preprocessing.MinMaxScaler().fit_transform(data[:, 3:])

# Randomly choose 40 instances to use in GP initialization, sample from the rest
rng = np.random.default_rng()
rng.shuffle(data, axis = 0)
gp_split = data[:40]
sample_split = data[40:]

# # Inputs are 3 dimensional, objectives are the other dims
# x_vals = df[[0, 1, 2]].to_numpy()
# y_vals = df[[3, 4]].to_numpy()

# # Visualize the functions (two functions)
# title1 = "$2sin(\pi x_1)sin(\pi x_2) + 4sin(2 \pi x_1)sin(2 \pi x_2)$"
# #title1 = "Six-Hump Camel Back (Neg)"
# title2 = "Rescaled Branin-Hoo (Neg)"
# func_val1, func_val2 = plot_func_list(func_list, (0, 1), (0, 1), title1, title2)
#
# # Plot pareto front (two functions)
# hotels = pd.DataFrame({"price": func_val1, "distance_to_beach": func_val2})
# mask = paretoset(hotels, sense=["max", "max"])
# plot_pareto_front(func_val1, func_val2, mask)
#
#
# # Generate synthetic data
# data = np.random.uniform(low=0, high=1, size=(40, 2))  # Can be generated with opt problem instance for syn. data
# y = problem_model.observe(data, std=5)

# data = x_vals[0:40,:]
# scaler = preprocessing.StandardScaler().fit(data)
# data2 = scaler.transform(data)
# data2 = data
# y = y_vals[:40,:]
# print(scaler.mean_, scaler.scale_)

#problem_model = OptimizationProblem(dataset=(x_vals[41:-1, :], y_vals[41:-1, :]), scaler=scaler)

problem_model = OptimizationProblem(dataset=(sample_split[:, :3], sample_split[:, 3:]))

# Specify kernel and mean function for GP prior
kernel_list = [(gpf.kernels.SquaredExponential(lengthscales=[0.5, 0.5, 0.5])) for _ in range(2)] # lengthscales=[0.5, 0.5]
gp = GaussianProcessModel(gp_split[:, :3], gp_split[:, 3:], multi=False, periodic=False, m=2, kernel_list=kernel_list, verbose=True)


# Adaptive Epsilon PAL algorithm
pareto_set, pareto_set_cells = AdaptiveEpsilonPAL(problem_model, epsilon=0.05, delta=0.15, gp=gp,
                                                  initial_hypercube=Hypercube(1, (0.5, 0.5, 0.5))).algorithm()
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


# Plot Pareto set
a = np.squeeze(np.array(pareto_nodes_center)).reshape(-1, 3)
print(a.shape)
#y_obs = problem_model.observe(a, std=0)
#plot_pareto_front(func_val1, func_val2, mask, y_obs[:,0], y_obs[:,1], title=epsilon)




# print(pareto_set, pareto_set_cells)
# print(scaler.mean_, scaler.scale_)
#
# data_alg = np.array([[-0.625, -0.375], [0.1875, 0.8125], [0.1875, 0.9375], [0.375, 0.625], [-0.9375, -0.6875], [-0.9375, -0.5625], [-0.6875, -0.6875], [-0.6875, -0.5625]])
# y = problem_model.observe(data_alg, std=0)
#
# xlist = [[3.5, 8.5, 3. ], [3.5, 8.5, 4. ], [3.5, 9.5, 1. ], [3.5, 8.5, 1. ], [3.5, 9.5, 4.],  [3.5, 7.5, 4. ],[5.5, 6.5, 1. ],[5.5, 6.5, 2. ], [5.5, 6.5, 4. ],[5.5, 6.5, 3. ],
#          [5.5, 7.5, 2. ],[5.5, 7.5, 1. ], [5.5, 7.5, 4. ], [5.5, 7.5, 3. ], [5.5, 8.5, 2. ], [5.5, 8.5, 3. ], [5.5, 9.5, 2. ], [5.5, 9.5, 1. ], [4.5, 6.5, 2. ], [4.5, 6.5, 4. ],
#          [4.5, 6.5, 3.], [4.5, 7.5, 2. ]]
y_obs = np.empty((a.shape[0], 2))
i = 0
for row in a:
    data_alg = np.array(row)
    y = problem_model.observe(data_alg, std=0)
    y_obs[i, :] = y
    i += 1


# Visualize the functions (two functions)
#title1 = "Objective 1$"
#title1 = "Six-Hump Camel Back (Neg)"
#title2 = "Objective 2"
#func_val1, func_val2 = plot_func_list(func_list, (0, 1), (0, 1), title1, title2)

# Plot pareto front (two functions)
hotels = pd.DataFrame({"price": data[:, 3], "distance_to_beach": data[:, 4]})
mask = paretoset(hotels, sense=["max", "max"])
plot_pareto_front(data[:, 3], data[:, 4], mask, y_obs[:,0], y_obs[:,1])

