import numpy as np
import gpflow as gpf
from sklearn import preprocessing
from AdaptiveEpsilonPAL import AdaptiveEpsilonPAL
from OptimizationProblem import OptimizationProblem
from GaussianProcessModel import GaussianProcessModel
from Hypercube import Hypercube

from utils_plot import plot_func_list, plot_pareto_front

from paretoset import paretoset
import pandas as pd


# Generate the function lists
df = pd.read_csv("data.txt", sep=';', header = None)
x_vals = df[[0,1,2]].to_numpy()
y_vals = df[[3,4]].to_numpy()




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

data = x_vals[0:40,:]
scaler = preprocessing.StandardScaler().fit(data)
data2 = scaler.transform(data)
data2 = data
y = y_vals[:40,:]
print(scaler.mean_, scaler.scale_)

problem_model = OptimizationProblem(dataset=(x_vals[41:-1, :], y_vals[41:-1, :]), scaler=scaler)

# Specify kernel and mean function for GP prior
kernel_list = [(gpf.kernels.SquaredExponential()) for _ in range(2)] # lengthscales=[0.5, 0.5]
gp = GaussianProcessModel(data2, y, multi=False, m=2, kernel_list=kernel_list, verbose=True)


# Adaptive Epsilon PAL algorithm
pareto_set, pareto_set_cells = AdaptiveEpsilonPAL(problem_model, epsilon=10, delta=0.25, gp=gp,
                                                  initial_hypercube=Hypercube(4, (4, 8, 2.5))).algorithm()

print(pareto_set, pareto_set_cells)
print(scaler.mean_, scaler.scale_)

data_alg = np.array([[-0.625, -0.375], [0.1875, 0.8125], [0.1875, 0.9375], [0.375, 0.625], [-0.9375, -0.6875], [-0.9375, -0.5625], [-0.6875, -0.6875], [-0.6875, -0.5625]])
y = problem_model.observe(data_alg, std=0)

xlist = [[3.5, 8.5, 3. ], [3.5, 8.5, 4. ], [3.5, 9.5, 1. ], [3.5, 8.5, 1. ], [3.5, 9.5, 4.],  [3.5, 7.5, 4. ],[5.5, 6.5, 1. ],[5.5, 6.5, 2. ], [5.5, 6.5, 4. ],[5.5, 6.5, 3. ],
         [5.5, 7.5, 2. ],[5.5, 7.5, 1. ], [5.5, 7.5, 4. ], [5.5, 7.5, 3. ], [5.5, 8.5, 2. ], [5.5, 8.5, 3. ], [5.5, 9.5, 2. ], [5.5, 9.5, 1. ], [4.5, 6.5, 2. ], [4.5, 6.5, 4. ],
         [4.5, 6.5, 3.], [4.5, 7.5, 2. ]]
func_val1 = np.empty((len(xlist),2))
i = 0
for x in xlist:
    data_alg = np.array(x)
    y = problem_model.observe(data_alg, std=0)
    func_val1[i, :] = y
    i += 1

df = pd.read_csv("data.txt", sep=';', header = None)
x_vals = df[[0,1,2]].to_numpy()
y = df[[3,4]].to_numpy()


# Visualize the functions (two functions)
title1 = "Objective 1$"
#title1 = "Six-Hump Camel Back (Neg)"
title2 = "Objective 2"
#func_val1, func_val2 = plot_func_list(func_list, (0, 1), (0, 1), title1, title2)

# Plot pareto front (two functions)
hotels = pd.DataFrame({"price": y[:,0], "distance_to_beach": y[:,1]})
mask = paretoset(hotels, sense=["max", "max"])
plot_pareto_front(y[:,0], y[:,1], mask, func_val1[:,0], func_val1[:,1])

