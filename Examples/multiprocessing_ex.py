# importing the multiprocessing module
import multiprocessing
import os
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




def worker1(epsilon):
    #np.random.seed(134340)

    # Generate the function lists
    func1 = lambda x: (2 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) + 4 * np.sin(2 * np.pi * x[:, 0]) * np.sin(
        2 * np.pi * x[:, 1])) / (81 / 16)
    func2 = lambda x: (2 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) - 6 * np.sin(2 * np.pi * x[:, 0]) * np.sin(
        2 * np.pi * x[:, 1])) / (169 / 24)
    func_list = [func1, func2]
    problem_model = OptimizationProblem(func_list)



    # Generate synthetic data
    data = np.random.uniform(low=-1, high=1, size=(40, 2))  # Can be generated with opt problem instance for syn. data
    y = problem_model.observe(data, std=0.05)

    # Specify kernel and mean function for GP prior
    kernel_list = [(gpf.kernels.SquaredExponential(lengthscales=[0.2, 0.2])) for _ in
                   range(2)]  # lengthscales=[0.5, 0.5]
    #kernel_list = [gpf.kernels.Periodic(gpf.kernels.SquaredExponential(lengthscales=[0.1, 0.1])) for _ in range(2)] # lengthscales=[0.5, 0.5]
    gp = GaussianProcessModel(data, y, multi=False, periodic=True, m=2, kernel_list=kernel_list, verbose=True)

    # Adaptive Epsilon PAL algorithm
    alg_object = AdaptiveEpsilonPAL(problem_model, epsilon=epsilon, delta=0.15, gp=gp,
                                    initial_hypercube=Hypercube(2, (0, 0)))
    pareto_set, pareto_set_cells = alg_object.algorithm()
    hmax = alg_object.hmax
    time_elapsed = alg_object.time_elapsed
    tau_eval = alg_object.tau
    t_eval = alg_object.t
    hmax_len = alg_object.hmax_len

    # Visualize the functions (two functions)
    title1 = "$2sin(\pi x_1)sin(\pi x_2) + 4sin(2 \pi x_1)sin(2 \pi x_2)$"
    title2 = "$2sin(\pi x_1)sin(\pi x_2) - 6sin(2 \pi x_1)sin(2 \pi x_2)$"
    func_val1, func_val2 = plot_func_list(func_list, (-1, 1), (-1, 1), title1, title2, h = int(1/hmax_len))

    # Plot pareto front (two functions)
    hotels = pd.DataFrame({"price": func_val1, "distance_to_beach": func_val2})
    mask = paretoset(hotels, sense=["max", "max"])
    plot_pareto_front(func_val1, func_val2, mask)


    # Print nodes in the Pareto set
    printl(pareto_set)

    # Get the center of each node in the Pareto set and plot after observing
    pareto_nodes_center = [node.get_center() for node in pareto_set]

    # Print the cell centers of the the Pareto node cells
    # print([[cell.get_center() for cell in cells] for cells in pareto_set_cells])
    # print(np.array([[cell.get_center() for cell in cells] for cells in pareto_set_cells]))
    # print(np.array(pareto_nodes_center))

    # data_alg = np.array([[-0.625, -0.375], [0.1875, 0.8125], [0.1875, 0.9375], [0.375, 0.625], [-0.9375, -0.6875], [-0.9375, -0.5625], [-0.6875, -0.6875], [-0.6875, -0.5625]])
    # data_alg2 = np.array([[-0.6875, -0.5625], [-0.1875, -0.8125], [-0.3125, -0.3125], [0.4375, 0.6875], [-0.5625, -0.5625], [0.6875, 0.4375], [0.5625, 0.5625],[0.3125, 0.3125], [0.4375, 0.3125],[0.5625, 0.4375], [-0.4375, -0.3125], [0.4375, 0.5625], [-0.4375, -0.5625], [0.4375, 0.4375], [-0.4375, -0.6875], [-0.3125, -0.6875],[-0.6875, -0.6875], [0.6875, 0.5625], [0.6875, 0.3125], [0.6875, 0.6875], [-0.6875, -0.3125], [0.3125, 0.6875], [-0.5625, -0.6875], [0.5625, 0.3125], [-0.5625, -0.4375],[-0.6875, -0.4375],[0.3125, 0.5625], [-0.4375, -0.4375], [-0.3125, -0.4375], [-0.3125, -0.5625], [-0.5625, -0.3125]])

    # print(np.squeeze(np.array(pareto_nodes_center)).shape)

    # Plot Pareto set

    # Plot pareto front (two functions)
    a = np.squeeze(np.array(pareto_nodes_center)).reshape(-1, 2)
    print(a.shape)
    y = problem_model.observe(a, std=0)

    # Error metric
    p_set = np.hstack((func_val1.reshape(-1, 1), func_val2.reshape(-1, 1)))

    c = 0
    for row in p_set:
        a = y - row
        b = np.linalg.norm(a, axis=1)
        # print("b")
        # print(b)
        c += np.min(b)

    title = "$\epsilon = $" + '%.2f' % epsilon + ", Error = " + '%.3f' % (c / p_set.shape[0]) + r'$, \tau $ :' + str(
        tau_eval) + ", Time(s) :" + '%.3f' % time_elapsed
    plot_pareto_front(func_val1, func_val2, mask, y[:, 0], y[:, 1], title=title, plotfront=True)
    plot_pareto_front(func_val1, func_val2, mask, y[:, 0], y[:, 1], title=title, plotfront=False)


    # 2nd
    # Get the center of each node in the Pareto set and plot after observing
    cell_list = []
    for node in pareto_set:
        for cell in node.hypercube_list:
            cell_list.append(cell)

    cells = [hypercube.get_center() for hypercube in cell_list]

    a = np.squeeze(np.array(cells)).reshape(-1, 2)
    print(a.shape)
    y = problem_model.observe(a, std=0)

    # Error metric
    p_set = np.hstack((func_val1.reshape(-1, 1), func_val2.reshape(-1, 1)))

    c1 = 0
    for row in p_set:
        a = y - row
        b = np.linalg.norm(a, axis=1)
        # print("b")
        # print(b)
        c1 += np.min(b)

    title = "$\epsilon = $" + '%.2f' % epsilon + ", Error = " + '%.3f' % (c1 / p_set.shape[0]) + r'$, \tau $ :' + str(
        tau_eval) + ", Time(s) :" + '%.3f' % time_elapsed
    plot_pareto_front(func_val1, func_val2, mask, y[:, 0], y[:, 1], title=title, plotfront=True)
    plot_pareto_front(func_val1, func_val2, mask, y[:, 0], y[:, 1], title=title, plotfront=False)

    return tau_eval, c / p_set.shape[0], (c1 / p_set.shape[0])



if __name__ == "__main__":
    # printing main program process id
    print("ID of main process: {}".format(os.getpid()))

    pool3 = multiprocessing.Pool(processes=3)
    p3 = pool3.map(worker1, [1, 1, 1])
    np.savetxt("sine_name0_1.txt", np.asarray(p3))

    # creating processes
    pool = multiprocessing.Pool(processes=3)
    p = pool.map(worker1, [0.5, 0.5, 0.5])
    np.savetxt("sine_name0_3.txt", np.asarray(p))



    pool2 = multiprocessing.Pool(processes=3)
    p2 = pool2.map(worker1, [0.3, 0.3, 0.3])
    np.savetxt("sine_name0_05.txt", np.asarray(p2))

