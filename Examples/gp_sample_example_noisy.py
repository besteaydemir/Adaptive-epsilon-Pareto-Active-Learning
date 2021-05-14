import multiprocessing
import time

import numpy as np
import gpflow as gpf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing

from AdaptiveEpsilonPAL import AdaptiveEpsilonPAL
from OptimizationProblem import OptimizationProblem
from GaussianProcessModel import GaussianProcessModel
from Hypercube import Hypercube
from utils import *
from utils_plot import plot_pareto_front, plot_func_list

from paretoset import paretoset
import pandas as pd


def worker1(epsilonseedls):

    # Set seed for reproducibility
    epsilon, seed, ls = epsilonseedls
    np.random.seed(seed)


    # Sample from a GP
    bounds = [(0., 1.), (0., 1.)]
    noise_var = 0
    kernel = gpf.kernels.SquaredExponential(lengthscales=0.3, variance=0.4)
    func1 = sample_gp_function(kernel, bounds, noise_var, num_samples=50)
    func2 = sample_gp_function(kernel, bounds, noise_var, num_samples=50)


    # Set the number of objectives
    m = 2
    d = 2


    # Generate the optimization problem, used for sampling
    func_list = [func1, func2]
    problem_model = OptimizationProblem(cardinality=2500, N=2, D_1=2, func_list=func_list)


    # Visualize the functions (two functions)
    title1 = "$GP Sample 1$"
    title2 = "$GP Sample 2$"
    func_val1, func_val2 = plot_func_list(func_list, (0, 1), (0, 1), title1, title2, no_points=50)


    func_val1 = func_val1.reshape(-1, )
    func_val2 = func_val2.reshape(-1, )


    # Plot pareto front (two functions)
    hotels = pd.DataFrame({"func1": func_val1, "func2": func_val2})
    mask = paretoset(hotels, sense=["max", "max"])
    plot_pareto_front(func_val1, func_val2, mask, plotfront=True, figtitle="initpareto")



    # Specify kernel and mean function for GP prior
    kernel_gp = gpf.kernels.SquaredExponential(lengthscales=ls, variance=0.4)
    kernel_list = [kernel_gp, kernel_gp]
    gp = GaussianProcessModel(kernel_list=kernel_list, d=d, verbose=True)


    # Adaptive Epsilon PAL algorithm
    delta = 0.15
    alg_object = AdaptiveEpsilonPAL(problem_model, epsilon=epsilon, delta=delta, gp=gp,
                                    initial_hypercube=Hypercube(1, (0.5, 0.5)))
    titles = "epsilon" + str(epsilon) + "delta" + str(delta) + "seed" + str(seed) + "ls" + str(ls)
    pareto_set, pareto_set_cells = alg_object.algorithm(titles = titles)



    if pareto_set:

        hmax = alg_object.hmax
        time_elapsed = alg_object.time_elapsed
        tau_eval = alg_object.tau

        # Print nodes in the Pareto set
        printl(pareto_set)

        # Get the center of each node in the Pareto set and plot after observing
        pareto_nodes_center = [node.get_center() for node in pareto_set]

        # Plot Pareto set
        a = np.squeeze(np.array(pareto_nodes_center)).reshape(-1, d)

        y_obs = problem_model.observe(a, std=0.0)

        # Error metric
        p_set = np.hstack((func_val1[mask].reshape(-1, 1), func_val2[mask].reshape(-1, 1)))


        c = 0
        for row in p_set:
            a = y_obs - row
            b = np.linalg.norm(a, axis=1)
            c += np.min(b)


        title = "$\epsilon = $" + '%.2f' % epsilon + " $ \delta = $" + '%.2f' % delta + ", Error = " + '%.3f' % (c / p_set.shape[0]) + " " + r'$, \tau $ :' + str(
            tau_eval)
        figtitle = "epsilon" + str(epsilon) + "delta" + str(delta) + "Error" + str(c / p_set.shape[0]) + 'tau' + str(
            tau_eval) + "seed" + str(seed) + "ls" + str(ls)


        plot_pareto_front(func_val1, func_val2, mask, y_obs[:, 0], y_obs[:, 1], title=title,
                           plotfront=True, figtitle=figtitle)



        # # 2nd
        # Get the center of each node in the Pareto set and plot after observing
        cell_list = []
        for node in pareto_set:
            for cell in node.hypercube_list:
                cell_list.append(cell)

        cells = [hypercube.get_center() for hypercube in cell_list]
        # Plot Pareto set
        a = np.squeeze(np.array(cells)).reshape(-1, 2)

        y_obs = np.empty((a.shape[0], 2))
        i = 0
        for row in a:
            data_alg = np.array(row).reshape(-1, 2)
            y = problem_model.observe(data_alg, std=0)
            y_obs[i, :] = y
            i += 1


        # Error metric
        p_set2 = np.hstack((func_val1[mask].reshape(-1, 1), func_val2[mask].reshape(-1, 1)))

        c2 = 0
        for row in p_set2:
            a = y_obs - row
            b = np.linalg.norm(a, axis=1)
            c2 += np.min(b)


        title = "$\epsilon = $" + '%.2f' % epsilon + " $ \delta = $" + '%.2f' % delta +  ", Error = " + '%.3f' % (c2 / p_set2.shape[0]) + r'$, \tau $ :' + str(
            tau_eval)
        figtitle = "epsilon" + str(epsilon) + "delta" + str(delta) + "Error" + str(c2 / p_set2.shape[0]) + 'tau' + str(
            tau_eval) + "seed" + str(seed) + "cell" + "ls" + str(ls)


        plot_pareto_front(func_val1, func_val2, mask, y_obs[:, 0], y_obs[:, 1], title=title,
                          plotfront=True, figtitle=figtitle)


        # Plot close up
        figtitle = "epsilon" + str(epsilon) + "delta" + str(delta) + "Error" + str(c2 / p_set2.shape[0]) + 'tau' + str(
            tau_eval) + "seed" + str(seed) + "cell" + "_lim" + "ls" + str(ls)

        plot_pareto_front(func_val1, func_val2, mask, y_obs[:, 0], y_obs[:, 1], title=title,
                          plotfront=True, figtitle=figtitle, lim=[0.1, 0.8])


        #Plot masked
        objs = pd.DataFrame({"obj1": y_obs[:, 0], "obj2": y_obs[:, 1]})
        mask_pareto = paretoset(objs, sense=["max", "max"])

        figtitle = "epsilon" + str(epsilon) + "delta" + str(delta) + "Error" + str(c2 / p_set2.shape[0]) + 'tau' + str(
            tau_eval) + "seed" + str(seed) + "cell" + "_pareto" + "ls" + str(ls)

        plot_pareto_front(func_val1, func_val2, mask,  y_obs[:, 0], y_obs[:, 1], title=title,
                          plotfront=True, figtitle=figtitle, mask_pareto=mask_pareto)



        # Plot close up and paretoed
        figtitle = "epsilon" + str(epsilon) + "delta" + str(delta) + "Error" + str(c2 / p_set2.shape[0]) + 'tau' + str(
            tau_eval) + "seed" + str(seed) + "cell" + "_lim_pareto" + "ls" + str(ls)

        plot_pareto_front(func_val1, func_val2, mask, y_obs[:, 0], y_obs[:, 1], title=title,
                          plotfront=True, figtitle=figtitle, mask_pareto=mask_pareto, lim=[0.1, 0.8])


        return (tau_eval, c / p_set.shape[0], c2 / p_set2.shape[0], time_elapsed, epsilon, seed, hmax)

    else:
        return -1, -1, -1, -1, -1, -1, -1


if __name__ == "__main__":

    # worker1((0.1, 7))
    # worker1((0.2, 7))
    # worker1((0.4, 7))
    worker1((30, 7, 0.3))


    pool3 = multiprocessing.Pool(processes=4)
    p3 = pool3.map(worker1, [(0.1, 7, 0.33), (0.1, 7, 0.40), (0.1, 7, 0.27), (0.1, 7, 0.20)])
    np.savetxt("final_gp2_noisy_1.txt", np.asarray(p3))
    #
    # pool = multiprocessing.Pool(processes=2)
    # p = pool.map(worker1, [(0.1, 7), (0.05, 7)])
    # np.savetxt("final_gp2_2.txt", np.asarray(p))

    # pool = multiprocessing.Pool(processes=1)
    # p = pool.map(worker1, [(0.02, 7)])
    # np.savetxt("final_gp2_3.txt", np.asarray(p))




    # pool4 = multiprocessing.Pool(processes=2)
    # p4 = pool4.map(worker1, [(0.1, 1), (0.1, 2)])
    # np.savetxt("01_yeslsm_norm.txt", np.asarray(p4))
    #
    # pool5 = multiprocessing.Pool(processes=2)
    # p5 = pool5.map(worker1, [(0.1, 5), (0.1, 6)])
    # np.savetxt("01_yeslsm_norm2.txt", np.asarray(p5))
    #
    # pool6 = multiprocessing.Pool(processes=2)
    # p6 = pool6.map(worker1, [(0.2, 1), (0.2, 2)])
    # np.savetxt("02_yeslsm_norm.txt", np.asarray(p6))
    #
    # pool7 = multiprocessing.Pool(processes=2)
    # p7 = pool7.map(worker1, [(0.2, 5), (0.2, 6)])
    # np.savetxt("02_yeslsm_norm2.txt", np.asarray(p7))



