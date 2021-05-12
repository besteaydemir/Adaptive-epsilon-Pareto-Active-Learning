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
from utils import printl
from utils_plot import plot_pareto_front, plot_func_list

from paretoset import paretoset
import pandas as pd


def worker1(epsilonseed):
    # Set seed for reproducibility
    epsilon, seed = epsilonseed
    np.random.seed(seed)

    # Generate the function lists
    func1 = lambda x: (2 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) + 4 * np.sin(2 * np.pi * x[:, 0]) * np.sin(
        2 * np.pi * x[:, 1])) / (81 / 16)
    func2 = lambda x: (2 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) - 6 * np.sin(2 * np.pi * x[:, 0]) * np.sin(
        2 * np.pi * x[:, 1])) / (169 / 24)
    func_list = [func1, func2]
    problem_model = OptimizationProblem(func_list)

    # Visualize the functions (two functions)
    title1 = "$2sin(\pi x_1)sin(\pi x_2) + 4sin(2 \pi x_1)sin(2 \pi x_2)$"
    title2 = "$2sin(\pi x_1)sin(\pi x_2) - 6sin(2 \pi x_1)sin(2 \pi x_2)$"
    func_val1, func_val2 = plot_func_list(func_list, (0, 1), (0, 1), title1, title2)

    # Plot pareto front (two functions)
    hotels = pd.DataFrame({"price": func_val1, "distance_to_beach": func_val2})
    mask = paretoset(hotels, sense=["max", "max"])
    plot_pareto_front(func_val1, func_val2, mask, plotfront=True)

    # Generate synthetic data
    data = np.random.uniform(low=0, high=1, size=(40, 2))  # Can be generated with opt problem instance for syn. data
    y = problem_model.observe(data, std=0.1)

    # Specify kernel and mean function for GP prior
    kernel_list = [(gpf.kernels.SquaredExponential()) for _ in
                   range(2)]  # lengthscales=[0.5, 0.5]
    # kernel_list = [gpf.kernels.Periodic(gpf.kernels.SquaredExponential(lengthscales=[0.1, 0.1])) for _ in range(2)] # lengthscales=[0.5, 0.5]
    gp = GaussianProcessModel(data, y, multi=False, periodic=True, m=2, kernel_list=kernel_list, verbose=True)

    # Load the dataset into a data frame


    # # Standardize the design space and the objectives
    # scaler = preprocessing.MinMaxScaler()
    # data[:, :4] = scaler.fit_transform(data[:, :4])
    # data[:, 4:] = preprocessing.MinMaxScaler().fit_transform(data[:, 4:]) * 2 - 1
    #
    # # plt.scatter(data[:,2], data[:,4])
    # # plt.show()
    # # plt.scatter(data[:, 1], data[:, 4])
    # # plt.show()
    # # plt.scatter(data[:, 2], data[:, 3])
    # # plt.show()
    #
    # # fig, axs = plt.subplots(1, 2, tight_layout=True)
    # #
    # # # We can set the number of bins with the `bins` kwarg
    # # axs[0].hist(data[:, 3], bins=20)
    # # axs[1].hist(data[:, 4], bins=20)
    # # plt.show()
    # #
    # # X = data[:, :3]
    # # Y = data[:, 4, None]
    # #
    # # fig = plt.figure()
    # # ax = Axes3D(fig)
    # #
    # # ax.scatter(X[:, 0], X[:, 1], X[:, 2], s = 100, c=Y.flatten(), cmap = plt.get_cmap("magma"))
    # # plt.show()
    # #
    # # Randomly choose instances to use in GP initialization, sample from the rest
    # np.random.shuffle(data)
    # gp_split = data[:40]
    # sample_split = data[40:]
    #
    #
    # #t = 1000 * time.time()  # current time in milliseconds
    # #np.random.seed(int(t) % 2 ** 32)
    #
    # problem_model = OptimizationProblem(dataset=(sample_split[:, :4], sample_split[:, 4:]))
    #
    # # Specify kernel and mean function for GP prior
    # a = np.array([0.1, 0.1, 11,0.1])
    # b = 1 + np.random.randn(4,) * 0.1
    # ls = list(a * b )
    # a = np.array([0.1, 0.1, 11, 0.1])
    # b = 1 + np.random.randn(4, ) * 0.1
    # ls2 = list(a * b )
    # kernel_list = [(gpf.kernels.SquaredExponential()),
    #                (gpf.kernels.SquaredExponential())]  # lengthscales=[0.1, 0.1, 0.1]
    # gp = GaussianProcessModel(X=gp_split[:, :4], Y=gp_split[:, 4:], multi=False, periodic=False, m=2,
    #                           kernel_list=kernel_list, verbose=True)

    # Adaptive Epsilon PAL algorithm

    delta = 0.10
    alg_object = AdaptiveEpsilonPAL(problem_model, epsilon=epsilon, delta=delta, gp=gp,
                                    initial_hypercube=Hypercube(1, (0.5, 0.5)))
    titles = "epsilon" + str(epsilon) + "delta" + str(delta) + "seed" + str(seed)
    pareto_set, pareto_set_cells = alg_object.algorithm(titles = titles)

    if pareto_set:

        hmax = alg_object.hmax
        time_elapsed = alg_object.time_elapsed
        tau_eval = alg_object.tau
        t_eval = alg_object.t

        # Print nodes in the Pareto set
        printl(pareto_set)

        # Get the center of each node in the Pareto set and plot after observing
        pareto_nodes_center = [node.get_center() for node in pareto_set]

        # Plot Pareto set
        a = np.squeeze(np.array(pareto_nodes_center)).reshape(-1, 2)

        y_obs = problem_model.observe(a, std=0)

        # y_obs = np.empty((a.shape[0], 2))
        # i = 0
        # for row in a:
        #     data_alg = np.array(row).reshape((1,2))
        #     print(data_alg.shape)
        #     y = problem_model.observe(data_alg, std=0)
        #     y_obs[i, :] = y
        #     i += 1


        # Error metric
        p_set = np.hstack((func_val1[mask].reshape(-1, 1), func_val2[mask].reshape(-1, 1)))
        print(p_set)
        c = 0
        for row in p_set:
            a = y_obs - row
            b = np.linalg.norm(a, axis=1)
            c += np.min(b)

        title = "$\epsilon = $" + '%.2f' % epsilon + " $ \delta = $" + '%.2f' % delta + " " + r'$, \tau $ :' + str(
            tau_eval)
        figtitle = "epsilon" + str(epsilon) + "delta" + str(delta) + "Error" + str(c / p_set.shape[0]) + 'tau' + str(
            tau_eval) + "seed" + str(seed)

        plot_pareto_front(y1= y_obs[:, 0], y2= y_obs[:, 1], title=title,
                          plotfront=True, figtitle = figtitle)
        # plot_pareto_front(sample_split[:, 3], sample_split[:, 4], mask, y_obs[:, 0], y_obs[:, 1], title=title,
        #                   plotfront=False)





        # # 2nd
        # # Get the center of each node in the Pareto set and plot after observing
        # cell_list = []
        # for node in pareto_set:
        #     for cell in node.hypercube_list:
        #         cell_list.append(cell)
        #
        # cells = [hypercube.get_center() for hypercube in cell_list]
        # # Plot Pareto set
        # a = np.squeeze(np.array(cells)).reshape(-1, 4)
        #
        # y_obs = np.empty((a.shape[0], 2))
        # i = 0
        # for row in a:
        #     data_alg = np.array(row)
        #     y = problem_model.observe(data_alg, std=0)
        #     y_obs[i, :] = y
        #     i += 1
        #
        # # Plot pareto front (two functions)
        # hotels = pd.DataFrame({"price": sample_split[:, 4], "distance_to_beach": sample_split[:, 5]})
        # mask = paretoset(hotels, sense=["max", "max"])
        #
        # # Error metric
        # p_set2 = np.hstack((sample_split[:, 4][mask].reshape(-1, 1), sample_split[:, 5][mask].reshape(-1, 1)))
        # print(p_set2)
        # c2 = 0
        # for row in p_set2:
        #     # row =
        #     print(row)
        #     a = y_obs - row
        #     print(a)
        #     b = np.linalg.norm(a, axis=1)
        #     print(b)
        #     c2 += np.min(b)
        #     print(c2)
        #     print("ended")
        # print(p_set2.shape[0])
        #
        # title = "$\epsilon = $" + '%.2f' % epsilon + " $ \delta = $" + '%.2f' % delta +  ", Error = " + '%.3f' % (c2 / p_set2.shape[0]) + r'$, \tau $ :' + str(
        #     tau_eval)
        # figtitle = "epsilon" + str(epsilon) + "delta" + str(delta) + "Error" + str(c2 / p_set2.shape[0]) + 'tau' + str(
        #     tau_eval) + "seed" + str(seed) + "cell"
        #
        # plot_pareto_front(sample_split[:, 4], sample_split[:, 5], mask, y_obs[:, 0], y_obs[:, 1], title=title,
        #                   plotfront=True, figtitle=figtitle)
        # # plot_pareto_front(sample_split[:, 3], sample_split[:, 4], mask, y_obs[:, 0], y_obs[:, 1], title=title,
        # #                   plotfront=False)
        # print(tau_eval, c / p_set.shape[0], c2 / p_set2.shape[0], time_elapsed, epsilon, seed)
        return tau_eval, c / p_set.shape[0], time_elapsed, epsilon, seed
    else:
        return -1, -1, -1, -1, -1


if __name__ == "__main__":
    worker1((10, 1))

    # pool3 = multiprocessing.Pool(processes=2)
    # p3 = pool3.map(worker1, [(0.4, 1), (0.4, 2)])
    # np.savetxt("04_yeslsm_norm.txt", np.asarray(p3))
    #
    # pool = multiprocessing.Pool(processes=2)
    # p = pool.map(worker1, [(0.4, 5), (0.4, 6)])
    # np.savetxt("04_yeslsm_norm2.txt", np.asarray(p))
    #
    # pool6 = multiprocessing.Pool(processes=2)
    # p6 = pool6.map(worker1, [(0.2, 1), (0.2, 2)])
    # np.savetxt("02_yeslsm_norm.txt", np.asarray(p6))
    #
    # pool7 = multiprocessing.Pool(processes=2)
    # p7 = pool7.map(worker1, [(0.2, 5), (0.2, 6)])
    # np.savetxt("02_yeslsm_norm2.txt", np.asarray(p7))
    #
    pool4 = multiprocessing.Pool(processes=1)
    p4 = pool4.map(worker1, [(10, 1)])
    np.savetxt("01_yeslsm_norm.txt", np.asarray(p4))

    pool5 = multiprocessing.Pool(processes=3)
    p5 = pool5.map(worker1, [(0.1, 5), (0.1, 6), (0.1, 2)])
    np.savetxt("01_yeslsm_norm2.txt", np.asarray(p5))





