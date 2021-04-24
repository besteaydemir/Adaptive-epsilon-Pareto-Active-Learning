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
from utils_plot import plot_pareto_front

from paretoset import paretoset
import pandas as pd


def worker1(epsilon):
    # Set seed for reproducibility


    # Load the dataset into a data frame
    data = pd.read_csv("data.txt", sep=';', header=None).to_numpy()

    # Standardize the design space and the objectives
    scaler = preprocessing.MinMaxScaler()
    data[:, :3] = scaler.fit_transform(data[:, :3])
    data[:, 3:] = preprocessing.MinMaxScaler().fit_transform(data[:, 3:]) * 2 -1

    # plt.scatter(data[:,2], data[:,4])
    # plt.show()
    # plt.scatter(data[:, 1], data[:, 4])
    # plt.show()
    # plt.scatter(data[:, 2], data[:, 3])
    # plt.show()

    # fig, axs = plt.subplots(1, 2, tight_layout=True)
    #
    # # We can set the number of bins with the `bins` kwarg
    # axs[0].hist(data[:, 3], bins=20)
    # axs[1].hist(data[:, 4], bins=20)
    # plt.show()
    #
    # X = data[:, :3]
    # Y = data[:, 4, None]
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    #
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2], s = 100, c=Y.flatten(), cmap = plt.get_cmap("magma"))
    # plt.show()
    #
    # Randomly choose instances to use in GP initialization, sample from the rest
    np.random.shuffle(data)
    gp_split = data[:40]
    sample_split = data[40:]

    np.random.seed(10)
    #t = 1000 * time.time()  # current time in milliseconds
    #np.random.seed(int(t) % 2 ** 32)

    problem_model = OptimizationProblem(dataset=(sample_split[:, :3], sample_split[:, 3:]))

    # Specify kernel and mean function for GP prior
    af = np.array([1, 5, 1])
    bf = 1 + np.random.randn(3,) * 0.4
    lsf = list(af * bf)
    ad = np.array([1, 5, 30])
    bd = 1 + np.random.randn(3, ) * 0.4
    ls2d = list(ad * bd)
    kernel_list = [gpf.kernels.SquaredExponential(lengthscales=lsf), gpf.kernels.SquaredExponential(lengthscales=ls2d)]  # lengthscales=[0.1, 0.1, 0.1]
    gp = GaussianProcessModel(X=gp_split[:, :3], Y=gp_split[:, 3:], multi=False, periodic=False, m=2,
                              kernel_list=kernel_list, verbose=True)

    # Adaptive Epsilon PAL algorithm

    delta = 0.10
    alg_object = AdaptiveEpsilonPAL(problem_model, epsilon=epsilon, delta=delta, gp=gp,
                                    initial_hypercube=Hypercube(1, (0.5, 0.5, 0.5)))

    pareto_set, pareto_set_cells = alg_object.algorithm()

    hmax = alg_object.hmax
    time_elapsed = alg_object.time_elapsed
    tau_eval = alg_object.tau
    t_eval = alg_object.t

    # Print nodes in the Pareto set
    printl(pareto_set)

    # Get the center of each node in the Pareto set and plot after observing
    pareto_nodes_center = [node.get_center() for node in pareto_set]

    # Plot Pareto set
    a = np.squeeze(np.array(pareto_nodes_center)).reshape(-1, 3)

    y_obs = np.empty((a.shape[0], 2))
    i = 0
    for row in a:
        data_alg = np.array(row)
        y = problem_model.observe(data_alg, std=0)
        y_obs[i, :] = y
        i += 1

    # Plot pareto front (two functions)
    hotels = pd.DataFrame({"price": sample_split[:, 3], "distance_to_beach": sample_split[:, 4]})
    mask = paretoset(hotels, sense=["max", "max"])

    # Error metric
    p_set = np.hstack((sample_split[:, 3][mask].reshape(-1, 1), sample_split[:, 4][mask].reshape(-1, 1)))
    print(p_set)
    c = 0
    for row in p_set:
        # row =
        print(row)
        a = y_obs - row
        print(a)
        b = np.linalg.norm(a, axis=1)
        print(b)
        c += np.min(b)
        print(c)
        print("ended")
    print(p_set.shape[0])

    title = "$\epsilon = $" + '%.2f' % epsilon + " $ \delta = $" + '%.2f' % delta + ", Error = " + '%.3f' % (c / p_set.shape[0]) + r'$, \tau $ :' + str(
        tau_eval)

    plot_pareto_front(sample_split[:, 3], sample_split[:, 4], mask, y_obs[:, 0], y_obs[:, 1], title=title,
                      plotfront=True)
    plot_pareto_front(sample_split[:, 3], sample_split[:, 4], mask, y_obs[:, 0], y_obs[:, 1], title=title,
                      plotfront=False)


    # 2nd
    # Get the center of each node in the Pareto set and plot after observing
    cell_list = []
    for node in pareto_set:
        for cell in node.hypercube_list:
            cell_list.append(cell)

    cells = [hypercube.get_center() for hypercube in cell_list]
    # Plot Pareto set
    a = np.squeeze(np.array(cells)).reshape(-1, 3)

    y_obs = np.empty((a.shape[0], 2))
    i = 0
    for row in a:
        data_alg = np.array(row)
        y = problem_model.observe(data_alg, std=0)
        y_obs[i, :] = y
        i += 1

    # Plot pareto front (two functions)
    hotels = pd.DataFrame({"price": sample_split[:, 3], "distance_to_beach": sample_split[:, 4]})
    mask = paretoset(hotels, sense=["max", "max"])

    # Error metric
    p_set2 = np.hstack((sample_split[:, 3][mask].reshape(-1, 1), sample_split[:, 4][mask].reshape(-1, 1)))
    print(p_set2)
    c2 = 0
    for row in p_set2:
        # row =
        print(row)
        a = y_obs - row
        print(a)
        b = np.linalg.norm(a, axis=1)
        print(b)
        c2 += np.min(b)
        print(c2)
        print("ended")
    print(p_set2.shape[0])

    title = "$\epsilon = $" + '%.2f' % epsilon + " $ \delta = $" + '%.2f' % delta +  ", Error = " + '%.3f' % (c2 / p_set2.shape[0]) + r'$, \tau $ :' + str(
        tau_eval)

    plot_pareto_front(sample_split[:, 3], sample_split[:, 4], mask, y_obs[:, 0], y_obs[:, 1], title=title,
                      plotfront=True)
    plot_pareto_front(sample_split[:, 3], sample_split[:, 4], mask, y_obs[:, 0], y_obs[:, 1], title=title,
                      plotfront=False)
    return tau_eval, c / p_set.shape[0], c2 / p_set2.shape[0], time_elapsed


if __name__ == "__main__":

    pool3 = multiprocessing.Pool(processes=2)
    p = pool3.map(worker1, [0.4, 0.4])
    np.savetxt("name5.txt", np.asarray(p))

    # pool2 = multiprocessing.Pool(processes=2)
    # p2 = pool2.map(worker1, [0.1, 0.1, 0.1])
    # np.savetxt("name1_5.txt", np.asarray(p2))
    #
    # pool3 = multiprocessing.Pool(processes=1)
    # p3 = pool3.map(worker1, [0.1, 0.1, 0.1])
    # np.savetxt("name0_15.txt", np.asarray(p3))

