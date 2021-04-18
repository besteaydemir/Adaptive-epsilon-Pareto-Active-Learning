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






def worker1(epsilon):
    # Set seed for reproducibility
    np.random.seed(10)

    # Load the dataset into a data frame
    data = pd.read_csv("data.txt", sep=';', header=None).to_numpy()

    # Standardize the design space and the objectives
    scaler = preprocessing.MinMaxScaler()
    data[:, :3] = scaler.fit_transform(data[:, :3])
    #data[:, 3:] = preprocessing.MinMaxScaler().fit_transform(data[:, 3:])

    # Randomly choose 40 instances to use in GP initialization, sample from the rest
    rng = np.random.default_rng()
    rng.shuffle(data, axis=0)
    gp_split = data[:20]
    sample_split = data[20:]

    problem_model = OptimizationProblem(dataset=(sample_split[:, :3], sample_split[:, 3:]))

    # Specify kernel and mean function for GP prior
    kernel_list = [(gpf.kernels.SquaredExponential()) for _ in range(2)]  # lengthscales=[0.1, 0.1, 0.1]
    gp = GaussianProcessModel(X=gp_split[:, :3], Y=gp_split[:, 3:], multi=False, periodic=False, m=2,
                              kernel_list=kernel_list, verbose=True)

    # Adaptive Epsilon PAL algorithm
    alg_object = AdaptiveEpsilonPAL(problem_model, epsilon=epsilon, delta=0.15, gp=gp,
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

    title = "$\epsilon = $" + '%.2f' % epsilon + ", Error = " + '%.3f' % (c / p_set.shape[0]) + r'$, \tau $ :' + str(
        tau_eval) + ", Time(s) :" + '%.3f' % time_elapsed

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

    title = "$\epsilon = $" + '%.2f' % epsilon + ", Error = " + '%.3f' % (c / p_set.shape[0]) + r'$, \tau $ :' + str(
        tau_eval) + ", Time(s) :" + '%.3f' % time_elapsed

    plot_pareto_front(sample_split[:, 3], sample_split[:, 4], mask, y_obs[:, 0], y_obs[:, 1], title=title,
                      plotfront=True)
    plot_pareto_front(sample_split[:, 3], sample_split[:, 4], mask, y_obs[:, 0], y_obs[:, 1], title=title,
                      plotfront=False)
    return tau_eval, c / p_set.shape[0]


if __name__ == "__main__":
    # printing main program process id
    print("ID of main process: {}".format(os.getpid()))
    np.random.seed(9)

    pool = multiprocessing.Pool(processes=2)
    p = pool.map(worker1, [5, 5, 5])
    np.savetxt("name5.txt", np.asarray(p))

    pool2 = multiprocessing.Pool(processes=2)
    p2 = pool2.map(worker1, [1.5, 1.5, 1.5])
    np.savetxt("name1_5.txt", np.asarray(p2))

    pool3 = multiprocessing.Pool(processes=2)
    p3 = pool3.map(worker1, [0.5, 0.5, 0.5])
    np.savetxt("name0_5.txt", np.asarray(p3))



    # # creating processes
    # p1 = multiprocessing.Process(target=worker1, args=(0.3,))
    # p2 = multiprocessing.Process(target=worker1, args=(0.01,))
    # p3 = multiprocessing.Process(target=worker1, args=(0.1,))
    #
    # # starting processes
    # p1.start()
    # p2.start()
    # p3.start()
    #
    # # process IDs
    # print("ID of process p1: {}".format(p1.pid))
    # print("ID of process p2: {}".format(p2.pid))
    #
    # # wait until processes are finished
    # p1.join()
    # p2.join()
    # p3.join()

    # both processes finished
    print("Both processes finished execution!")

    # # check if processes are alive
    # print("Process p1 is alive: {}".format(p1.is_alive()))
    # print("Process p2 is alive: {}".format(p2.is_alive()))