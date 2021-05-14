import multiprocessing
import time

import numpy as np
import gpflow as gpf
from sklearn import preprocessing

from AdaptiveEpsilonPAL import AdaptiveEpsilonPAL
from OptimizationProblem import OptimizationProblem
from GaussianProcessModel import GaussianProcessModel
from Hypercube import Hypercube
from utils import printl
from utils_plot import plot_pareto_front
from gpflow.utilities import print_summary
from paretoset import paretoset
import pandas as pd
import tensorflow as tf


def worker1(epsilonseed):

    # Set seed for reproducibility
    epsilon, seed = epsilonseed
    np.random.seed(seed)


    # Load the dataset into a data frame
    data = pd.read_csv("Data/data.txt", sep=';', header=None).to_numpy()


    # Standardize the design space and the objectives
    data[:, :3] = preprocessing.MinMaxScaler().fit_transform(data[:, :3])
    data[:, 3:] = preprocessing.MinMaxScaler().fit_transform(data[:, 3:]) * 2 - 1


    # Set the number of objectives
    m = 2
    d = 3


    # Randomly choose instances to use in GP initialization, sample from the rest
    np.random.shuffle(data)
    gp_split = data[:40]
    sample_split = data[40:]


    # Find kernel parameters by using gp_split data
    kernel_list = [gpf.kernels.SquaredExponential(), gpf.kernels.SquaredExponential()]
    opt = gpf.optimizers.Scipy()
    noise_variance = 1

    for i in range(m):
        model = gpf.models.GPR(data=(gp_split[:, :3], gp_split[:, 3:][:, i].reshape(-1, 1)), kernel=kernel_list[i],
                              noise_variance=noise_variance)
        print_summary(model)
        print("Log likelihood ", tf.keras.backend.get_value(model.log_marginal_likelihood()))

        # Tune the model parameters according to data
        opt.minimize(
            model.training_loss,
            variables=model.trainable_variables,
            method="l-bfgs-b",
            options={"disp": False, "maxiter": 100}
        )
        print_summary(model)
        print("Log likelihood ", tf.keras.backend.get_value(model.log_marginal_likelihood()))



    # Generate the problem model to sample from
    problem_model = OptimizationProblem(cardinality=206-40, N=2, D_1=d, dataset=(sample_split[:, :3], sample_split[:, 3:]))


    # Specify kernel for GP prior
    gp = GaussianProcessModel(kernel_list=kernel_list, d=d, verbose=True)


    # Adaptive Epsilon PAL algorithm
    delta = 0.10
    alg_object = AdaptiveEpsilonPAL(problem_model, epsilon=epsilon, delta=delta, gp=gp,
                                    initial_hypercube=Hypercube(1, (0.5, 0.5, 0.5)))

    titles = "epsilon" + str(epsilon) + "delta" + str(delta) + "seed" + str(seed)
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

        y_obs = np.empty((a.shape[0], m))


        # Observing for node centers
        i = 0
        for row in a:
            data_alg = np.array(row)
            y = problem_model.observe(data_alg, std=0)
            y_obs[i, :] = y
            i += 1


        # Plot pareto front (two functions)
        objs = pd.DataFrame({"obj1": sample_split[:, 3], "obj2": sample_split[:, 4]})
        mask = paretoset(objs, sense=["max", "max"])


        # Error metric
        p_set = np.hstack((sample_split[:, 3][mask].reshape(-1, 1), sample_split[:, 4][mask].reshape(-1, 1)))


        # Error term
        c = 0
        for row in p_set:
            a = y_obs - row
            b = np.linalg.norm(a, axis=1)
            c += np.min(b)


        title = "$\epsilon = $" + '%.2f' % epsilon + " $ \delta = $" + '%.2f' % delta + ", Error = " + '%.3f' % (c / p_set.shape[0]) + r'$, \tau $ :' + str(
            tau_eval)
        figtitle = "epsilon" + str(epsilon) + "delta" + str(delta) + "Error" + str(c / p_set.shape[0]) + 'tau' + str(
            tau_eval) + "seed" + str(seed)

        plot_pareto_front(sample_split[:, 3], sample_split[:, 4], mask, y_obs[:, 0], y_obs[:, 1], title=title,
                          plotfront=True, figtitle = figtitle)



        # 2nd
        # Get the center of each node in the Pareto set and plot after observing
        cell_list = []
        for node in pareto_set:
            for cell in node.hypercube_list:
                cell_list.append(cell)

        cells = [hypercube.get_center() for hypercube in cell_list]
        # Plot Pareto set
        a = np.squeeze(np.array(cells)).reshape(-1, d)

        y_obs = np.empty((a.shape[0], m))
        i = 0
        for row in a:
            data_alg = np.array(row)
            y = problem_model.observe(data_alg, std=0)
            y_obs[i, :] = y
            i += 1

        # Plot pareto front (two functions)
        objs = pd.DataFrame({"obj1": sample_split[:, 3], "obj2": sample_split[:, 4]})
        mask = paretoset(objs, sense=["max", "max"])


        # Error metric
        p_set2 = np.hstack((sample_split[:, 3][mask].reshape(-1, 1), sample_split[:, 4][mask].reshape(-1, 1)))

        c2 = 0

        for row in p_set2:
            a = y_obs - row
            b = np.linalg.norm(a, axis=1)
            c2 += np.min(b)

        title = "$\epsilon = $" + '%.2f' % epsilon + " $ \delta = $" + '%.2f' % delta +  ", Error = " + '%.3f' % (c2 / p_set2.shape[0]) + r'$, \tau $ :' + str(
            tau_eval)
        figtitle = "epsilon" + str(epsilon) + "delta" + str(delta) + "Error" + str(c2 / p_set2.shape[0]) + 'tau' + str(
            tau_eval) + "seed" + str(seed) + "cell"

        plot_pareto_front(sample_split[:, 3], sample_split[:, 4], mask, y_obs[:, 0], y_obs[:, 1], title=title,
                          plotfront=True, figtitle=figtitle)


        #Plot masked
        objs = pd.DataFrame({"obj1": y_obs[:, 0], "obj2": y_obs[:, 1]})
        mask_pareto = paretoset(objs, sense=["max", "max"])

        figtitle = "epsilon" + str(epsilon) + "delta" + str(delta) + "Error" + str(c2 / p_set2.shape[0]) + 'tau' + str(
            tau_eval) + "seed" + str(seed) + "cell" + "paretoed"

        plot_pareto_front(sample_split[:, 3], sample_split[:, 4], mask,  y_obs[:, 0], y_obs[:, 1], title=title,
                          plotfront=True, figtitle=figtitle, mask_pareto=mask_pareto)

        return tau_eval, c / p_set.shape[0], c2 / p_set2.shape[0], time_elapsed, epsilon, seed, hmax
    else:
        return -1, -1, -1, -1, -1, -1, -1


if __name__ == "__main__":
    worker1((5, 7))
    # worker1((0.2, 7))
    # worker1((0.1, 7))
    # worker1((0.04, 7))

    # pool3 = multiprocessing.Pool(processes=2)
    # p3 = pool3.map(worker1, [(0.8, 13), (0.4, 13), (0.2, 13)])
    # np.savetxt("finalrun_snw.txt", np.asarray(p3))
    #
    # pool = multiprocessing.Pool(processes=2)
    # p = pool.map(worker1, [(0.1, 13), (0.5, 13)])
    # np.savetxt("finalrun2_snw.txt", np.asarray(p))
    #
    # pool = multiprocessing.Pool(processes=1)
    # p = pool.map(worker1, [(0.05, 13)])
    # np.savetxt("finalrun3_snw.txt", np.asarray(p))
    #
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



