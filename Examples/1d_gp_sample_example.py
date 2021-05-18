import gpflow as gpf


from AdaptiveEpsilonPAL import AdaptiveEpsilonPAL
from OptimizationProblem import OptimizationProblem
from GaussianProcessModel import GaussianProcessModel
from Hypercube import Hypercube
from utils import *
from utils_plot import plot_pareto_front, plot_func_list, plot_func_list_1d

from paretoset import paretoset
import pandas as pd

# Incomplete
def worker1(epsilonseed):

    # Set seed for reproducibility
    epsilon, seed = epsilonseed
    np.random.seed(seed)
    print(epsilon)


    # Sample from a GP
    bounds = [(0., 1.)]
    noise_var = 1e-5
    kernel = gpf.kernels.SquaredExponential(lengthscales=0.20, variance=0.3) #
    func1 = sample_gp_function(kernel, bounds, noise_var, num_samples=300)
    func2 = sample_gp_function(kernel, bounds, noise_var, num_samples=300)


    # Generate the optimization problem, used for sampling
    func_list = [func1, func2]
    problem_model = OptimizationProblem(cardinality=500, N=2, D_1=1, func_list=func_list)


    # Visualize the functions (two functions)
    title1 = "$GP Sample 1$"
    title2 = "$GP Sample 2$"
    func_val1, func_val2 = plot_func_list_1d(func_list, (0, 1), title1, title2, no_points=500)

    print(func_val1.shape)

    func_val1 = func_val1.reshape(-1, )
    func_val2 = func_val2.reshape(-1, )


    # Plot pareto front (two functions)
    hotels = pd.DataFrame({"func1": func_val1.reshape(-1,), "func2": func_val2.reshape(-1,)})
    mask = paretoset(hotels, sense=["max", "max"])
    plot_pareto_front(func_val1.reshape(-1,), func_val2.reshape(-1,), mask, plotfront=True)


    # Generate synthetic data for the initial model GP
    # data = np.random.uniform(low=0, high=1, size=(40, 2))
    # y = problem_model.observe(data, std=0.01)
    # print(y.shape)

    data = np.array([[-1e7]])
    y = np.array([[0., 0.]]).reshape(1,2)


    # Specify kernel and mean function for GP prior
    kernel_list = [(gpf.kernels.SquaredExponential()) for _ in
                   range(2)]  # lengthscales=[0.5, 0.5]
    kernel_gp = gpf.kernels.SquaredExponential(lengthscales=0.20, variance=0.3)
    kernel_list = [kernel_gp, kernel_gp]
    gp = GaussianProcessModel(data, y, multi=False, periodic=True, m=2, kernel_list=kernel_list, verbose=True)


    # Adaptive Epsilon PAL algorithm

    delta = 0.15
    alg_object = AdaptiveEpsilonPAL(problem_model, epsilon=epsilon, delta=delta, gp=gp,
                                    initial_hypercube=Hypercube(1, [0.5]))
    titles = "epsilon" + str(epsilon) + "delta" + str(delta) + "seed" + str(seed)
    pareto_set, pareto_set_cells = alg_object.algorithm(titles = titles, vis=True)



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
        a = np.squeeze(np.array(pareto_nodes_center)).reshape(-1, 1)

        y_obs = problem_model.observe(a, std=0.0)

        # Error metric
        p_set = np.hstack((func_val1[mask].reshape(-1, 1), func_val2[mask].reshape(-1, 1)))
        print(p_set)
        c = 0
        for row in p_set:
            a = y_obs - row
            print("------------------AAAA", a.shape)
            b = np.linalg.norm(a, axis=1)
            c += np.min(b)

        title = "$\epsilon = $" + '%.2f' % epsilon + " $ \delta = $" + '%.2f' % delta + ", Error = " + '%.3f' % (c / p_set.shape[0]) + " " + r'$, \tau $ :' + str(
            tau_eval)
        figtitle = "epsilon" + str(epsilon) + "delta" + str(delta) + "Error" + str(c / p_set.shape[0]) + 'tau' + str(
            tau_eval) + "seed" + str(seed)

        # plot_pareto_front(y1=y_obs[:, 0], y2=y_obs[:, 1], title=title,
        #                   plotfront=True, figtitle=figtitle)
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
        a = np.squeeze(np.array(cells)).reshape(-1, 1)

        y_obs = np.empty((a.shape[0], 2))
        i = 0
        for row in a:
            data_alg = np.array(row).reshape(-1, 1)
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            print(data_alg.shape)
            y = problem_model.observe(data_alg, std=0)
            y_obs[i, :] = y
            i += 1


        # Error metric
        p_set2 = np.hstack((func_val1[mask].reshape(-1, 1), func_val2[mask].reshape(-1, 1)))
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
        figtitle = "epsilon" + str(epsilon) + "delta" + str(delta) + "Error" + str(c2 / p_set2.shape[0]) + 'tau' + str(
            tau_eval) + "seed" + str(seed) + "cell"

        # plot_pareto_front(y1=y_obs[:, 0], y2=y_obs[:, 1], title=title,
        #                   plotfront=True, figtitle=figtitle)
        plot_pareto_front(func_val1, func_val2, mask, y_obs[:, 0], y_obs[:, 1], title=title,
                          plotfront=True, figtitle=figtitle)
        print(tau_eval, c / p_set.shape[0], c2 / p_set2.shape[0], time_elapsed, epsilon, seed, hmax)
        #return tau_eval, c / p_set.shape[0], time_elapsed, epsilon, seed




    else:
        return -1, -1, -1, -1, -1, -1


if __name__ == "__main__":
    worker1((0.1, 7))
    worker1((0.2, 7))
    worker1((0.4, 7))
    worker1((0.05, 7))


    # pool3 = multiprocessing.Pool(processes=1)
    # p3 = pool3.map(worker1, [(0.4, 1)])
    # np.savetxt("test.txt", np.asarray(p3))

    # pool = multiprocessing.Pool(processes=2)
    # p = pool.map(worker1, [(0.4, 5), (0.4, 6)])
    # np.savetxt("04_yeslsm_norm2.txt", np.asarray(p))
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



