import numpy as np
import gpflow
import tensorflow as tf
from gpflow.utilities import print_summary


class GaussianProcessModel:
    def __init__(self, kernel_list, d, verbose=False):
        """
        Class for handling GP objects.

        :param kernel_list: Kernel list given in the beginning.
        :param d: Input dimension.
        :param verbose:
        """
        self.m = len(kernel_list)

        self.X = np.array([[-1e9] * d])
        self.Y = np.array([[0.] * self.m]).reshape(1, self.m)

        self.variance_list = []
        self.lengthscales_list = []

        self.v = None
        self.L = None

        self.kernel_list = kernel_list

        self.verbose = verbose


        self.model = self.gp_list()


    def gp_list(self):
        gp_list = []

        for i in range(self.m):

            kernel = self.kernel_list[i]
            m = gpflow.models.GPR(data=(self.X, self.Y[:, i].reshape(-1, 1)), kernel=kernel, noise_variance=0.1)

            self.lengthscales_list.append(m.kernel.lengthscales)
            self.variance_list.append(m.kernel.variance)
            gp_list.append(m)

            if self.verbose is True:
                print("For objective function ", i)
                print_summary(m)
                print("Log likelihood ", tf.keras.backend.get_value(m.log_marginal_likelihood()))


        self.L = np.array([ls.numpy() for ls in self.lengthscales_list]).mean()
        self.v = np.array([var.numpy() for var in self.variance_list]).mean()

        return gp_list


    def inference(self, x):
        mus = np.empty((self.m, 1))
        var = np.empty((self.m, 1))

        for i, gp in enumerate(self.model):
            mus[i], var[i] = gp.predict_f(x)

        return mus, np.sqrt(var)


    def update(self, x, y):
        self.X = np.vstack((self.X, x))
        self.Y = np.vstack((self.Y, y))
        self.model = self.gp_list()
