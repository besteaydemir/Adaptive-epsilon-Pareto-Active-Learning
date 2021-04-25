import numpy as np
from Hypercube import Hypercube
from typing import List
from math import sqrt
from statistics import mean
import gpflow
import tensorflow as tf
from gpflow.utilities import print_summary


class GaussianProcessModel:
    def __init__(self, X, Y, multi, periodic, m, kernel_list, verbose=False):
        self.X = X
        self.Y = Y

        self.multi = multi
        self.m = m

        self.variance_list = [] #list of size m
        self.lengthscales_list = []
        self.kernelperiod_list = []

        self.v = None
        self.L = None

        self.kernel_list = kernel_list

        self.verbose = verbose

        if multi:
            pass
            # self.model=multioutput()
        else:
            self.model = self.gp_list()


    def gp_list(self):
        #np.random.seed(134340)
        gp_list = []
        opt = gpflow.optimizers.Scipy()

        for i in range(self.m):
            #print(self.Y[:,i].shape)  # Yshape problems
            kernel = self.kernel_list[i]
            m = gpflow.models.GPR(data=(self.X, self.Y[:,i].reshape(-1,1)), kernel=kernel, noise_variance=0.1)
            print(m.log_marginal_likelihood())

            if len(self.variance_list) < 2 or len(self.lengthscales_list) < 2:
            #if True:

                # Tune the model parameters according to data
                opt.minimize(
                    m.training_loss,
                    variables=m.trainable_variables,
                    method="l-bfgs-b",
                    options={"disp": False, "maxiter": 100}
                )
                # self.lengthscales_list.append(m.kernel.base_kernel.lengthscales)
                # self.variance_list.append(m.kernel.base_kernel.variance)
                # self.kernelperiod_list.append(m.kernel.period)


                self.lengthscales_list.append(m.kernel.lengthscales)
                self.variance_list.append(m.kernel.variance)

                print(m.kernel.lengthscales)
                print(m.kernel.lengthscales.numpy())



            else:
                # m.kernel.base_kernel.lengthscales.assign(self.lengthscales_list[i])
                # m.kernel.base_kernel.variance.assign(self.variance_list[i])
                # m.kernel.period.assign(self.kernelperiod_list[i])

                m.kernel.lengthscales.assign(self.lengthscales_list[i])
                m.kernel.variance.assign(self.variance_list[i])




            gp_list.append(m)

            if True: #self.verbose:
                print("For objective function ", i)
                print_summary(m)
                print("Log likelihood after optimization: ", tf.keras.backend.get_value(m.log_marginal_likelihood()))

        # self.L = (self.lengthscales_list[0][1].numpy() + self.lengthscales_list[0][0].numpy() + self.lengthscales_list[1][0].numpy()
        #            + self.lengthscales_list[1][1].numpy() + self.lengthscales_list[0][2].numpy()
        #            + self.lengthscales_list[1][2].numpy()) / 6
        self.L = (self.lengthscales_list[0].numpy() + self.lengthscales_list[1].numpy()) / 2
        print("LLLL", self.L)

        self.v = (self.variance_list[0].numpy() + self.variance_list[1].numpy()) / 2

        return gp_list

    # def gp_multioutput(self, mean_list=None, kernel_list=None):
    #     return gp

    def inference(self, x):
        mus = np.empty((self.m,1))
        #sigmas = np.empty((self.m, 1))
        var = np.empty((self.m, 1))

        if self.multi:
            mu, sigma = self.model.predict_f(x)
            #todo
        else:
            for i, gp in enumerate(self.model):
                mus[i], var[i] = gp.predict_f(x)

        return mus, np.sqrt(var)

    def update(self, x, y):
        self.X = np.vstack((self.X, x))
        self.Y = np.vstack((self.Y, y))
        self.model = self.gp_list()
