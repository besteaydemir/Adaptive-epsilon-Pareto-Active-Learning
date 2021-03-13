import numpy as np
from Hypercube import Hypercube
from typing import List
from math import sqrt
from statistics import mean
import gpflow
from gpflow.utilities import print_summary


class GaussianProcessModel:
    def __init__(self, X, Y, multi, m, mean_list, kernel_list, verbose=False):
        self.X = X
        self.Y = Y

        self.multi = multi
        self.m = m

        self.mean_list = mean_list
        self.kernel_list = kernel_list

        if multi:
            pass
            #self.model=multioutput()
        else:
            self.model = self.gp_list(verbose)

    # def gp_multioutput(self, mean_list=None, kernel_list=None):
    #     return gp

    def gp_list(self, verbose=False):
        """
        Returns a list of tuned GP models.
        :param kernel_list:
        :param mean_list:
        :param verbose:
        :return:
        """
        gp_list = []
        opt = gpflow.optimizers.Scipy()
        for i in range(self.m):
            kernel = self.kernel_list[i]
            mean_func = self.mean_list[i]
            m = gpflow.models.GPR(data=(self.X, self.Y[i]), kernel=kernel, mean_function=mean_func)

            # Tune the model parameters
            opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

            gp_list[i] = m

            if verbose:
                print_summary(m)

        return gp_list

    def inference(self, x):
        if self.multi:
            mu, sigma = self.model.predict_y(x)
        else:
            for gp in self.model:
                mu, sigma = gp.predict_y(x)

        return mu, sigma

    def update(self, x, y):
        self.X.append(x)
        self.y.append(y)

        self.model = self.gp_list()
