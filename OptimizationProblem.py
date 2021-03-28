import numpy as np


class OptimizationProblem:
    def __init__(self, func_list):
        self.func_list = func_list
        self.m = len(func_list)
        self.cardinality = 121
        self.N = 2
        self.D_1 = 2
        self.alpha = 1
        self.v = 7
        self.L = 0.2

    def observe(self, x, std=0):
        obs = np.array([func(x) + std * np.random.randn(x.shape[0], ) for func in self.func_list]).T
        return obs

    # def observe(self, x):
    # find the closest
    ##observe that in the data

