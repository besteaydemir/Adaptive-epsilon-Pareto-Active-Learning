import numpy as np


class OptimizationProblem:
    def __init__(self, func_list):
        self.func_list = func_list
        self.m = len(func_list)
        self.D_1 = 2
        self.alpha = 1
        self.problem_model.v
        self.problem_model.L

    def observe(self, x, std=0):
        obs = [func(x) + std * np.random.randn(self.m, ) for func in self.func_list]
        return obs

    # def observe(self, x):
    # find the closest
    ##observe that in the data

