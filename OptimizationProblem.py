import numpy as np


class OptimizationProblem:
    def __init__(self, func_list = None, dataset = None, scaler=None):
        self.func_list = func_list
        if func_list is not None:
            self.m = len(func_list)
        self.m = 2
        if scaler is not None:
            self.scaler =scaler
        self.cardinality = 2500
        self.N = 2
        self.D_1 = 2
        self.alpha = 1

        if dataset is not None:
            self.x = dataset[0]
            self.y = dataset[1]

    def observe(self, x, std=0):
        if self.func_list is not None:
            print("xxxxxxxxxxxxxxxxxx", x.shape)
            print(self.func_list[0](x).shape)
            obs = np.array([func(x).reshape(-1, ) + std * np.random.randn(x.shape[0], ) for func in self.func_list]).T
        else:
            #xinv = self.scaler.inverse_transform(x)
            xinv = x
            # Find the closest one
            a = (self.x - xinv)
            #print(x)
            #print(self.x)
            #print(a)

            b = np.linalg.norm(a, axis=1)
            #print("b")
            #print(b)
            c = np.argmin(b)
            #print(c)
            obs = self.y[c] # T?
            #print(obs)

        return obs

    # def observe(self, x):
    # find the closest
    ##observe that in the data

