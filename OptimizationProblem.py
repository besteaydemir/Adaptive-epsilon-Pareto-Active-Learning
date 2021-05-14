import numpy as np
import matplotlib.pyplot as plt

class OptimizationProblem:
    def __init__(self, cardinality, N, D_1, func_list = None, dataset = None, scaler=None):
        self.func_list = func_list
        if func_list is not None:
            self.m = len(func_list)
        self.m = 2
        if scaler is not None:
            self.scaler =scaler
        self.cardinality = cardinality
        self.N = N
        self.D_1 = D_1
        self.alpha = 1

        # Observed so far
        self.xlist=[]
        self.ylist=[]

        if dataset is not None:
            self.x = dataset[0]
            self.y = dataset[1]

    def observe(self, x, std=0):
        if self.func_list is not None:
            obs = np.array([func(x).reshape(-1, ) + std * np.random.randn(x.shape[0], ) for func in self.func_list]).T
        else:
            xinv = x
            # Find the closest one
            a = (self.x - xinv)

            b = np.linalg.norm(a, axis=1)
            c = np.argmin(b)
            obs = self.y[c]

        return obs

    def plot_gp_1d(self, rang, gp, x, y):
        xx = np.linspace(rang[0], rang[1], 100).reshape(100, 1)

        ## predict mean and variance of latent GP at test points
        mean, var = gp.model[0].predict_f(xx)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Points
        if self.xlist:
            ax.plot(self.xlist, self.ylist, "kx", mew=2)


        ax.plot(x[0][0], y[:,0][0], "kx", mew=2)

        self.xlist.append(float(x))
        self.ylist.append(float(y[:,0]))

        # Func
        ax.plot(xx, self.func_list[0](xx))

        # GP
        ax.plot(xx, mean, "C6", lw=2)
        ax.fill_between(
            xx[:, 0],
            mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
            mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
            color="C0",
            alpha=0.2,
        )

        plt.xlim([0, 1])
        plt.ylim([-1.5, 1.5])
        plt.show()

    # def observe(self, x):
    # find the closest
    ##observe that in the data

