import time
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import gpflow as gpf
from utils import *

noise_var = 1e-5
bounds = [(-3., 3.), (-3,5)]


kernel = gpf.kernels.Matern32(lengthscales=[1,2])
fun = sample_gp_function(kernel, bounds, noise_var, 50)
print("a")



x = np.linspace(-3, 3, 13)
t = time.time()
param_set = linearly_spaced_combinations(bounds, 30)
print(param_set.shape)
t2 = time.time()
print(t2-t)
print("besss")
print(fun(param_set).shape)

plt.plot(param_set[0],param_set[0], fun(param_set))

plt.show()