import numpy as np
import pandas as pd
import os

import gpflow as gpf
from gpflow.utilities import print_summary

from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Load the dataset into a data frame
d = os.path.dirname(os.getcwd())
data = pd.read_csv(d + "\Data\data.txt", sep=';',
                   header=None).to_numpy()

# Standardize the design space and the objectives
scaler = preprocessing.MinMaxScaler()
data[:, :3] = scaler.fit_transform(data[:, :3])
# data[:, 3:] = preprocessing.MinMaxScaler().fit_transform(data[:, 3:]) * 2 - 1

plt.scatter(data[:, 0], data[:, 3])
plt.title("Input 1, Objective 1")
plt.show()
plt.scatter(data[:, 0], data[:, 4])
plt.title("Input 1, Objective 2")
plt.show()
plt.scatter(data[:, 1], data[:, 3])
plt.title("Input 2, Objective 1")
plt.show()
plt.scatter(data[:, 1], data[:, 4])
plt.title("Input 2, Objective 2")
plt.show()
plt.scatter(data[:, 2], data[:, 3])
plt.title("Input 3, Objective 1")
plt.show()
plt.scatter(data[:, 2], data[:, 4])
plt.title("Input 3, Objective 2")
plt.show()



fig, axs = plt.subplots(1, 2, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].hist(data[:, 3], bins=20)
axs[1].hist(data[:, 4], bins=20)
plt.title("Objective 2")
plt.show()

X = data[:, :3]
Y = data[:, 4, None]

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(X[:, 0], X[:, 1], X[:, 2], s = 100, c=Y.flatten(), cmap = plt.get_cmap("magma"))
plt.title("Objective 2")
plt.show()


Y = data[:, 3, None]

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(X[:, 0], X[:, 1], X[:, 2], s = 100, c=Y.flatten(), cmap = plt.get_cmap("magma"))
plt.show()



np.random.seed(7)
# Randomly choose instances to use in GP initialization, sample from the rest
np.random.shuffle(data)
gp_split = data[:40]
sample_split = data[40:]

X = gp_split[:, :3]
Y = gp_split[:, 3, None]
kernel = gpf.kernels.SquaredExponential(lengthscales=[1, 1, 2])
m = gpf.models.GPR(data=(X, Y), kernel=kernel, noise_variance=1)

print_summary(m)
print(m.log_marginal_likelihood())

opt = gpf.optimizers.Scipy()
opt.minimize(
                    m.training_loss,
                    variables=m.trainable_variables,
                    method="l-bfgs-b",
                    options={"disp": False, "maxiter": 100}
                )

print_summary(m)
print(m.log_marginal_likelihood())



