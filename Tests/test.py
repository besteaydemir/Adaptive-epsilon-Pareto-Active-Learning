import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gpflow.utilities import print_summary

# Set up
data = np.genfromtxt("data.txt", delimiter=",")
X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)
X = 6*np.random.randn(10,1)
Y = 2*X**2 + np.random.randn(10,1)

_ = plt.plot(X, Y, "kx", mew=2)
plt.show()

# Choose a kernel
k = gpflow.kernels.Matern52()
print_summary(k)

m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)
m.likelihood.variance.assign(0.01)
m.kernel.lengthscales.assign(4)
print_summary(m)

# Optimize model parameters (kernel, likelihood, mean function)
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
print_summary(m)

# Predictions
## generate test points for prediction
xx = np.linspace(-10, 21, 100).reshape(100, 1)  # test points must be of shape (N, D)

## predict mean and variance of latent GP at test points
mean, var = m.predict_f(xx)
mean1, var1 =
print(mean1, var1)

## generate 10 samples from posterior
tf.random.set_seed(1)  # for reproducibility
samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)

## plot
plt.figure(figsize=(12, 6))
plt.plot(X, Y, "kx", mew=2)
plt.plot(xx, mean, "C0", lw=2)
plt.fill_between(
    xx[:, 0],
    mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
    mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0",
    alpha=0.2,
)

plt.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
#_ = plt.xlim(-0.1, 1.1)
plt.show()