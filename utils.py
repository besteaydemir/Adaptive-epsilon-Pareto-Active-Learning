from collections import Sequence

import numpy as np
import scipy as sp


def dominated_by(v1, v2, epsilon=None):  # Weakly
    """
    Checks if one hypercube is dominated by another.
    :param v1: List
    :param v2: List
    :param epsilon: Accuracy level given as input to the algorithm
    :return:
    """

    v11 = np.asarray(v1).reshape(-1,1)
    v22 = np.asarray(v2).reshape(-1,1)

    if epsilon is not None:
        return np.all(v11 <= v22 + epsilon)
    return np.all(v1 <= v2)


def printl(list):
    """
    List print function.

    :param list: List of nodes.
    :return:
    """
    print("~~~~List begin~~~~~~~~~~~~~~~~~~~~~~~~~~", len(list))
    for node in list:
        print(node)
    print("~~~~List end~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


# Code taken from https://github.com/befelix/SafeOpt
def linearly_spaced_combinations(bounds, num_samples):
    """
    Return 2-D array with all linearly spaced combinations with the bounds.

    Parameters
    ----------
    bounds: sequence of tuples
        The bounds for the variables, [(x1_min, x1_max), (x2_min, x2_max), ...]
    num_samples: integer or array_likem
        Number of samples to use for every dimension. Can be a constant if
        the same number should be used for all, or an array to fine-tune
        precision. Total number of data points is num_samples ** len(bounds).

    Returns
    -------
    combinations: 2-d array
        A 2-d arrray. If d = len(bounds) and l = prod(num_samples) then it
        is of size l x d, that is, every row contains one combination of
        inputs.
    """
    num_vars = len(bounds)

    if not isinstance(num_samples, Sequence):
        num_samples = [num_samples] * num_vars

    if len(bounds) == 1:
        return np.linspace(bounds[0][0], bounds[0][1], num_samples[0])[:, None]

    # Create linearly spaced test inputs
    inputs = [np.linspace(b[0], b[1], n) for b, n in zip(bounds,
                                                         num_samples)]

    # Convert to 2-D array
    return np.array([x.ravel() for x in np.meshgrid(*inputs)]).T


# Code taken from https://github.com/befelix/SafeOpt
def sample_gp_function(kernel, bounds, noise_var, num_samples,
                           interpolation='kernel', mean_function=None):
        """
        Sample a function from a gp with corresponding kernel within its bounds.

        Parameters
        ----------
        kernel: instance of GPy.kern.*
        bounds: list of tuples
            [(x1_min, x1_max), (x2_min, x2_max), ...]
        noise_var: float
            Variance of the observation noise of the GP function
        num_samples: int or list
            If integer draws the corresponding number of samples in all
            dimensions and test all possible input combinations. If a list then
            the list entries correspond to the number of linearly spaced samples of
            the corresponding input
        interpolation: string
            If 'linear' interpolate linearly between samples, if 'kernel' use the
            corresponding mean RKHS-function of the GP.
        mean_function: callable
            Mean of the sample function

        Returns
        -------
        function: object
            function(x, noise=True)
            A function that takes as inputs new locations x to be evaluated and
            returns the corresponding noisy function values. If noise=False is
            set the true function values are returned (useful for plotting).
        """
        inputs = linearly_spaced_combinations(bounds, num_samples)
        cov = kernel(inputs) + np.eye(inputs.shape[0]) * 1e-6
        output = np.random.multivariate_normal(np.zeros(inputs.shape[0]),
                                               cov)

        if interpolation == 'linear':

            def evaluate_gp_function_linear(x, noise=True):
                """
                Evaluate the GP sample function with linear interpolation.

                Parameters
                ----------
                x: np.array
                    2D array with inputs
                noise: bool
                    Whether to include prediction noise
                """
                x = np.atleast_2d(x)
                y = sp.interpolate.griddata(inputs, output, x, method='linear')

                # Work around weird dimension squishing in griddata
                y = np.atleast_2d(y.squeeze()).T

                if mean_function is not None:
                    y += mean_function(x)
                if noise:
                    y += np.sqrt(noise_var) * np.random.randn(x.shape[0], 1)
                return y

            return evaluate_gp_function_linear

        elif interpolation == 'kernel':
            cho_factor = sp.linalg.cho_factor(cov)
            alpha = sp.linalg.cho_solve(cho_factor, output)
            # print(alpha)
            # print("alpha")


            def evaluate_gp_function_kernel(x, noise=True):
                """
                Evaluate the GP sample function with kernel interpolation.

                Parameters
                ----------
                x: np.array
                    2D array with inputs
                noise: bool
                    Whether to include prediction noise
                """
                x = np.atleast_2d(x)
                y = kernel(x, inputs).numpy().dot(alpha)

                y = y[:, None]
                if mean_function is not None:
                    y += mean_function(x)
                if noise:
                    y += np.sqrt(noise_var) * np.random.randn(x.shape[0], 1)
                return y

            return evaluate_gp_function_kernel
