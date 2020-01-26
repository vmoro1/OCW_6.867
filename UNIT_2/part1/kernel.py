import numpy as np


def polynomial_kernel(X, Y, c, p):
    """
        Computes the polynomial kernel between two matrices X and Y:
        K(x, y) = (<x, y> + c)^p for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    return (X @ Y.T + c) ** p


def rbf_kernel(X, Y, gamma):
    """
        Computes the Gaussian RBF kernel between two matrices X and Y:
        K(x, y) = exp(-gamma ||x-y||^2) for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    ##x - y för varje rad i y för varje rad i x
#    K = []
#    for x_vec in X:
#        K_row = []
#        for y_vec in Y:
#            diff = x_vec - y_vec
#            norm = np.linalg.norm(diff)
#            norm **= 2
#            K_row.append(norm)
#        K.append(K_row)
#    K = np.array(K)
#    K *= -gamma
#    K = np.exp(K)
#    return K
#   Vectorized solution
    X_sq = (X ** 2).sum(axis=1, keepdims=True)
    Y_sq = (Y ** 2).sum(axis=1, keepdims=True)
    K = -gamma * (X_sq + Y_sq.T - 2 * X @ Y.T)
    return np.exp(K)