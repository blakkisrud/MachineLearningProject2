"""

Utility functions and library functions for project 2

"""

import numpy as np
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

def simple_func(x, a0, a1, a2):
    """
    Stupid-simple function to test the code
    """

    return a0 + a1*x + a2*x*x

def gradient_simple_function(X, y, beta):

    """
    Lets for starter just expand the b0 + b1*x-case 
    and cross our fingers
    """

    n = int((X.shape[0]))

    return (2.0/n)*X.T@(X@beta - y)

def one_d_design_matrix(x, n):
    """
    Function to create a design matrix for a one-dimensional
    polynomial of degree n
    """

    X = np.zeros((x.shape[0], n+1))

    for i in range(n+1):
        X[:, i] = x**i

    return X

def gradient_descent_step(X, y, beta, eta):
    """
    Function to perform a single gradient descent step

    eta is the learning rate

    """

    n = int((X.shape[0]))

    return beta - eta*(2.0/n)*X.T@(X@beta - y)

def eta_from_hessian(X):
    """
    Function to calculate a suggested learning rate
    from the Hessian matrix
    """

    n = int((X.shape[0]))

    H = (2.0/n)*X.T@X

    EigValues, EigVectors = np.linalg.eig(H)

    return 1.0/np.max(EigValues)

