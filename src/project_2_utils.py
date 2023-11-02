"""

Utility functions and library functions for project 2

"""

import numpy as np
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

def simple_func(x, a0, a1, a2, noise_sigma = 0.0):
    """
    Stupid-simple function to test the code
    """

    return (a0 + a1*x + a2*x*x) + np.random.randn(len(x))*noise_sigma

def gradient_simple_function(X, y, beta):

    """
    Lets for starter just expand the b0 + b1*x-case 
    and cross our fingers
    """

    n = int((X.shape[0]))

    y = (2.0/n)*X.T@(X@beta - y)

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

def gradient_descent(X, y, beta, eta, MaxIterations = 100000, epsilon = 1.0e-8):

    """
    Function to perform gradient descent

    eta is the learning rate

    """

    n = int((X.shape[0]))

    for iter in range(MaxIterations):
        gradient = (2.0/n)*X.T.dot(X.dot(beta)-y)
        beta -= eta*gradient
        if (np.linalg.norm(gradient) < epsilon):
            break

    return beta

def gradient_descent_with_momentum(X, y, beta, eta, 
                                   gamma, MaxIterations = 100000, 
                                   epsilon = 1.0e-8):
    
    """
    Function to perform gradient descent with momentum
    For now simply with the polynomial function
    """

    # Holding now the MSE and the beta-values
    beta_list, scores = [], []

    change = 0.0

    n = int((X.shape[0]))

    for iter in range(MaxIterations):

        gradient = (2.0/n)*X.T.dot(X.dot(beta)-y)
        change = gamma*change + eta*gradient
        beta -= change

        y_predict = X.dot(beta)
        mse = np.mean((y-y_predict)**2.0)

        beta_list.append(beta)
        scores.append(mse)

        #print("Now doing iteration: ", iter)

        if (np.linalg.norm(gradient) < epsilon):
            break

    return (beta, beta_list, scores)


def eta_from_hessian(X):
    """
    Function to calculate a suggested learning rate
    from the Hessian matrix
    """

    n = int((X.shape[0]))

    H = (2.0/n)*X.T@X

    EigValues, EigVectors = np.linalg.eig(H)

    return 1.0/np.max(EigValues)

def gradient_descent_with_minibatches(X, y, beta, eta, minibatch_size = 5, VERBOSE = False):

    n_epochs = 50
    n = int((X.shape[0]))

    minibatch_size = 20

    np.random.seed(42)

    n_iterations = n_epochs * n // minibatch_size

    scores = []

    for epoch in range(n_epochs):
        shuffled_indices = np.random.permutation(n)
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for iteration in range(n_iterations):
            start_idx = iteration * minibatch_size
            end_idx = start_idx + minibatch_size
            xi = X_shuffled[start_idx:end_idx]
            yi = y_shuffled[start_idx:end_idx]
            gradient = 2/minibatch_size * xi.T.dot(xi.dot(beta) - yi)
            beta = beta - eta * gradient

        if VERBOSE:
            print("Now doing epoch: ", epoch)
            print("Current beta: ", beta)

        y_predict = X.dot(beta)

        mse = np.mean((y-y_predict)**2.0)

        scores.append(mse)

        

    return beta, scores