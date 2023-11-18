"""

Utility functions and library functions for project 2

"""

import numpy as np
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from autograd import elementwise_grad
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

"""
LOSS FUNCTIONS'
- MSE 
- CROSS-ENTROPY
"""

def mse_loss(yhat, y, lmbd=0.0, w=0):
    if lmbd == 0.0:
        return np.square(yhat - y)
    else:
        l2_regularization = 0.5 * lmbd * sum(np.sum(weight**2) for weight in w)
        return np.square(yhat - y) + l2_regularization
    
def mse_loss_deriv(yhat, y):
    return 2 * (yhat - y)

def cross_entropy_loss(yhat, y, lmbd=0.0, w=0):
    if lmbd == 0.0:
        return - (y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
    else:
        l2_regularization = 0.5 * lmbd * sum(np.sum(weight**2) for weight in w)
        return - (y * np.log(yhat) + (1 - y) * np.log(1 - yhat)) + l2_regularization
    
def cross_entropy_loss_deriv(yhat, y):
    return (yhat - y) / (yhat * (1 - yhat))

"""

Use the classe-implementations of the schedulers
from the lecture notes

"""
class Scheduler():

    """
    This is the abstracted class
    """

    def __init__(self, eta):

        self.eta = eta

    def update_change(self, gradient):
        raise NotImplementedError

    def reset(self):
        pass

class ConstantScheduler(Scheduler):

    def __init__(self, eta):
        super().__init__(eta)

    def update_change(self, gradient):
        return self.eta*gradient

    def reset(self):
        pass

class MomentumScheduler(Scheduler):

    def __init__(self, eta: float, momentum: float):
        super().__init__(eta)
        self.momentum = momentum
        self.change = 0

    def update_change(self, gradient):
        self.change = self.momentum * self.change + self.eta * gradient
        return self.change

    def reset(self):
        pass

class AdagradScheduler(Scheduler):

    """
    NB! This does not work well with the general gradient descent function
    it runs but does not converge to any nice value
    """
    def __init__(self, eta):
        super().__init__(eta)
        self.G_t = None

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero

        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        self.G_t += gradient @ gradient.T

        G_t_inverse = 1 / (
            delta + np.sqrt(np.reshape(np.diagonal(self.G_t), (self.G_t.shape[0], 1)))
        )

        return self.eta * gradient * G_t_inverse

    def reset(self):
        self.G_t = None

class RMS_propScheduler(Scheduler):
    def __init__(self, eta, rho):
        super().__init__(eta)
        self.rho = rho
        self.second = 0.0

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero
        self.second = self.rho * self.second + (1 - self.rho) * gradient * gradient
        return self.eta * gradient / (np.sqrt(self.second + delta))

    def reset(self):
        self.second = 0.0

class AdamScheduler(Scheduler):
    def __init__(self, eta, rho, rho2):
        super().__init__(eta)
        self.rho = rho
        self.rho2 = rho2
        self.moment = 0
        self.second = 0
        self.n_epochs = 1

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero

        self.moment = self.rho * self.moment + (1 - self.rho) * gradient
        self.second = self.rho2 * self.second + (1 - self.rho2) * gradient * gradient

        moment_corrected = self.moment / (1 - self.rho**self.n_epochs)
        second_corrected = self.second / (1 - self.rho2**self.n_epochs)

        return self.eta * moment_corrected / (np.sqrt(second_corrected + delta))

    def reset(self):
        self.n_epochs += 1
        self.moment = 0
        self.second = 0

# End of the scheduler class definitions
#==================================================================

# Activation functions and ways to calculate the gradients of them

def identity(X):
    return X


def sigmoid(X):
    try:
        return 1.0 / (1 + np.exp(-X))
    except FloatingPointError:
        return np.where(X > np.zeros(X.shape), np.ones(X.shape), np.zeros(X.shape))


def softmax(X):
    # X = X - np.max(X, axis=-1, keepdims=True)
    delta = 10e-10
    return np.exp(X) / (np.sum(np.exp(X), axis=-1, keepdims=True) + delta)


def RELU(X):
    return np.where(X > np.zeros(X.shape), X, np.zeros(X.shape))


def LRELU(X):
    delta = 10e-4
    return np.where(X > np.zeros(X.shape), X, delta * X)


def derivate(func):
    if func.__name__ == "RELU":

        def func(X):
            return np.where(X > 0, 1, 0)

        return func

    elif func.__name__ == "LRELU":

        def func(X):
            delta = 10e-4
            return np.where(X > 0, 1, delta)

        return func

    elif func.__name__ == "sigmoid":

        return sigmoid_derivated

    else:
        return elementwise_grad(func)


# End of activation functions
#==================================================================

def logistic_regression(X_train, y_train, X_eval, y_eval):

    # Create a logistic regression model
    model = LogisticRegression()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_eval)

    # Evaluate the model
    accuracy = accuracy_score(y_eval, y_pred)
    confusion_mat = confusion_matrix(y_eval, y_pred)
    classification_rep = classification_report(y_eval, y_pred)

    print(y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{confusion_mat}")
    print(f"Classification Report:\n{classification_rep}")
    mse = mean_squared_error(y_eval, y_pred)

    return mse, accuracy

def random_forest(X_train, y_train, X_eval, y_eval):


    # Create a logistic regression model
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=100)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_eval)

    # Evaluate the model
    accuracy = accuracy_score(y_eval, y_pred)
    confusion_mat = confusion_matrix(y_eval, y_pred)
    classification_rep = classification_report(y_eval, y_pred)

    print(y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{confusion_mat}")
    print(f"Classification Report:\n{classification_rep}")
    mse = mean_squared_error(y_eval, y_pred)

    return mse, accuracy

def general_gradient_descent(X, y, beta, scheduler,
                             cost_func,
                             gradient_cost_func,
                             epsilon = 1.0e-4,
                             max_iterations = 100000,
                             return_diagnostics = False):

    """
    This is the general gradient descent, can be performed with
    all cost functions and gradients (that need to be analytically defined)
    """

    if return_diagnostics:
        mse_vector = []
        beta_vector = []
        iteration_vec = []

    n = int((X.shape[0]))

    for iter in range(max_iterations):
        gradient = gradient_cost_func(X, y, beta)
        change = scheduler.update_change(gradient)
        beta -= change
        if (np.linalg.norm(gradient) < epsilon):
            print("Gradient descent converged")
            break

        if return_diagnostics:
            y_predict = X.dot(beta)
            mse = np.mean((y-y_predict)**2.0)

            mse_vector.append(mse)
            beta_vector.append(beta)
            iteration_vec.append(iter)

    print("Number of iterations: ", iter)

    if return_diagnostics:

        diagnostic_output = {"mse": mse_vector,
                             "beta_vector": beta_vector,
                             "iteration": iteration_vec,
                             "beta": beta}

        return diagnostic_output

    else:

        return beta

def general_stochastic_gradient_descent(X, y, beta, 
                                        scheduler,
                                        cost_func,
                                        mini_batch_size,
                                        epochs,
                                        gradient_cost_func,
                                        epsilon = 1.0e-4,
                                        return_diagnostics = False):

    """
    This is the general stochastic gradient descent, can be performed with
    all cost functions and gradients (that need to be analytically defined)
    """

    n = int((X.shape[0]))

    indices = np.arange(n)

    n_iterations = epochs * n // mini_batch_size

    mse_by_epoch = []
    beta_by_epoch = []

    best_mse = 1e9

    for epoch in range(epochs):

        beta_vector = []
        mse_in_batch = np.zeros(n_iterations)

        for iteration, i in zip(range(n_iterations), range(n_iterations)):

            # Sample with replacement

            idx = np.random.choice(indices, size=mini_batch_size, 
                                   replace=False)
            
            xi = X[idx]
            yi = y[idx]

            gradient = gradient_cost_func(xi, yi, beta)
            change = scheduler.update_change(gradient)

            beta -= change

            y_pred = X.dot(beta)

            mse = np.mean((y-y_pred)**2.0)

            mse_in_batch[i] = mse
            beta_vector.append(beta)


        # Find the optimal beta

        smallest_mse_in_batch = np.min(mse_in_batch)
        optimal_beta = beta_vector[np.argmin(mse_in_batch)]

        mse_by_epoch.append(smallest_mse_in_batch)
        beta_by_epoch.append(optimal_beta)

    optimal_beta_over_epochs = beta_by_epoch[np.argmin(mse_by_epoch)]
    smallest_mse_over_epochs = np.min(mse_by_epoch)

    return optimal_beta_over_epochs, smallest_mse_over_epochs

    








        # For the epoch, find the 


            #if (np.linalg.norm(gradient) < epsilon):
            #    print("Gradient descent converged")
            #    break

        #if return_diagnostics:
        #    y_predict = X.dot(beta)
        #    mse = np.mean((y-y_predict)**2.0)

        #    mse_vector.append(mse)
        #    beta_vector.append(beta)
        #    iteration_vec.append(iter)

    print("Number of iterations: ", iter)

        

    if return_diagnostics:

        diagnostic_output = {"mse": mse_vector,
                             "beta_vector": beta_vector,
                             "iteration": iteration_vec,
                             "beta": beta}

        return diagnostic_output

    else:

        return beta

def general_minibatch_gradient_descent(X, y, beta, scheduler,
                                        cost_func,
                                        mini_batch_size,
                                        gradient_cost_func,
                                        epsilon = 1.0e-4,
                                        max_iterations = 100000,
                                        return_diagnostics = False):
    """
    This is the general stochastic gradient descent, can be performed with
    all cost functions and gradients (that need to be analytically defined)
    """

    if return_diagnostics:
        mse_vector = []
        beta_vector = []
        iteration_vec = []

    n = int((X.shape[0]))

    for iter in range(max_iterations):

        shuffled_indices = np.random.permutation(n)
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for i in range(0, n, mini_batch_size):

            X_mini = X_shuffled[i:i+mini_batch_size]
            y_mini = y_shuffled[i:i+mini_batch_size]

            gradient = gradient_cost_func(X_mini, y_mini, beta)
            change = scheduler.update_change(gradient)
            beta -= change
            if (np.linalg.norm(gradient) < epsilon):
                print("Gradient descent converged")
                break

        if return_diagnostics:
            y_predict = X.dot(beta)
            mse = np.mean((y-y_predict)**2.0)

            mse_vector.append(mse)
            beta_vector.append(beta)
            iteration_vec.append(iter)

    print("Number of iterations: ", iter)

    if return_diagnostics:
            
            diagnostic_output = {"mse": mse_vector,
                                "beta_vector": beta_vector,
                                "iteration": iteration_vec,
                                "beta": beta}
    
            return diagnostic_output
    
    else:

        return beta


def time_step_length(t, t0, t1):
    """
    Function to calculate the time step length
    """

    return t0/(t + t1)

def OLS(X, z):
    """
    X: Design matrix
    z: Data

    Returns: MSE, R2, z_tilde, beta

    """

    beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z)
    z_tilde = X.dot(beta)

    return beta

def Ridge(X, z, lmbd):

    """
    X: Design matrix
    z: Data
    lmbd: The lambda parameter

    Returns: MSE, R2, z_tilde, beta

    """

    beta = np.linalg.pinv(X.T.dot(X) + lmbd*np.identity(X.shape[1])).dot(X.T).dot(z)
    z_tilde = X.dot(beta)

    return beta

def simple_func(x, a0, a1, a2, noise_sigma = 0.0):
    """
    Stupid-simple function to test the code
    """

    return (a0 + a1*x + a2*x*x) + np.random.randn(len(x))*noise_sigma

def simple_cost_func(X, y, beta):
    """
    Function to calculate the cost function
    """

    n = int((X.shape[0]))

    return (1.0/n)*np.sum((X@beta - y)**2.0)

def gradient_simple_function(X, y, beta):

    """
    Lets for starter just expand the b0 + b1*x-case 
    and cross our fingers
    """

    n = int((X.shape[0]))

    y = (2.0/n)*X.T@(X@beta - y)

    return y

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

def gradient_descent_with_minibatches(X, y, 
                                      beta, eta, 
                                      minibatch_size = 5, 
                                      VERBOSE = False):

    n_epochs = 50
    n = int((X.shape[0]))

    indices = np.arange(n)

    np.random.seed(42)

    n_iterations = n_epochs * n // minibatch_size

    scores = []

    for epoch in range(n_epochs):
        
        for iteration in range(n_iterations):

            # Sample with replacement

            idx = np.random.choice(indices, size=minibatch_size, 
                                   replace=False) # False for this run to not select same in each iteration
            
            xi = X[idx]
            yi = y[idx]

            gradient = 2/minibatch_size * xi.T.dot(xi.dot(beta) - yi)
            
            beta = beta - eta * gradient

        if VERBOSE:
            print("Now doing epoch: ", epoch)
            print("Current beta: ", beta)

        y_predict = X.dot(beta)

        mse = np.mean((y-y_predict)**2.0)

        scores.append(mse)

    return beta, scores

def sigmoid_th(z):
    f = lambda z: 1 / (1 + np.exp(-z))
    vf = np.vectorize(f)
    return vf(z)

def sigmoid_derivated(z):
    f = lambda z: sigmoid(z) * (1 - sigmoid(z))
    vf = np.vectorize(f)
    return vf(z)

def identity_th(z):
    f = lambda z: z
    vf = np.vectorize(f)
    return vf(z)

def identity_derived(z):
    f = lambda z: 1
    vf = np.vectorize(f)
    return vf(z)

def gradient_descent_with_time_decay(X, y, beta, eta0, minibatch_size=5):

    n_epochs = 50
    n = int((X.shape[0]))

    n_iterations = n_epochs * n // minibatch_size

    scores = []

    eta = eta0

    for epoch in range(1,n_epochs+1):

        shuffled_indices = np.random.permutation(n)
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for iteration in range(n_iterations):
            t = epoch*n_iterations + iteration
            print(eta)
            eta = time_step_length(float(t), 1.0, 10.0)
            start_idx = iteration * minibatch_size
            end_idx = start_idx + minibatch_size
            xi = X_shuffled[start_idx:end_idx]
            yi = y_shuffled[start_idx:end_idx]
            gradient = 2/minibatch_size * xi.T.dot(xi.dot(beta) - yi)
            beta = beta - eta * gradient

        y_predict = X.dot(beta)

        mse = np.mean((y-y_predict)**2.0)

        scores.append(mse)

    return beta, scores

def RIDGE_with_hp(X, y, hp_dict):

    """
    Ridge with hp-tuning

    """

    if "lmbd" not in hp_dict.keys():

        print("You need to specify a lambda value")
        return None
    
    else:

        lmbd = hp_dict["lmbd"]

    n = int((X.shape[0]))

    beta = np.linalg.pinv(X.T.dot(X) + lmbd*np.identity(X.shape[1])).dot(X.T).dot(y)

    return beta

def k_fold_hyper_parameter_tuning(X, y, function, hp_dict, k = 5):

    """
    Function to perform k-fold cross validation
    TODO: Make this work with all the different ways of performing
    ML
    """

    n = int((X.shape[0]))

    idx = np.arange(n)

    np.random.shuffle(idx)

    idx_folds = np.array_split(idx, k)

    mse_by_lambda = np.zeros(len(hp_dict["lmbd"]))
    
    for i, lmbd in enumerate(hp_dict["lmbd"]):

        mse_by_fold = np.zeros(k)

        for j in range(k):

            idx_test = idx_folds[j]
            idx_train = np.concatenate(idx_folds[:j] + idx_folds[j+1:])

            X_train_fold = X[idx_train,:]
            X_test_fold = X[idx_test,:]

            y_train_fold = y[idx_train]
            y_test_fold = y[idx_test]

            beta_ridge = function(X_train_fold, y_train_fold, lmbd)

            y_pred = X_test_fold @ beta_ridge

            mse = mean_squared_error(y_test_fold, y_pred)

            mse_by_fold[j] = mse

        mse_by_lambda[i] = np.mean(mse_by_fold)

    optimal_lambda = hp_dict["lmbd"][np.argmin(mse_by_lambda)]

    return optimal_lambda

def sgd_tuning(X, y, list_of_etas, list_of_batch_sizes, k_folds, fixed_params):

    """
    Function to perform k-fold cross validation
    hp-tuning of the sgd-function
    """

    grid_table = pd.DataFrame(columns=["eta", "batch_size", "mse"])

    for eta in list_of_etas:

        for batch_size in list_of_batch_sizes:

            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            
            current_fold = 1 # Hacky

            for train_index, test_index in kfold.split(X):

                try:

                    print("Doing SGD")

                except:

                    print("Something went wrong")
                    continue


    return None


