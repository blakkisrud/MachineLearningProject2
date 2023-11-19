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

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

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

def cross_entropy_loss(yhat, y, lmbd=0.0, w=0, l1_ratio=0.0):
    if lmbd == 0.0:
        return - (y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
    else:
        # regularization = 0.5 * lmbd * sum(np.sum(weight**2) for weight in w)
        regularization = lmbd * np.sum([(1 - l1_ratio) * np.sum(weights**2) + l1_ratio * np.sum(np.abs(weights)) for weights in w])
        loss = - (y * np.log(yhat) + (1 - y) * np.log(1 - yhat)) + regularization
        # print(loss.shape)
        # sys.exit()
        return loss
    
def cross_entropy_loss_deriv(yhat, y):
    return (yhat - y) / (yhat * (1 - yhat))

"""

Use the class-implementations of the schedulers
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
        #self.moment = 0
        #self.second = 0

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
                             cost_func, # Not in use
                             gradient_cost_func,
                             epsilon = 1.0e-4,
                             max_iterations = 1000000,
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
    change_vector = []

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

            change_vector.append(change)

            beta -= change

            y_pred = X.dot(beta)

            mse = np.mean((y-y_pred)**2.0)

            mse_in_batch[i] = mse
            beta_vector.append(beta)

        scheduler.reset()

        print("Now doing epoch: ", epoch, end="\r")

        # Find the optimal beta

        smallest_mse_in_batch = np.min(mse_in_batch)
        optimal_beta = beta_vector[np.argmin(mse_in_batch)]

        mse_by_epoch.append(smallest_mse_in_batch)
        beta_by_epoch.append(optimal_beta)

    optimal_beta_over_epochs = beta_by_epoch[np.argmin(mse_by_epoch)]
    smallest_mse_over_epochs = np.min(mse_by_epoch)

    return optimal_beta_over_epochs, smallest_mse_over_epochs, mse_by_epoch, change_vector

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

    np.random.seed(42) # TODO: Set global seed

    return (a0 + a1*x + a2*x*x) + np.random.randn(len(x))*noise_sigma

def standard_simple_function_dataset(x = np.arange(0, 10, 0.01), a0=1, a1=5, a2=3, noise_sigma = 0.2, random_state = 42,
                             test_size = 0.2, scale = True, dims_in_design_matrix = 1):
    """
    Return the simple function with the design matrix, training and evaluation data 
    set to perform deterministic tests

    """

    y = simple_func(x, a0, a1, a2, noise_sigma = noise_sigma)

    X = one_d_design_matrix(x, dims_in_design_matrix)

    if scale:

        X = X[:,1]
        X = X.reshape(-1, 1)

        X_scaler = StandardScaler()
        y_scaler = MinMaxScaler(feature_range=(0, 1))

        X = X_scaler.fit_transform(X)
        y = y_scaler.fit_transform(y.reshape(-1, 1)).reshape(-1, 1)

        y = y.reshape(-1, 1)

    else:

        X = X
        y = y.reshape(-1, 1)
        
    idx = np.arange(len(y))

    idx_train, idx_test = train_test_split(idx, test_size=test_size, 
                                           random_state=random_state)
    
    X_train = X[idx_train]
    X_eval = X[idx_test]

    y_train = y[idx_train]
    y_eval = y[idx_test]

    return X_train, X_eval, y_train, y_eval, X, y, x, idx_train, idx_test



    
    x = np.arange(0, 10, 0.01)
    y = utils.simple_func(x, 1, 5, 3, noise_sigma=0.2)


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

def sgd_tuning(X, y, fixed_params, list_of_etas, 
               list_of_batch_sizes, k_folds):

    """
    Function to perform k-fold cross validation
    hp-tuning of the sgd-function
    """

    grid_table = pd.DataFrame(columns=["eta", "batch_size", "Fold", "MSE_test"])

    for eta in list_of_etas:

        for batch_size in list_of_batch_sizes:

            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            
            current_fold = 1 # Hacky

            for train_index, test_index in kfold.split(X):

                try:

                    scheduler = fixed_params["scheduler"]
                    scheduler.eta = eta # TODO: Do checks for the different schedulers

                    run = general_stochastic_gradient_descent(X[train_index], y[train_index],
                                                                np.random.randn(3, 1), 
                                                                mini_batch_size=batch_size,
                                                                scheduler=scheduler,
                                                                epochs=fixed_params["epochs"],
                                                                cost_func=fixed_params["cost_func"],
                                                                gradient_cost_func=fixed_params["gradient_cost_func"],
                                                                return_diagnostics=True)
                    
                    optimal_beta = run[0] # TODO: Use dict
                    mse_training = run[1]

                    y_pred = X[test_index] @ optimal_beta

                    mse = mean_squared_error(y[test_index], y_pred)

                    run_table = pd.DataFrame({"eta": eta,
                                                "batch_size": batch_size,
                                                "Fold": current_fold,
                                                "MSE_test": mse}, index=[0])
                    
                    grid_table = grid_table._append(run_table, ignore_index=True)

                    current_fold += 1

                except Exception as e:

                    run_table = pd.DataFrame({"eta": eta,
                                                "batch_size": batch_size,
                                                "Fold": current_fold,
                                                "MSE_test": np.nan}, index=[0])
                    
                    grid_table = grid_table._append(run_table, ignore_index=True)

    return grid_table

def gd_tuning(X, y, fixed_params, list_of_etas, k_folds):

    """
    Function to perform k-fold cross validation
    hp-tuning of the gd-function
    """

    grid_table = pd.DataFrame(columns=["eta", "Fold", "MSE_test"])

    scheduler = fixed_params["scheduler"]

    for eta in list_of_etas:

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        current_fold = 1 # Hacky
        scheduler.eta = eta # TODO: Do checks for the different schedulers

        for train_index, test_index in kfold.split(X):

            try:

                run = general_gradient_descent(X[train_index], y[train_index],
                                                np.random.randn(3, 1), 
                                                scheduler=scheduler,
                                                max_iterations=fixed_params["max_iterations"],
                                                cost_func=fixed_params["cost_func"],
                                                gradient_cost_func=fixed_params["gradient_cost_func"],
                                                return_diagnostics=False)
                
                optimal_beta = run

                y_pred = X[test_index] @ optimal_beta

                try:

                    mse = mean_squared_error(y[test_index], y_pred)

                except Exception as e:

                    mse = np.nan

                run_table = pd.DataFrame({"eta": eta,
                                            "Fold": current_fold,
                                            "MSE_test": mse}, index=[0])
                
                grid_table = grid_table._append(run_table, ignore_index=True)

                current_fold += 1

                scheduler.reset()


            except Exception as e:

                print(e)

                run_table = pd.DataFrame({"eta": eta,
                                            "Fold": current_fold,
                                            "MSE_test": np.nan}, index=[0])
                
                grid_table = grid_table._append(run_table, ignore_index=True)

    return grid_table

def ridge_tuning(X_train, y,  lmbd_list, k_folds):

    """
    Function to perform k-fold cross validation
    hp-tuning of the ridge-function
    """

    grid_table = pd.DataFrame(columns=["lambda", "Fold", "MSE_test"])

    for lmbd in lmbd_list:

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        current_fold = 1 # Hacky

        for train_index, test_index in kfold.split(X_train):

            try:

                beta_ridge = Ridge(X_train[train_index], y[train_index], lmbd)

                y_pred = X_train[test_index] @ beta_ridge

                mse = mean_squared_error(y[test_index], y_pred)

                run_table = pd.DataFrame({"lambda": lmbd,
                                            "Fold": current_fold,
                                            "MSE_test": mse}, index=[0])
                
                grid_table = grid_table._append(run_table, ignore_index=True)

                current_fold += 1

            except Exception as e:

                run_table = pd.DataFrame({"lambda": lmbd,
                                            "Fold": current_fold,
                                            "MSE_test": np.nan}, index=[0])
                
                grid_table = grid_table._append(run_table, ignore_index=True)

    return grid_table

def optimal_hyper_params(grid_table, min_col = "MSE_test"):
    """
    Function to find the optimal hyper parameters

    Input is a table of ND grid searches and the column
    to minimize on

    Assumes that the grid_table has been made with 
    a k-fold cross validation, and that the dataframe
    only contains the hyper parameters and the MSE
    in addition to the folds

    TODO: Ensure that this is the case, as this is a bit 
    hacky right now

    Return values are the optimal hyper parameters
    in a name-value dictionary

    """

    cols_in_frame = list(grid_table.columns)
    # Remove string from list

    cols_in_frame.remove(min_col)
    cols_in_frame.remove("Fold")

    hp_params = cols_in_frame

    average_mse_table = grid_table.groupby(hp_params).mean()

    average_mse_table.drop(columns = ["Fold"], inplace=True)

    min_mse_index = average_mse_table[min_col].idxmin()
    index_names = average_mse_table.index.names

    if len(index_names) == 1:

        min_index_values = [min_mse_index]

    else:

        min_index_values = list(min_mse_index)

    min_mse_value = average_mse_table.loc[min_mse_index, min_col]

    optimal_hp = dict(zip(index_names, min_index_values))

    return optimal_hp, min_mse_value

def ffn_tuning(X, y, net, folds, hp_dict):

    """

    """

    hp_names = list(hp_dict.keys())

    for hp in hp_names:

        for hp_val in hp_dict[hp]:

            kfold = KFold(n_splits=folds, shuffle=True, random_state=42)

            for train_index, test_index in kfold.split(X):

                X_train = X[train_index]
                X_test = X[test_index]

                y_train = y[train_index]
                y_test = y[test_index]







    return None

def plot_heatmap_from_tuning(grid_table, param1, param2):

    fig, ax = plt.subplots(figsize=(10, 10))
    grid_table_pivot = grid_table.pivot(param1, param2, "MSE_test")
    sns.heatmap(grid_table_pivot, annot=True, ax=ax)
    plt.show()

class exploratory_data_analysis():
    def __init__(self):
        pass

    def update_attr(self, attr):
        for key, val in attr.items():
            self.__setattr__(key, val)
        print("Updating attributes:", attr)

