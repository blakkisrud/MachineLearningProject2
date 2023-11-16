"""
Script to do the 1D analysis for all of the different ways of finding
"""

import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
from random import random, seed
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import pandas as pd

import project_2_utils as utils
from project_2_utils import ConstantScheduler
from project_2_utils import MomentumScheduler
from project_2_utils import AdamScheduler
from project_2_utils import AdagradScheduler
from project_2_utils import RMS_propScheduler

from neural_net import fnn

path_to_output = "plots_1d"

# Make the data set and split into training and test

x = np.arange(0, 20, 0.01)
y = utils.simple_func(x, 1, 5, 3, noise_sigma=0.2)

x = np.arange(0, 10, 0.1)
n = len(x)

X = utils.one_d_design_matrix(x, 2)

y = utils.simple_func(x, 1, 5, 3, noise_sigma=4.0)
y = y.reshape(-1, 1)

idx_train, idx_test = train_test_split(np.arange(n), test_size=0.2)

X_train = X[idx_train]
X_test = X[idx_test]

y_train = y[idx_train]
y_test = y[idx_test]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, 'o', label='Data points', alpha=0.5)

plt.savefig(path_to_output + "/data_points.png")

# Find the hyper-parameters for the different methods
# Use the MSE for hyper-parameter tuning

result_table = pd.DataFrame(columns=["Method", "MSE", "Hyper-parameter"])

# OLS
# Ridge
# SGD
# SGD with momentum plus other schedulers

# OLS - no hyper-parameters

beta_ols = utils.OLS(X_train, y_train)

y_pred = X_test @ beta_ols

mse = mean_squared_error(y_test, y_pred)

run_table = pd.DataFrame({"Method": ["OLS"], "MSE": [mse], "Hyper-parameter": ["None"]})

result_table = result_table._append(run_table)

# Ridge - hyper-parameter is lambda

lambdas = np.logspace(-5, 5, 11)
mse_by_lambda = np.zeros(len(lambdas))

# Do it with k-fold validation
# NB! This should be made into a subroutine/method/function

fold_size = 5
n_folds = int(np.ceil(len(X_train) / fold_size))

for l, i in zip(lambdas, range(len(lambdas))):

    mse_by_fold = np.zeros(n_folds)

    for j in range(n_folds):

        idx_test = np.arange(j * fold_size, (j + 1) * fold_size)
        idx_train = np.setdiff1d(np.arange(len(X_train)), idx_test)

        X_train_fold = X_train[idx_train]
        X_test_fold = X_train[idx_test]

        y_train_fold = y_train[idx_train]
        y_test_fold = y_train[idx_test]

        beta_ridge = utils.Ridge(X_train_fold, y_train_fold, l)

        y_pred = X_test_fold @ beta_ridge

        mse = mean_squared_error(y_test_fold, y_pred)

        mse_by_fold[j] = mse

    mse_by_lambda[i] = np.mean(mse_by_fold)

optimal_lambda = lambdas[np.argmin(mse_by_lambda)]

print(optimal_lambda)

beta_ridge = utils.Ridge(X_train, y_train, optimal_lambda)

y_pred = X_test @ beta_ridge

mse = mean_squared_error(y_test, y_pred)

run_table = pd.DataFrame({"Method": ["Ridge"], "MSE": [mse], "Hyper-parameter": [optimal_lambda]})

result_table = result_table._append(run_table)

print(result_table)

# Now with SGD - hyper-parameters are learning rate, batch size and lambda

# SGD - hyper-parameters are learning rate, batch size and lambda
# SGD with momentum plus other schedulers

list_of_schedulers = [ConstantScheduler(0.01), MomentumScheduler(0.01, 0.9), 
                      AdamScheduler(0.01, 0.9, 0.999), 
                      AdagradScheduler(0.01), 
                      RMS_propScheduler(0.01, 0.9)]

list_of_etas = [0.01, 0.1, 1.0]
list_of_batch_sizes = [1, 5, 10, 20, 50, 100]

for eta in list_of_etas:

    for batch_size in list_of_batch_sizes:

        hp_dict = {"eta": eta, "batch_size": batch_size}

        print(hp_dict)

sys.exit()

scheduler = ConstantScheduler(0.01)
batch_size = 50

output = utils.general_gradient_descent(X_train, y_train,
                                        np.random.randn(3, 1),
                                        scheduler=scheduler,
                                        max_iterations=100000,
                                        cost_func=utils.simple_cost_func,
                                        gradient_cost_func=utils.gradient_simple_function,
                                        return_diagnostics=True)

print(output)

sys.exit()

for eta in list_of_etas:

    for batch_size in list_of_batch_sizes:

        print(eta, batch_size)

        scheduler = ConstantScheduler(eta)

        output = utils.general_minibatch_gradient_descent(X_train, y_train,
                                                            np.random.randn(3, 1),
                                                            scheduler=scheduler,
                                                            batch_size=batch_size,
                                                            max_iterations=100000,
                                                            cost_func=utils.simple_cost_func,
                                                            gradient_cost_func=utils.gradient_simple_function,
                                                            return_diagnostics=True)
        
        beta = output["beta"]

        y_pred = X_test @ beta

        mse = mean_squared_error(y_test, y_pred)

        run_table = pd.DataFrame({"Method": ["SGD"], "MSE": [mse], "Hyper-parameter": [eta]})

        result_table = result_table._append(run_table)





sys.exit()

for l, i in zip(lambdas, range(len(lambdas))):

    beta_ridge = utils.Ridge(X_train, y_train, l)

    y_pred = X_test @ beta_ridge

    mse = mean_squared_error(y_test, y_pred)

    mse_by_lambda[i] = mse

optimal_lambda = lambdas[np.argmin(mse_by_lambda)]

run_table = pd.DataFrame({"Method": ["Ridge"], "MSE": [mse], "Hyper-parameter": [optimal_lambda]})

result_table = result_table._append(run_table)
