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
from sklearn.model_selection import KFold

import pandas as pd

import project_2_utils as utils
from project_2_utils import ConstantScheduler
from project_2_utils import MomentumScheduler
from project_2_utils import AdamScheduler
from project_2_utils import AdagradScheduler
from project_2_utils import RMS_propScheduler

from neural_net import fnn

path_to_output = "plots_1d"

random_state = 42

# Make the data set and split into training and test

X_train, X_test, y_train, y_test, X, y, x, idx_train, idx_test = utils.standard_simple_function_dataset(dims_in_design_matrix=2, 
                                                                                            scale=False)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, 'o', label='Data points', alpha=0.5)

plt.savefig(path_to_output + "/data_points.png")

result_table = pd.DataFrame(columns=["Method", "MSE", "Optimal hyper-parameter"])

# OLS
# Ridge
# SGD
# SGD with momentum plus other schedulers

DO_OLS = False
DO_SGD = False
DO_GD = False
DO_FFN = True

if DO_OLS:

# OLS - no hyper-parameters

    beta_ols = utils.OLS(X_train, y_train)

    y_pred = X_test @ beta_ols

    mse = mean_squared_error(y_test, y_pred)

    run_table = pd.DataFrame({"Method": ["OLS"], "MSE": [mse], "Optimal hyper-parameter": ["None"]})

    result_table = result_table._append(run_table)

    ax.plot(x[idx_test], y[idx_test], 'o', label='Data points', alpha=0.5)
    ax.plot(x[idx_test], y_pred, 'o', label='OLS', alpha=0.5)

    plt.legend()

    plt.savefig(path_to_output + "/ols.png")

    # Ridge - hyper-parameter is lambda
    
    lambdas = np.logspace(-5, 5, 11)

    foo = utils.ridge_tuning(X_train, y_train, lambdas, k_folds=3)

    utils.optimal_hyper_params(foo)

    mse_by_lambda = np.zeros(len(lambdas))

    # Do it with k-fold validation
    # NB! This should be made into a subroutine/method/function

    n_folds = 3

    for l, i in zip(lambdas, range(len(lambdas))):

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        current_fold = 1 # Hacky

        mse_by_fold = np.zeros(n_folds)

        for train_index, test_index in kfold.split(X_train):

            beta_ridge = utils.Ridge(X_train[train_index], y_train[train_index], l)

            y_pred = X_train[test_index] @ beta_ridge

            mse = mean_squared_error(y_train[test_index], y_pred)

            mse_by_fold[current_fold-1] = mse

            current_fold += 1

        mse_by_lambda[i] = np.mean(mse_by_fold)
        print("MSE by lambda:", mse_by_lambda[i], "Lambda:", l)

    optimal_lambda = lambdas[np.argmin(mse_by_lambda)]

    optimal_hp_params = {"lambda": optimal_lambda}

    beta_ridge = utils.Ridge(X_train, y_train, optimal_lambda)

    y_pred = X_test @ beta_ridge

    mse = mean_squared_error(y_test, y_pred)

    run_table = pd.DataFrame({"Method": ["Ridge"], "MSE": [mse], "Optimal hyper-parameter": [optimal_hp_params]})

    result_table = result_table._append(run_table)

    print(result_table)

    x_test = x[idx_test]
    y_test = y[idx_test]

    #ax.plot(x_test, y_test, 'o', label='Data points', alpha=0.5)
    ax.plot(x_test, y_pred, 'o', label='OLS', alpha=0.5)

# Now with SGD - hyper-parameters are learning rate, batch size and lambda

# SGD - hyper-parameters are learning rate, batch size and lambda
# SGD with momentum plus other schedulers

if DO_SGD:

    list_of_schedulers = [ConstantScheduler(0.01), 
                          MomentumScheduler(0.01, 0.9), 
                          AdamScheduler(0.01, 0.9, 0.999),
                          RMS_propScheduler(0.01, 0.9)
                          ]

    list_of_etas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    list_of_batch_sizes = [10, 20, 30]
    k_fold = 3

    for scheduler in list_of_schedulers:

        fixed_params = {"cost_func": utils.simple_cost_func,
                        "gradient_cost_func": utils.gradient_simple_function,
                        "epochs": 10,  # TODO: Remove
                        "scheduler": scheduler}

        grid_table = pd.DataFrame(columns=["eta", "batch_size", "MSE_test", "Fold"])

        tuning_table = utils.sgd_tuning(X_train, y_train, 
                         fixed_params, 
                         list_of_etas, 
                         list_of_batch_sizes, 
                         k_fold)
        
        print(tuning_table)

        optimal_hp_params, min_mse_training = utils.optimal_hyper_params(tuning_table)

        method_string = "SGD" + "-", str(scheduler.__class__.__name__)

        # Evaluate with the optimal hyper-parameters

        scheduler_optimal = scheduler
        scheduler.eta = optimal_hp_params["eta"]

        optimal_beta, mse_training = utils.general_stochastic_gradient_descent(
            X_train, y_train, np.random.randn(3, 1), 
            mini_batch_size=optimal_hp_params["batch_size"],
            epochs=10,
            scheduler=scheduler_optimal,
            cost_func=utils.simple_cost_func, 
            gradient_cost_func=utils.gradient_simple_function, 
            return_diagnostics=True)

        y_pred = X_test @ optimal_beta

        try:

            mse_eval = mean_squared_error(y_test, y_pred)
            print("MSE eval:", mse_eval)

        except:

            mse_eval = np.nan

        # Save parameters for plotting to work around the
        # matplotlib issue 

        x_plot_test = x[idx_test]
        y_plot_test = y[idx_test]

        run_table = pd.DataFrame({"Method": [method_string],
                                    "MSE": [mse_eval], 
                                    "Optimal hyper-parameter": [optimal_hp_params]})

        result_table = result_table._append(run_table)

        print(result_table)

# GD, no mini-batches

if DO_GD:

    list_of_schedulers = [ConstantScheduler(0.01),
                            MomentumScheduler(0.01, 0.9),
                            AdamScheduler(0.01, 0.9, 0.999),
                            RMS_propScheduler(0.01, 0.9)]

    for scheduler in list_of_schedulers:
        
        fixed_params = {"cost_func": utils.simple_cost_func,
                        "gradient_cost_func": utils.gradient_simple_function,
                        "max_iterations": 10000,  # TODO: Remove
                        "scheduler": scheduler}

        list_of_etas = [0.0001, 0.001, 0.01, 0.1]
        k_folds = 3

        grid_table = utils.gd_tuning(X_train, y_train,
                                        fixed_params,
                                        list_of_etas,
                                        k_folds)
        
        print(grid_table)

        #optimal_hp_params, min_mse_training = utils.optimal_hyper_params(grid_table)
        
        

#        grid_table = pd.DataFrame(columns=["eta", "MSE", "Fold"])
#
#        for eta, i in zip(list_of_etas, range(len(list_of_etas))):
#
#            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
#
#            current_fold = 1
#
#            scheduler.eta = eta
#
#            for train_index, test_index in kfold.split(X_train):
#
#                beta = utils.general_gradient_descent(X_train[train_index], y_train[train_index],
#                                                np.random.randn(3, 1),
#                                                scheduler=scheduler,
#                                                max_iterations=1000,
#                                                cost_func=utils.simple_cost_func,
#                                                gradient_cost_func=utils.gradient_simple_function,
#                                                return_diagnostics=False)
#
#                y_pred = X_train[test_index] @ beta
#
#                try:
#
#                    mse = mean_squared_error(y_train[test_index], y_pred)
#
#                except:
#
#                    mse = np.nan
#
#                run_table = pd.DataFrame({"eta": eta,
#                                            "MSE": mse,
#                                            "Fold": current_fold}, index=[0])
#
#                grid_table = grid_table._append(run_table, ignore_index=True)
#
#                current_fold += 1
#
#                scheduler.reset()
#
#        scheduler.reset()
#
#        average_mse_table = grid_table.groupby(["eta"]).mean()
#        average_mse_table.drop(columns=["Fold"], inplace=True)
#
#        # Find the minimum MSE
#
#        min_mse_index = np.argmin(average_mse_table["MSE"])
#        min_mse = np.min(average_mse_table["MSE"])
#
#        optimal_eta = average_mse_table.index[min_mse_index]
#
#        optimal_hp_params = {"eta": optimal_eta}
#
#        print("Optimal eta:", optimal_eta)
#
#        scheduler_optimal = scheduler
#        scheduler_optimal.eta = optimal_eta
#
#        beta_optimal = utils.general_gradient_descent(X_train, y_train,
#                                            np.random.randn(3, 1),
#                                            scheduler=scheduler_optimal,
#                                            max_iterations=10000,
#                                            cost_func=utils.simple_cost_func,
#                                            gradient_cost_func=utils.gradient_simple_function,
#                                            return_diagnostics=False)
#
#        y_pred = X_test @ beta_optimal
#
#        try:
#            mse_eval = mean_squared_error(y_test, y_pred)
#        except:
#            mse_eval = np.nan
#
#        method_string = "GD" + "-", str(scheduler.__class__.__name__)
#
#        run_table = pd.DataFrame({"Method": [method_string], "MSE": [mse_eval], 
#                                  "Optimal hyper-parameter": [optimal_hp_params]})
#
#        result_table = result_table._append(run_table, ignore_index=True)

# Now for the FFN

if DO_FFN:

    # Data has to be re-initialized because of the X-matrix 
    X_train, X_test, y_train, y_test, X, y, x, idx_train, idx_test = utils.standard_simple_function_dataset(dims_in_design_matrix=1, 
                                                                                            scale=True)


    num_obs = len(x)
    n = len(x)

    eta_list = [0.001, 1.0, 10]

    activation_func_list = [
        utils.sigmoid,
        #utils.RELU,
        #utils.LRELU,
        #utils.softmax
    ]

    schedule_list = [
        #ConstantScheduler(0.1),
        #MomentumScheduler(0.1, 0.9),
        #AdagradScheduler(0.1),
        #RMS_propScheduler(0.1, 0.9),
        AdamScheduler(0.1, 0.9, 0.999),
    ]

    dims_hidden_list = [[10, 10]]

    dims_hidden = dims_hidden_list[0]

    num_batches = 2
    loss_func_name = "MSE"
    epochs_max = 10

    # Define outcome functions

    outcome_func = utils.identity
    outcome_func_deriv = utils.identity_derived

    output_dim = 1
    input_dim = 1

    k_folds = 3

    for activation_func in activation_func_list:

        for scheduler in schedule_list:

            for dims_hidden in dims_hidden_list:

                grid_table = pd.DataFrame(columns=["eta", 
                                                   "MSE_test", 
                                                   "Fold"])

                for eta in eta_list:

                    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

                    current_fold = 1 # Hacky

                    for train_index, test_index in kfold.split(X_train):

                        print(activation_func, scheduler, eta)

                        activation_func_deriv = utils.derivate(activation_func)

                        scheduler.eta = eta

                        net = fnn(dim_input=input_dim, dim_output=output_dim, dims_hiddens=dims_hidden, learning_rate=eta, loss_func_name=loss_func_name,
                                  activation_func=activation_func, outcome_func=outcome_func, activation_func_deriv=activation_func_deriv,
                                  outcome_func_deriv=outcome_func_deriv,
                                  batches=num_batches,
                                  lmbd=0,
                                  scheduler=scheduler, random_state=random_state)

                        net.init_random_weights_biases()

                        loss_epochs = net.train(X_train, y_train, 
                                                scheduler=scheduler,
                                                epochs=epochs_max, 
                                                verbose=False)

                        y_pred = net.predict_feed_forward(X_test)

                        try:

                            mse = mean_squared_error(y_test, y_pred)

                        except:

                            mse = np.nan

                        run_table = pd.DataFrame({"eta": eta,
                                                    "MSE_test": mse,
                                                    "Fold": current_fold}, index=[0])

                        grid_table = grid_table._append(run_table, ignore_index=True)

                        current_fold += 1

                        scheduler.reset()

                    # Find the minimum MSE

                scheduler.reset()

                print(grid_table)

                optimal_hp_params, min_mse_value = utils.optimal_hyper_params(grid_table)

                scheduler_optimal = scheduler
                scheduler_optimal.eta = optimal_hp_params["eta"] # TODO: Generalize

                net = fnn(dim_input=input_dim, dim_output=output_dim, dims_hiddens=dims_hidden, learning_rate=optimal_hp_params["eta"], loss_func_name=loss_func_name,
                                activation_func=activation_func, outcome_func=outcome_func, activation_func_deriv=activation_func_deriv,
                                outcome_func_deriv=outcome_func_deriv,
                                batches=num_batches,
                                lmbd=0,
                                scheduler=scheduler, random_state=random_state)

                net.init_random_weights_biases()

                print("Training with optmial eta")

                loss_epochs = net.train(X_train, y_train,
                                        scheduler=scheduler_optimal,
                                        epochs_max=epochs_max, verbose=True)

                y_pred = net.predict_feed_forward(X_test)

                ax.plot(x[idx_test], y_pred, 'x', label='Fitted FFN', alpha=0.5)

                try:

                    mse_eval = mean_squared_error(y_test, y_pred)

                except:

                    mse_eval = np.nan

                method_string = "FFN" + "-", str(scheduler.__class__.__name__) + "-", str(activation_func.__name__) + "-", str(dims_hidden)

                run_table = pd.DataFrame({"Method": [method_string], "MSE": [mse],
                                            "Optimal hyper-parameter": [optimal_hp_params]})

                result_table = result_table._append(run_table, ignore_index=True)

print(result_table)

sys.exit()
