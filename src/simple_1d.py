"""
Script to test the implementation of the learning methods.

When run as a script, this script will generate a simple data set and
test the implementation of the learning methods on said set.

The end result is a table with the MSE for each method that is saved
to a file.

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

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

import pandas as pd

import project_2_utils as utils
from project_2_utils import ConstantScheduler
from project_2_utils import MomentumScheduler
from project_2_utils import AdamScheduler
from project_2_utils import AdagradScheduler
from project_2_utils import RMS_propScheduler

from neural_net import fnn

path_to_output = "one_d_output"

if not os.path.exists(path_to_output):
    os.makedirs(path_to_output)

random_state = 42

# Make the data set and split into training and test
# The test-set is held off until the end of each learning method

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

# Decide which methods to run

DO_OLS = True
DO_SGD = True
DO_GD = True
DO_FFN = True
DO_SKLEARN = False

if DO_OLS:

# OLS - no hyper-parameters

    beta_ols = utils.OLS(X_train, y_train)

    y_pred = X_test @ beta_ols

    mse = mean_squared_error(y_test, y_pred)

    run_table = pd.DataFrame({"Method": ["OLS"], "MSE": [mse], "Optimal hyper-parameter": ["None"]})

    result_table = result_table._append(run_table)

    ax.plot(x[idx_test], y[idx_test], 'o', label='Data points', alpha=0.5)
    ax.plot(x[idx_test], y_pred, 'o', label='OLS', alpha=0.8)

    plt.legend()

    plt.savefig(path_to_output + "/ols.png")

    # Ridge - hyper-parameter is lambda
    
    lambdas = np.logspace(-5, 5, 11)

    tuning_table = utils.ridge_tuning(X_train, y_train, lambdas, k_folds=3)

    print(tuning_table)

    optimal_hp_params, min_mse_value = utils.optimal_hyper_params(tuning_table)

    optimal_lambda = optimal_hp_params["lambda"]

    mse_by_lambda = np.zeros(len(lambdas))

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

        optimal_beta, mse_training, mse_training_epoch, change_vector = utils.general_stochastic_gradient_descent(
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
                            RMS_propScheduler(0.01, 0.9)
                             ]

    for scheduler in list_of_schedulers:
        
        fixed_params = {"cost_func": utils.simple_cost_func,
                        "gradient_cost_func": utils.gradient_simple_function,
                        "max_iterations": 1000,  # TODO: Adjust
                        "scheduler": scheduler}

        list_of_etas = [0.00001, 0.0001, 0.001, 0.01, 0.1]
        k_folds = 3

        grid_table = utils.gd_tuning(X_train, y_train,
                                        fixed_params,
                                        list_of_etas,
                                        k_folds)
        
        print(grid_table)

        optimal_hp_params, min_mse_training = utils.optimal_hyper_params(grid_table)

        scheduler.reset()
        scheduler_optimal = scheduler
        scheduler_optimal.eta = optimal_hp_params["eta"]

        beta_optimal = utils.general_gradient_descent(X_train, y_train,
                                            np.random.randn(3, 1),
                                            scheduler=scheduler_optimal,
                                            max_iterations=100000,
                                            cost_func=utils.simple_cost_func,
                                            gradient_cost_func=utils.gradient_simple_function,
                                            return_diagnostics=True)
        
        if type(beta_optimal) == dict:
            beta_optimal = beta_optimal["beta"]

        y_pred = X_test @ beta_optimal

        try:
            mse_eval = mean_squared_error(y_test, y_pred)
        except:
            mse_eval = np.nan

        method_string = "GD" + "-", str(scheduler.__class__.__name__)

        run_table = pd.DataFrame({"Method": [method_string], "MSE": [mse_eval], 
                                  "Optimal hyper-parameter": [optimal_hp_params]})

        result_table = result_table._append(run_table, ignore_index=True)

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
        utils.RELU,
        utils.LRELU,
        #utils.softmax
    ]

    schedule_list = [
        ConstantScheduler(0.1),
        MomentumScheduler(0.1, 0.9),
        #AdagradScheduler(0.1),
        RMS_propScheduler(0.1, 0.9),
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

if DO_SKLEARN:

    print("Running sanity check with sklearn for the GD and momentum")

    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 1000)

    print(X)
    y = utils.simple_func(X, 1, 5, 3, noise_sigma=0.0)

    # Polynomial features
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X.reshape(-1, 1))

    # Implement gradient descent with momentum
    learning_rate = 0.001
    beta = np.zeros(X_poly.shape[1])  # Initialize coefficients
    gamma = 0.9  # Momentum parameter
    epochs = 1000
    prev_grad = np.zeros(X_poly.shape[1])  # Initialize momentum

    for epoch in range(epochs):
        # Calculate predictions and error
        y_pred = np.dot(X_poly, beta)
        error = y - y_pred

        # Calculate gradients
        gradient = -2 * np.dot(X_poly.T, error) / len(y)

        # Update momentum
        grad_with_momentum = gamma * prev_grad + learning_rate * gradient

        # Update parameters
        beta -= grad_with_momentum
        prev_grad = grad_with_momentum

        # Calculate and print MSE
        mse = mean_squared_error(y, y_pred)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: MSE = {mse}")

    print(f"Final Coefficients: {beta}")

print("----------------------------------------")
print("Final result table")
print(result_table)

result_table.to_excel(path_to_output + "/result_table.xlsx")

print("End of run")

