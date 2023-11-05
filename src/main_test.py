"""
Script to test the implementation of the FFN

Need some re-writing and beatufication but it works

Doing simple logging to the console for now

"""

import numpy as np
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
from random import random, seed
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import project_2_utils as utils
from project_2_utils import ConstantScheduler
from project_2_utils import MomentumScheduler
from project_2_utils import AdamScheduler
from project_2_utils import AdagradScheduler
from project_2_utils import RMS_propScheduler

from neural_net import fnn


# MAIN HYPERVARIABLES
# FOR VARIABLES RELATED TO EACH DATA SET, E.G. NUMBER OF SAMPLES, SEE THE LOADING IF-TESTS FURTHER BELOW

data_mode = 1           # What data to analyse (comment out before running from terminal)
data_mode_names = {1:"simple_1d_function", 2:"wisconsin_classif"}  # add MNIST, Franke, terrain ?


loss_func_name = "MSE"
# loss_func_name = "cross-entropy"      # does not work, yet (see nn class for start of implementation)


random_state = 42   # does nothing, yet


plot_dir = "figs/"      # Where to plot


# Make a directory for the test-suite
test_plot = os.path.join(plot_dir, f"test_suite_{data_mode_names[data_mode]}")


if not os.path.exists(test_plot):
    os.makedirs(test_plot)



while data_mode not in data_mode_names.keys():
    data_mode = input(f"SELECT INPUT FROM {data_mode_names}")
    try:
        data_mode = int(data_mode)
    except Exception as e:
        print(*e.args)

    if not data_mode in data_mode_names.keys():
        print("Invalid input, please try again...")


if data_mode == 1:
    print(f"LOADING {data_mode_names[data_mode].upper()}")
    # Set up data points

    n = 1000
    x = np.arange(0, 10, 0.01)

    X = utils.one_d_design_matrix(x, 1)
    X = X[:, 1]
    X = X.reshape(-1, 1)

    X = StandardScaler().fit_transform(X)

    y = utils.simple_func(x, 1, 5, 3, noise_sigma=1.0)
    y = MinMaxScaler(feature_range=(0, 1)).fit_transform(y.reshape(-1, 1)).reshape(-1, 1)

    print("SHAPE x / y:", X.shape, y.shape)

    outcome_func = utils.identity
    outcome_func_deriv = utils.identity_derived


# LOAD WISCONSIN BREAST CANCER DATASET FOR CLASSIFICATION
elif data_mode == 2:
    print(f"LOADING {data_mode_names[data_mode].upper()}")

    # outcome func softmax ? outputs probability distributed values, which sigmoid does not, according to http://neuralnetworksanddeeplearning.com/chap3.html
    outcome_func = utils.softmax
    outcome_func_deriv = utils.derivate(utils.softmax)

    sys.exit()



# Set up parameters for the FFN


activation_func_list = [
                        utils.sigmoid, 
                        utils.RELU, 
                        utils.LRELU, 
                        utils.softmax,
                        ]

schedule_list = [
                ConstantScheduler(0.1),
                MomentumScheduler(0.1, 0.9),
                AdagradScheduler(0.1),
                RMS_propScheduler(0.1, 0.9),
                AdamScheduler(0.1, 0.9, 0.999),
                ]

print("TESTING ALL COMBINATIONS OF HIDDEN LAYER ACTIVATION FUNCTIONS", end="\t")
print([act.__name__ for act in activation_func_list])
print("WITH SCHEDULERS", end="\t")
print([type(sch) for sch in schedule_list])

error_log = ""



for activation_func in activation_func_list:

    for scheduler in schedule_list:

        try:

            print("\n-----------------------------")
            print("Now doing activation function: ", activation_func.__name__)
            print("Now doing scheduler: ", scheduler.__class__.__name__)

            activation_func = activation_func
            #activation_func_deriv = utils.sigmoid_derivated
            activation_func_deriv = utils.derivate(activation_func)


            # dims_hidden = [8, 8, 8]
            dims_hidden = [1]
            # dims_hidden = []

            lr = 0.1
            epochs = 100

            output_dim = 1  
            input_dim = 1

            net = fnn(dim_input=input_dim, dim_output=output_dim, dims_hiddens=dims_hidden, learning_rate=lr, loss_func_name=loss_func_name,
                      activation_func=activation_func, outcome_func=outcome_func, activation_func_deriv=activation_func_deriv,
                      outcome_func_deriv=outcome_func_deriv,
                      batches=32,
                      scheduler=scheduler)

            fig, ax = plt.subplots(ncols=2, figsize=(12, 8))
            ax, ax1 = ax
            ax1.set_ylim(0, 1)
            # fig1, ax1 = plt.subplots()

            linewidth = 4.0
            max_loss = 0

            net.init_random_weights_biases()

            loss_epochs = net.train(X, y, epochs=epochs, 
                                scheduler=scheduler,
                                plot=False, figax=(fig, ax), showplot=False, plot_title=f"MSE lr = {net.learning_rate}", verbose=False)

            i = 1 # Because random-init still lingers

            loss_final = loss_epochs[-1]
            print(i, f"final loss ({loss_func_name}) = {loss_final:.2e}", end="\t")
            max_loss = loss_final if loss_final > max_loss else max_loss

            yhat = net.predict_feed_forward(X)
            mse = mean_squared_error(y, yhat)
            mse2 = mean_squared_error(y, net.activations[-1])

            print(f"mse={mse:.2e}, {mse2:.2e}")
            print(f"weights", net.weights, "biases", net.biases)

            nrand = 1
            name_of_act_func = activation_func.__name__
            name_of_scheduler = scheduler.__class__.__name__
            title = f"hidden dims = {net.dims_hiddens}, eta={net.learning_rate:.3e}, N_obs={n}" + " " + name_of_act_func + " " + name_of_scheduler

            ax.plot(list(range(len(loss_epochs))), loss_epochs, c=f"C{i}", label=i)
            ax1.plot(x, yhat, c=f"C{i}", label=i)
            ax1.plot(x, y, linewidth=linewidth, c="black", zorder=0)
            ax.set_title(f"MSE during training")
            ax.set_xlabel("Epochs")

            ax1.set_title(f"predictions post-training")
            ax1.set_xlabel("x")
            ax.set_ylim(0, max_loss*1.1)

            ax.legend()
            ax1.legend()

            fname = os.path.join(test_plot, f"test_suite_{name_of_act_func}_{name_of_scheduler}_loss={loss_func_name}.png")

            fig.suptitle(title)

            plt.savefig(fname)

        except Exception as e:

            print("Exception: ", e)
            print("Continuing...")

            error_log += f"Exception: {e}\n" + f"Activation function: {activation_func.__name__}\n" + f"Scheduler: {scheduler.__class__.__name__}\n"

            continue

if error_log != "":
    print(error_log)
else:
    print("No errors found")

print("Have a nice day!")

sys.exit()

# Set up design matrix

X = np.c_[np.ones((n,1)), x]

# Hessian matrix

H = (2.0/n)*X.T.dot(X)

# Get the eigenvalues of the hessian

EigValues, EigVectors = np.linalg.eig(H)

print("Eigenvalues of Hessian matrix: ", EigValues)

# Having a OLS-solution for comparison

beta_linreg = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Set the learning rate, eta, and the number of iterations

eta = 0.001
MaxIterations = 100000
beta = np.random.randn(2,1)
epsilon = 1.0e-8

for iter in range(MaxIterations):
    gradient = (2.0/n)*X.T.dot(X.dot(beta)-y)
    beta -= eta*gradient
    if (np.linalg.norm(gradient) < epsilon):
        break

print("Number of iterations: ", iter)

print("Final beta values: ", beta)
print("Final gradient norm: ", np.linalg.norm(gradient))
print("Linear regression values: ", beta_linreg)

# Plot the results

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, 'o', label='Data points')
ax.plot(x, X.dot(beta), label='Gradient descent')
ax.plot(x, X.dot(beta_linreg), label='Linear regression')
plt.legend()
plt.savefig(plot_dir + "gradient_descent.png")


