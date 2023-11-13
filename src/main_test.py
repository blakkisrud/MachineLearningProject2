"""
Script to test the implementation of the FFN

Need some re-writing and beatufication but it works

Doing simple logging to the console for now

"""

# import numpy as np
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
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score

import project_2_utils as utils
from project_2_utils import ConstantScheduler
from project_2_utils import MomentumScheduler
from project_2_utils import AdamScheduler
from project_2_utils import AdagradScheduler
from project_2_utils import RMS_propScheduler

from neural_net import fnn


# MAIN HYPERVARIABLES
# FOR VARIABLES RELATED TO EACH DATA SET, E.G. NUMBER OF SAMPLES, SEE THE LOADING IF-TESTS FURTHER BELOW

data_mode = 2           # What data to analyse (comment out before running from terminal)
data_mode_names = {1:"simple_1d_function", 2:"wisconsin_classif"}  # add MNIST, Franke, terrain ?


# NETWORK PARAMETERS

# dims_hidden = [8, 8, 8]
# dims_hidden = [1]
dims_hidden = [4]
lr = 0.1
epochs_max = 1000   # maximum number of epochs to consider before tuning it as a HP
# num_batches = 32
num_batches = 4


# loss_func_name = "MSE"
loss_func_name = "cross-entropy"      # only use when final layer outcome are in range (0, 1] ! E.g. with sigmoid, softmax activations

FIND_OPTIMAL_EPOCHS = False

random_state = 42   # does nothing, yet


plot_dir = "figs/"      # Where to plot


# Make a directory for the test-suite
plot_folder = os.path.join(plot_dir, f"{data_mode_names[data_mode]}")


if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)



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

    num_obs = 1000
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

    output_dim = 1
    input_dim = 1


# LOAD WISCONSIN BREAST CANCER DATASET FOR CLASSIFICATION
elif data_mode == 2:
    print(f"LOADING {data_mode_names[data_mode].upper()}")

    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(as_frame=True, return_X_y=True)
    print("Shape x / y", X.shape, y.shape)
    num_obs = len(X)

    counts_y = np.unique(y, return_counts=True)
    print(f"outcome data balance: N_{counts_y[0][0]} = {counts_y[1][0]} ({counts_y[1][0] / num_obs * 100:.1f}%) / N_{counts_y[0][1]} = {counts_y[1][1]} ({counts_y[1][1] / num_obs * 100:.1f}%)")
    del counts_y


    feature_names = X.columns.values
    X = X.values
    X = StandardScaler().fit_transform(X)
    y = y.values.reshape(-1, 1)

    output_dim = 1
    input_dim = X.shape[1]

    # outcome func softmax ? outputs probability distributed values, which sigmoid does not, according to http://neuralnetworksanddeeplearning.com/chap3.html
    # outcome_func = utils.softmax
    # outcome_func_deriv = utils.derivate(utils.softmax)
    outcome_func = utils.sigmoid
    outcome_func_deriv = utils.derivate(utils.sigmoid)


# TODO: resevere test set NOT for training

# Set up parameters for the FFN

activation_func_list = [
                        utils.sigmoid,
                        utils.RELU,
                        utils.LRELU,
                        utils.softmax
                        ]

schedule_list = [
                # ConstantScheduler(0.1)
                ConstantScheduler(0.1),
                MomentumScheduler(0.1, 0.9),
                # AdagradScheduler(0.1),
                # RMS_propScheduler(0.1, 0.9),
                # AdamScheduler(0.1, 0.9, 0.999),
                ]

print("\nTESTING ALL COMBINATIONS OF HIDDEN LAYER ACTIVATION FUNCTIONS", end="\t")
print([act.__name__ for act in activation_func_list])
print("WITH SCHEDULERS", end="\t")
print([type(sch) for sch in schedule_list])

error_log = ""


max_loss = 0
for activation_func in activation_func_list:

    for scheduler in schedule_list:

        try:

            print("\n-----------------------------")
            print("Now doing activation function: ", activation_func.__name__)
            print("Now doing scheduler: ", scheduler.__class__.__name__)

            activation_func = activation_func
            activation_func_deriv = utils.derivate(activation_func)


            net = fnn(dim_input=input_dim, dim_output=output_dim, dims_hiddens=dims_hidden, learning_rate=lr, loss_func_name=loss_func_name,
                      activation_func=activation_func, outcome_func=outcome_func, activation_func_deriv=activation_func_deriv,
                      outcome_func_deriv=outcome_func_deriv,
                      batches=num_batches,
                      lmbd = 0.1,
                      scheduler=scheduler, random_state=random_state)

            nrand = 1
            name_of_act_func = activation_func.__name__
            name_of_scheduler = scheduler.__class__.__name__
            name_of_out_func = outcome_func.__name__


            linewidth = 4.0

            if FIND_OPTIMAL_EPOCHS:
                net.init_random_weights_biases(verbose=False)
                epochs_opt, loss_hptune_train, loss_hptune_val = net.find_optimal_epochs_kfold(X, y, k=3, epochs_max=epochs_max, plot=False, return_loss_values=True, verbose=False)
            else:
                epochs_opt = epochs_max

            net.init_random_weights_biases(verbose=True)


            loss_epochs = net.train(X, y, epochs=epochs_opt,
                                    scheduler=scheduler,
                                    verbose=False)
            #print("WEIGHTS post-training:", [np.round(w.reshape(-1), 1) for w in net.weights])

            i = 1 # Because random-init still lingers

            loss_final = loss_epochs[-1]
            print(i, f"final loss ({loss_func_name}) = {loss_final:.2e}", end="\t")
            max_loss = loss_final if loss_final > max_loss else max_loss

            yhat = net.predict_feed_forward(X)
            mse = mean_squared_error(y, yhat)

            title = f"hidden dims = {net.dims_hiddens}, eta={net.learning_rate:.3e}, N_obs={num_obs}" + " act=" + name_of_act_func + " " + name_of_scheduler + " out=" + name_of_out_func + " " + f" loss={loss_func_name}"
            fname = os.path.join(plot_folder, f"{name_of_act_func}_{name_of_scheduler}_{name_of_out_func}_{loss_func_name}.png")
            fname_hptune = os.path.join(plot_folder, f"{name_of_act_func}_{name_of_scheduler}_{name_of_out_func}_{loss_func_name}_hptune.png")


            # PLOTTING LOSS OVER EPOCHS WITH FINAL PREDICTIONS
            # TODO: ADD TEST SET PREDICTIONS / LOSS
            fig, ax = plt.subplots(ncols=2, figsize=(12, 8))
            ax, ax1 = ax
            ax1.set_ylim(0, 1)


            if data_mode == 2:
                # acc = accuracy_score(y, yhat)
                auc = roc_auc_score(y, yhat)
                y_hat_binary = np.zeros((yhat.shape[0], 1))
                y_hat_binary[yhat > 0.5] = 1
                acc = accuracy_score(y, y_hat_binary)
                print(f"mse={mse:.2e}, auc={auc:.3f}, acc={acc:.3f}")
                title += f"\nmse={mse:.2e}, auc={auc:.2f}, acc={acc:.2f}"
            else:
                print(f"mse={mse:.2e}")
                title += f"\nmse={mse:.2e}"



            # print(f"weights", net.weights, "biases", net.biases)

            ax.plot(list(range(len(loss_epochs))), loss_epochs, c=f"C{i}", label=i)
            ax.set_title(f"{loss_func_name} during training")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.set_ylim(0, max_loss * 1.1)
            ax.legend()

            if data_mode == 1:
                ax1.plot(x, yhat, c=f"C{i}", label=i)
                ax1.plot(x, y, linewidth=linewidth, c="black", zorder=0)
                ax1.set_xlabel("x")


            elif data_mode == 2:
                # yhat_thresh = [1 if p > 0.5 else 0 for p in yhat]
                yhat_0 = yhat[y == 0]
                yhat_1 = yhat[y == 1]
                print(len(yhat_0), len(yhat_1))
                ax1.plot(y[y == 0], yhat_0, "o", c=f"C{i}")
                ax1.plot(y[y == 1], yhat_1, "o", c=f"C{i}")

            ax1.set_title(f"predictions post-training")
            ax1.legend()
            fig.suptitle(title)

            fig.savefig(fname)

            if FIND_OPTIMAL_EPOCHS:
                # PLOTTING OPTIMAL NUMBER OF EPOCHS FOUND BY HP-TUNING
                fig_tune, ax_tune = plt.subplots(ncols=2, figsize=(12, 8), sharey=True)
                ax_tune, ax_tune1 = ax_tune

                epochs_tune = list(range(1, epochs_max + 1))
                ax_tune.set_title("Training loss")
                ax_tune1.set_title("Validation loss")

                for ki in range(len(loss_hptune_val)):
                    ax_tune.plot(epochs_tune, loss_hptune_train[ki], c=f"C{ki}")
                    ax_tune1.plot(epochs_tune, loss_hptune_val[ki], c=f"C{ki}")
                ylims = ax_tune.get_ylim()

                ax_tune.vlines(x=epochs_opt, ymin=ylims[0], ymax=ylims[1], linestyles=":", colors="black", label="optimal epoch")
                ax_tune1.vlines(x=epochs_opt, ymin=ylims[0], ymax=ylims[1], linestyles=":", colors="black", label="optimal epoch")

                ax_tune.set_ylabel(f"{loss_func_name}")
                ax_tune.set_xlabel("epoch")
                ax_tune1.set_xlabel("epoch")
                ax_tune.legend()
                ax_tune1.legend()

                fig_tune.suptitle(title)

                fig_tune.savefig(fname_hptune)


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

X = np.c_[np.ones((num_obs, 1)), x]

# Hessian matrix

H = (2.0 / num_obs) * X.T.dot(X)

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
    gradient = (2.0 / num_obs) * X.T.dot(X.dot(beta) - y)
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


