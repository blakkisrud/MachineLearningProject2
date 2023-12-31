"""
Script to test the implementation of the FFN

Need some re-writing and beatufication but it works

Doing simple logging to the console for now

"""

# import numpy as np
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
from sklearn.metrics import f1_score

import pandas as pd

import project_2_utils as utils
from project_2_utils import ConstantScheduler
from project_2_utils import MomentumScheduler
from project_2_utils import AdamScheduler
from project_2_utils import AdagradScheduler
from project_2_utils import RMS_propScheduler

from neural_net import fnn
from neural_net import grid_search
# MAIN HYPERVARIABLES
# FOR VARIABLES RELATED TO EACH DATA SET, E.G. NUMBER OF SAMPLES, SEE THE LOADING IF-TESTS FURTHER BELOW

# What data to analyse (comment out before running from terminal)
data_mode = 2
data_mode_names = {1: "simple_1d_function",
                   2: "wisconsin_classif"}  # add MNIST, Franke, terrain ?


# NETWORK PARAMETERS
dims_hidden = [8]
lr = 0.01       # Learning rate
lmbd = 0.01    # L2-regularization parameter
epochs_max = 1000   # maximum number of epochs to consider before tuning it as a HP
# num_batches = 32
num_batches = 4
dropout_retain_proba = 1.0

#loss_func_name = "MSE"
loss_func_name = "cross-entropy"      # only use when final layer outcome are in range (0, 1] ! E.g. with sigmoid, softmax activations

FIND_OPTIMAL_EPOCHS = False

random_state = 42   


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
    y = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        y.reshape(-1, 1)).reshape(-1, 1)

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
    print(
        f"outcome data balance: N_{counts_y[0][0]} = {counts_y[1][0]} ({counts_y[1][0] / num_obs * 100:.1f}%) / N_{counts_y[0][1]} = {counts_y[1][1]} ({counts_y[1][1] / num_obs * 100:.1f}%)")
    del counts_y

    feature_names = X.columns.values
    X = X.values
    X = StandardScaler().fit_transform(X)
    y = y.values.reshape(-1, 1)

    output_dim = 1
    input_dim = X.shape[1]

    # outcome func softmax ? outputs probability distributed values, which sigmoid does not, according to http://neuralnetworksanddeeplearning.com/chap3.html
    #outcome_func = utils.identity
    #outcome_func_deriv = utils.derivate(utils.identity)
    outcome_func = utils.sigmoid
    outcome_func_deriv = utils.derivate(utils.sigmoid)


idx = np.arange(len(y))

assert len(idx) == X.shape[0]

idx_train, idx_evaluation = train_test_split(idx, train_size=0.7,
                                             random_state=random_state)

X_train = X[idx_train, :]
X_evaluation = X[idx_evaluation, :]

y_train = y[idx_train]
y_evaluation = y[idx_evaluation]
# Set up parameters for the FFN
print(f'y-train shape: {np.shape(y_train)}')
activation_func_list = [
    utils.sigmoid,
    utils.RELU,
    utils.LRELU,
    utils.softmax
]

schedule_list = [
    ConstantScheduler(lr),
    MomentumScheduler(lr, 0.5),
    #AdagradScheduler(lr),
    RMS_propScheduler(lr, 0.9),
    AdamScheduler(lr, 0.9, 0.999)
]

print("\nTESTING ALL COMBINATIONS OF HIDDEN LAYER ACTIVATION FUNCTIONS", end="\t")
print([act.__name__ for act in activation_func_list])
print("WITH SCHEDULERS", end="\t")
print([type(sch) for sch in schedule_list])
print("FOR OUTCOME ACTIVATION", outcome_func.__name__,
      ", LOSS FUNCTION", loss_func_name)

error_log = ""

result_frame = pd.DataFrame(columns=["ActivationFunc",
                                     "Design",
                                     "Scheduler",
                                     "MSE",
                                     "Accuracy",
                                     "F1",
                                     "Accuracy train",
                                     "F1 train"],
                            index=np.arange(0))

max_loss = 0

for activation_func in activation_func_list:

    for scheduler in schedule_list:

        try:

            print("\n-----------------------------")
            print("Now doing activation function: ", activation_func.__name__)
            print("Now doing scheduler: ", scheduler.__class__.__name__)

            run_frame = pd.DataFrame(columns=["ActivationFunc",
                                              "Design",
                                              "Scheduler",
                                              "MSE",
                                              "Accuracy",
                                              "F1",
                                              "Accuracy train",
                                              "F1 train"],
                                     index=np.arange(1))

            activation_func = activation_func
            activation_func_deriv = utils.derivate(activation_func)

            run_frame["ActivationFunc"] = activation_func.__name__
            run_frame["Scheduler"] = scheduler.__class__.__name__
            run_frame["Design"] = str(dims_hidden)
            net = fnn(dim_input=input_dim, dim_output=output_dim, dims_hiddens=dims_hidden, learning_rate=lr, loss_func_name=loss_func_name,
                      activation_func=activation_func, outcome_func=outcome_func, activation_func_deriv=activation_func_deriv,
                      outcome_func_deriv=outcome_func_deriv,
                      batches=num_batches,
                      lmbd=lmbd,
                      scheduler=scheduler, random_state=random_state)

            nrand = 1
            name_of_act_func = activation_func.__name__
            name_of_scheduler = scheduler.__class__.__name__
            name_of_out_func = outcome_func.__name__

            linewidth = 4.0

            if FIND_OPTIMAL_EPOCHS:
                net.init_random_weights_biases(verbose=False)
                epochs_opt, loss_hptune_train, loss_hptune_val = net.find_optimal_epochs_kfold(
                    X, y, k=3, epochs_max=epochs_max, plot=False, return_loss_values=True, verbose=False)
            else:
                epochs_opt = epochs_max

            net.init_random_weights_biases(verbose=True)

            loss_epochs = net.train(X_train, y_train, epochs=epochs_opt,
                                    scheduler=scheduler, dropout_retain_proba=dropout_retain_proba,
                                    verbose=False)
            print("WEIGHTS post-training:",
                  [np.round(w.reshape(-1), 1) for w in net.weights])

            i = 1  # Because random-init still lingers

            loss_final = loss_epochs[-1]
            print(
                i, f"final loss ({loss_func_name}) = {loss_final:.2e}", end="\t")
            max_loss = loss_final if loss_final > max_loss else max_loss
            
            yhat = net.predict_feed_forward(X_evaluation)
            yhat = yhat.reshape(-1, 1)
            yhat_train = net.predict_feed_forward(X_train)
            yhat_train = yhat_train.reshape(-1, 1)

            mse = mean_squared_error(y_evaluation, yhat)
            run_frame["MSE"] = mse

            title = f"hidden dims = {net.dims_hiddens}, eta={net.learning_rate:.3e}, N_obs={num_obs}" + " act=" + \
                name_of_act_func + " " + name_of_scheduler + " out=" + \
                    name_of_out_func + " " + f" loss={loss_func_name}"
            fname = os.path.join(
                plot_folder, f"{name_of_act_func}_{name_of_scheduler}_{name_of_out_func}_{loss_func_name}.png")
            fname_hptune = os.path.join(
                plot_folder, f"{name_of_act_func}_{name_of_scheduler}_{name_of_out_func}_{loss_func_name}_hptune.png")

            # PLOTTING LOSS OVER EPOCHS WITH FINAL PREDICTIONS
            fig, ax = plt.subplots(ncols=2, figsize=(12, 8))
            ax, ax1 = ax
            # ax1.set_ylim(0, 1)

            if data_mode == 2:
                # Test evaluation
                auc = roc_auc_score(y_evaluation, yhat)
                y_hat_binary = np.zeros((yhat.shape[0], 1))
                y_hat_binary[yhat > 0.5] = 1
                acc = accuracy_score(y_evaluation, y_hat_binary)
                tp, fp, tn, fn = utils.calculate_metrics(y_evaluation, y_hat_binary)
                f1 = f1_score(y_evaluation, y_hat_binary)
                run_frame["F1"] = f1
                run_frame["Accuracy"] = acc

                # Train evaluation 
                y_hat_binary_train = np.zeros((yhat_train.shape[0], 1))
                y_hat_binary_train[yhat_train > 0.5] = 1
                acc_train = accuracy_score(y_train, y_hat_binary_train)
                f1_train = f1_score(y_train, y_hat_binary_train)
                run_frame["F1 train"] = f1_train
                run_frame["Accuracy train"] = acc_train
                
                print(f"The F1 score is: {f1}")
                print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
                print(f"mse={mse:.2e}, auc={auc:.3f}, acc={acc:.3f}")
                title += f"\nmse={mse:.2e}, auc={auc:.2f}, acc={acc:.2f}"
            else:
                print(f"mse={mse:.2e}")
                title += f"\nmse={mse:.2e}"

            # print(f"weights", net.weights, "biases", net.biases)

            ax.plot(list(range(len(loss_epochs))),
                    loss_epochs, c=f"C{i}", label=i)
            ax.set_title(f"{loss_func_name} during training")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            # ax.set_ylim(0, max_loss * 1.1)
            ax.legend()

            if data_mode == 1:
                ax1.plot(x, yhat, c=f"C{i}", label=i)
                ax1.plot(x, y, linewidth=linewidth, c="black", zorder=0)
                ax1.set_xlabel("x")

            elif data_mode == 2:
                # yhat_thresh = [1 if p > 0.5 else 0 for p in yhat]
                yhat_0 = yhat[y_evaluation == 0]
                yhat_1 = yhat[y_evaluation == 1]
                print(len(yhat_0), len(yhat_1))
                ax1.plot(y_evaluation[y_evaluation == 0], yhat_0, "o", c=f"C{i}")
                ax1.plot(y_evaluation[y_evaluation == 1], yhat_1, "o", c=f"C{i}")

            ax1.set_title(f"predictions post-training")
            ax1.legend()
            fig.suptitle(title)

            fig.savefig(fname)

            if FIND_OPTIMAL_EPOCHS:
                # PLOTTING OPTIMAL NUMBER OF EPOCHS FOUND BY HP-TUNING
                fig_tune, ax_tune = plt.subplots(
                    ncols=2, figsize=(12, 8), sharey=True)
                ax_tune, ax_tune1 = ax_tune

                epochs_tune = list(range(1, epochs_max + 1))
                ax_tune.set_title("Training loss")
                ax_tune1.set_title("Validation loss")

                for ki in range(len(loss_hptune_val)):
                    ax_tune.plot(
                        epochs_tune, loss_hptune_train[ki], c=f"C{ki}")
                    ax_tune1.plot(epochs_tune, loss_hptune_val[ki], c=f"C{ki}")
                ylims = ax_tune.get_ylim()

                ax_tune.vlines(
                    x=epochs_opt, ymin=ylims[0], ymax=ylims[1], linestyles=":", colors="black", label="optimal epoch")
                ax_tune1.vlines(
                    x=epochs_opt, ymin=ylims[0], ymax=ylims[1], linestyles=":", colors="black", label="optimal epoch")

                ax_tune.set_ylabel(f"{loss_func_name}")
                ax_tune.set_xlabel("epoch")
                ax_tune1.set_xlabel("epoch")
                ax_tune.legend()
                ax_tune1.legend()

                fig_tune.suptitle(title)

                fig_tune.savefig(fname_hptune)

            result_frame = result_frame._append(run_frame, ignore_index=True)

        except Exception as e:

            print("Exception: ", e)
            print("Continuing...")

            error_log += f"Exception: {e}\n" + f"Activation function: {activation_func.__name__}\n" + \
                f"Scheduler: {scheduler.__class__.__name__}\n"

            continue

print(result_frame)

# Perform Grid-Search beforehand to find L2-regularization parameter as well as learning rate
grid_search(X_train, y_train, 2, input_dim, output_dim, dims_hidden, loss_func_name, num_batches, epochs_max)



sys.exit()
