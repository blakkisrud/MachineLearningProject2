"""
Classes to run a neural network on the cancer
data set and score the accuracy of the model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
from sklearn.metrics import mean_squared_error

from neural_net import fnn
import project_2_utils as utils

from project_2_utils import ConstantScheduler
from project_2_utils import MomentumScheduler
from project_2_utils import AdamScheduler
from project_2_utils import AdagradScheduler
from project_2_utils import RMS_propScheduler

from project_2_utils import logistic_regression
from project_2_utils import random_forest

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.datasets import load_breast_cancer

import sys

#===========================================
#
# Pre-process the data
#
#===========================================

data = load_breast_cancer()

print((data["data"].shape))
print((data["target"]))

feature_names = data["feature_names"]
data_matrix = data["data"]
target_vector = data["target"]

# Construact a pandas frame

df = pd.DataFrame(data_matrix, columns=feature_names)

shuffled_df = df.sample(frac=1, random_state=42)  # 'frac=1' shuffles all rows, 'random_state' for reproducibility

# Preprocessing

X =  shuffled_df[feature_names].values
x = X

print(X.shape)

y = target_vector

#y = MinMaxScaler(feature_range=(0, 1)).fit_transform(y.reshape(-1, 1)).reshape(-1, 1)

X = StandardScaler().fit_transform(X)
y = MinMaxScaler(feature_range=(0, 1)).fit_transform(y.reshape(-1, 1)).reshape(-1, 1)

idx = np.arange(len(y))

assert len(idx) == X.shape[0]

idx_train, idx_evaluation = train_test_split(idx, train_size=0.7, 
                                             random_state=42)

X_train = X[idx_train,:]
X_evaluation = X[idx_evaluation,:]

y_train = y[idx_train]
y_evaluation = y[idx_evaluation]

num_obs = len(y)
print("SHAPE x / y:", X.shape, y.shape)

# For all runs

output_dim = 1
input_dim = len(feature_names)

lr = 0.1
epochs = 500
epochs_max = 500

# Set up the list of runs here

list_of_act_funcs = [utils.RELU, 
                     utils.sigmoid, 
                     utils.LRELU]

list_of_schedulers = [AdamScheduler(lr, 0.9, 0.999), 
                      ConstantScheduler(lr), 
                      MomentumScheduler(lr, 0.9), 
                      RMS_propScheduler(lr, 0.9)]

list_of_network_design = [[8,8]]

list_of_runs = []

for act in list_of_act_funcs:
    for sch in list_of_schedulers:
        for network_design in list_of_network_design:

            network_params = {}
            network_params["Act_func"] = act
            network_params["Deriv_act_func"] = utils.derivate(act)
            network_params["Scheduler"] = sch
            network_params["Design"] = network_design

            list_of_runs.append(network_params)

result_frame = pd.DataFrame(columns=["ActivationFunc",
                                     "Design",
                                     "Scheduler",
                                     "MSE"],
                                     index=np.arange(0))

for run in list_of_runs:

    run_frame = pd.DataFrame(columns=["ActivationFunc",
                                         "Design",
                                         "Scheduler",
                                         "MSE"],
                                         index = np.arange(1))

    activation_func = run["Act_func"]
    activation_func_deriv = run["Deriv_act_func"]

    outcome_func = utils.identity
    outcome_func_deriv = utils.identity_derived

    dims_hidden = run["Design"]
    scheduler = run["Scheduler"]

    net = fnn(dim_input=input_dim, dim_output=output_dim, dims_hiddens=dims_hidden, learning_rate=lr,
              activation_func=activation_func, outcome_func=outcome_func, activation_func_deriv=activation_func_deriv, 
              outcome_func_deriv=outcome_func_deriv,
              batches=5,
              scheduler=scheduler)
    
    # Try to train the network

    try:

        loss_epochs = net.train(X_train, y_train, epochs=epochs,
                scheduler=scheduler,
                verbose=False)
        
    except:
            
            print("Run failed")
            continue
    
    # Now that the net is traned, test it on the evaluation

    try:

        mse_run, accuracy_run = net.evaluate(X_evaluation, y_evaluation)
    
    except:
             
        print("Evaluation failed")
        mse = np.nan
        accuracy = np.nan
        continue

    run_frame["MSE"] = mse_run
    run_frame["Accuracy"] = accuracy_run
    run_frame["ActivationFunc"] = activation_func.__name__
    run_frame["Scheduler"] = str(scheduler.__class__.__name__)
    run_frame["Design"] = str(dims_hidden)

    result_frame = result_frame._append(run_frame, ignore_index = True)

print("Printing final output")
print("-----------------------------")

# Do the logistic regression

log_mse, log_accuracy = logistic_regression(X_train, 
                          y_train.ravel(),
                          X_evaluation,
                          y_evaluation.ravel())

run_frame = pd.DataFrame(columns=["ActivationFunc",
                                            "Design",
                                            "Scheduler",
                                            "MSE",
                                            "Accuracy"],
                                            index = np.arange(1))

run_frame["MSE"] = log_mse
run_frame["Accuracy"] = log_accuracy
run_frame["ActivationFunc"] = ""
run_frame["Scheduler"] = ""
run_frame["Design"] = "Logistic"

result_frame = result_frame._append(run_frame, ignore_index = True)

# Test random forest

rf_mse, rf_accuracy = random_forest(X_train,
                                    y_train.ravel(),
                                    X_evaluation,
                                    y_evaluation.ravel())

run_frame = pd.DataFrame(columns=["ActivationFunc",
                                            "Design",
                                            "Scheduler",
                                            "MSE",
                                            "Accuracy"],
                                            index = np.arange(1))

run_frame["MSE"] = rf_mse
run_frame["Accuracy"] = rf_accuracy
run_frame["ActivationFunc"] = ""
run_frame["Scheduler"] = ""
run_frame["Design"] = "Random Forest"

result_frame = result_frame._append(run_frame, ignore_index = True)

print(result_frame)

sys.exit()

activation_func = utils.RELU
activation_func_deriv = utils.derivate(activation_func)

outcome_func = utils.identity
outcome_func_deriv = utils.identity_derived

dims_hidden = [4,4]
scheduler = AdamScheduler(0.1, 0.9, 0.999)


output_dim = 1
input_dim = len(feature_names)

net = fnn(dim_input=input_dim, dim_output=output_dim, dims_hiddens=dims_hidden, learning_rate=lr,
          activation_func=activation_func, outcome_func=outcome_func, activation_func_deriv=activation_func_deriv, 
          outcome_func_deriv=outcome_func_deriv,
          batches=5,
          scheduler=AdamScheduler(lr, 0.9, 0.999))

net.init_random_weights_biases()
#epochs_opt, loss_hptune_train, loss_hptune_val = net.find_optimal_epochs_kfold(X, y, k=3, epochs_max=epochs_max, plot=False, return_loss_values=True)

loss_epochs = net.train(X_train, y_train, epochs=epochs,
        scheduler=scheduler,
        verbose=False)

print("\n")
print("Done!")







