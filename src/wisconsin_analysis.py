import pandas as pd

import project_2_utils as utils
from neural_net import fnn

import numpy as np
from matplotlib import pyplot as plt
import sys
import seaborn as sns
import os

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils import resample

## TESTS MULTIPLE ITERATIONS OF NN ARCHITECTURE AGAINST EACH OTHER
## 1: HOW DOES THE TEST SCORE CHANGE WITH MODEL SIZE / COMPLEXITY?

RANDOM_STATE = 42

NETWORK_ARCHITECTURES = ([16], [8], [4], [2], [])
# avaliable_data_factors = [1.0, 0.5, 0.1]
# avaliable_data_factors = [1.0, 0.8, 0.5, 0.3]   # to evaluate robustness to smaller train sets, after hyperparameter tuning
avaliable_data_factors = []   # to evaluate robustness to smaller train sets, after hyperparameter tuning
print(avaliable_data_factors)

LOSS_FUNC_NAME = "cross-entropy"
SCALE_INPUT = True
OUTCOME_ACTIVATION = utils.sigmoid
OUTCOME_ACTIVATION_DERIV = utils.derivate(OUTCOME_ACTIVATION)

HIDDEN_ACTIVATION = utils.sigmoid
HIDDEN_ACTIVATION_DERIV = utils.derivate(HIDDEN_ACTIVATION)

# HIDDEN_ACTIVATION = utils.RELU
# HIDDEN_ACTIVATION = utils.LRELU

# BOOTSTRAP_REPEATS = 3
BOOTSTRAP_REPEATS = 0 # if 0: no boostrap, run only on training data
L1_RATIO = 1.0
LMBD = 0.1
# LMBD = 1.0
LR = 0.01

# SCHEDULER = utils.ConstantScheduler(LR)
# SCHEDULER = utils.MomentumScheduler(LR, 0.9)
SCHEDULER = utils.AdamScheduler(LR, 0.9, 0.99)

df_scores = pd.DataFrame()

LOAD_HP_TUNING = False # loads from saved file if true, computes grid-search if false

EPOCHS_MAX = 1000   # maximum number of epochs to evaluate in HP-grid search by 3-fold cross-validation
BATCHES_MAX = 3
hp_tune_param_dict = {
    "scheduler.eta":np.logspace(-4, 0, 5),
    "lmbd":[0, 0.1, 0.15, 0.2], "l1_ratio":[0.0, 0.5, 1.0]
}   # + dropout_proba?

# LOAD DATA
X, y = load_breast_cancer(as_frame=True, return_X_y=True)
num_obs = len(X)
counts_y = np.unique(y, return_counts=True)
print(
    f"Outcome data balance: N_{counts_y[0][0]} = {counts_y[1][0]} ({counts_y[1][0] / num_obs * 100:.1f}%) / N_{counts_y[0][1]} = {counts_y[1][1]} ({counts_y[1][1] / num_obs * 100:.1f}%)")
del counts_y

feature_names = X.columns.values
X = X.values

if SCALE_INPUT:
    X = StandardScaler().fit_transform(X)
else:
    pass

y = y.values.reshape(-1, 1)
print("Shape x / y", X.shape, y.shape)

output_dim = 1
input_dim = X.shape[1]
num_obs_tot = X.shape[0]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)
print("Xtrain ytest:", X_train.shape, y_test.shape)


i = 0
for dims_hiddens in NETWORK_ARCHITECTURES:
    print("\n\n--- NEW NETWORK ARCHETICTURE (hidden layers):", dims_hiddens, "---")
    save_path_hptune = os.path.join("results", "wisconsin", f"hptune_{dims_hiddens}.csv")

    # INITIALIZE NEW NEURAL NETWORK
    net = fnn(dim_input=input_dim, dim_output=output_dim, dims_hiddens=dims_hiddens, loss_func_name=LOSS_FUNC_NAME,
              activation_func=HIDDEN_ACTIVATION, activation_func_deriv=HIDDEN_ACTIVATION_DERIV,
              outcome_func=OUTCOME_ACTIVATION, outcome_func_deriv=OUTCOME_ACTIVATION_DERIV,
              scheduler=SCHEDULER, random_state=RANDOM_STATE, normalize_outcome=False, epochs=EPOCHS_MAX,
              batches=BATCHES_MAX, l1_ratio=L1_RATIO, lmbd=LMBD
              )

    num_total_weights = np.sum([np.prod(np.shape(w)) for w in net.weights])
    num_total_biases = np.sum([np.prod(np.shape(b)) for b in net.biases])
    num_total_wb = num_total_weights + num_total_biases
    num_neurons = np.sum([*net.dims_hiddens, net.dim_input, net.dim_output])
    print([np.shape(w) for w in net.weights])
    print("WEIGHTS / BIASES / TOTAL / NEURONS", num_total_weights, num_total_biases, num_total_wb, num_neurons)

    # HYPER-PARAMTER TUNING USING KFOLD CROSS-VALIDATION
    hyper_parameters_optimal = net.find_optimal_hps_kfold(X_train, y_train, epochs_max=EPOCHS_MAX, k=3, hp_dict=hp_tune_param_dict, verbose=False, save_path=save_path_hptune, load_precomputed=LOAD_HP_TUNING)


    # HPs: learning rate, epochs, batches, lambda, dropout
    # lr = 0.01
    # epochs = 100
    # batches = 4
    # lmb = 0.0
    # dropout_retain_proba = 1.0
    # hyper_parameters_optimal = {
    #     "scheduler.eta":lr, "epochs":epochs, "batches":batches, "lmb":lmb, "dropout_retain_proba":dropout_retain_proba
    # }

    # hyper_parameters_optimal = {}
    net.update_parameters(hyper_parameters_optimal)

    # print([(aa, net.__getattribute__(aa)) for aa in list(filter(lambda a: a[:2] != "__", dir(net)))])
    # print([(aa, net.__getattribute__("scheduler").__getattribute__(aa)) for aa in list(filter(lambda a: a[:2] != "__", dir(net.__getattribute__("scheduler"))))])


    for n_max_pc in avaliable_data_factors:


        n_max = int(X_train.shape[0] * n_max_pc)
        idx_keep = np.random.choice(list(range(X_train.shape[0])), n_max)

        # print(X_train.shape, idx_keep.shape)
        X_train_red = X_train[idx_keep, :]
        y_train_red = y_train[idx_keep, :]
        # print(X_train_red.shape, idx_keep.shape)


        print(f"\nAVALIABLE DATA: {n_max_pc*100:.0f}% -> N={n_max} (train)")

        # TODO: update name after hp-tuning
        name = (f"Hidden = {dims_hiddens}, N_train = {n_max} ({n_max_pc*100:.0f}%),"
                f"\nlmbd={LMBD:.1e}, l1_ratio={L1_RATIO:.1f}, lr={LR:.2e}")

        loss_train_vals = []
        loss_test_vals = []
        acc_test_vals = []
        rec_test_vals = []
        prec_test_vals = []
        f1_test_vals = []

        for b in range(BOOTSTRAP_REPEATS + 1):

            i += 1  # move to innermost for-loop

            if b == 0:
                print("NO BOOT\t", b)
                fig, ax = plt.subplots()
                label_train = "Train"
                label_test = "Test"
                alpha = 1.0
                ls = "-"
            else:
                print("BOOT\t", b)
                label_train, label_test = None, None
                alpha = 0.7
                ls = ":"

            if b == 0:
                pass
                X_train_red_b = X_train_red
                y_train_red_b = y_train_red
            else:
                X_train_red_b, y_train_red_b = resample(X_train_red, y_train_red, replace=True, random_state=RANDOM_STATE + b)

            # RESET NETWORK WEIGHTS AND BIASES
            net.init_random_weights_biases(verbose=True)

            # TRAINING using optimal HPs
            loss_train, loss_test = net.train(X_train_red, y_train_red, X_test, y_test,
                                              verbose=True)

            num_nonzero_weights = [np.count_nonzero(w) for w in net.weights]
            num_nonzero_biases = [np.count_nonzero(b) for b in net.biases]
            print("\tNonzero weights / biases per layer:", num_nonzero_weights, num_nonzero_biases, end="\n")
            num_nonzero_wb = np.sum([num_nonzero_biases, num_nonzero_weights])
            print(f"\ttotal {num_nonzero_wb} of {num_total_wb} nonzero ({num_nonzero_wb / num_total_wb * 100:.0f}%)")
            # print("\t", net.weights)

            epochs_vals = np.linspace(0, net.epochs, net.epochs, dtype=int)

            ax.plot(epochs_vals, loss_train, label=label_train, c="C0", alpha=alpha, ls=ls)
            ax.plot(epochs_vals, loss_test, label=label_test, c="C1", alpha=alpha, ls=ls)

            ax.legend()
            ax.set_title(name)
            ax.set_ylim(0, None)

            # TESTING AND MODEL EVALUTATION -> accuracy, precision, recall, F1, ...
            yhat_test = net.predict_feed_forward(X_test)
            yhat_test_bin = np.zeros(shape=yhat_test.shape)
            yhat_test_bin[yhat_test > 0.5] = 1

            acc = accuracy_score(y_test, yhat_test_bin)
            rec = recall_score(y_test, yhat_test_bin)
            prec = precision_score(y_test, yhat_test_bin)
            f1 = f1_score(y_test, yhat_test_bin)

            df_scores.loc[i, "Boot"] = b
            df_scores.loc[i, "Hidden dims"] = str(dims_hiddens)
            df_scores.loc[i, "Num neurons"] = num_neurons
            df_scores.loc[i, "%N_train"] = n_max_pc

            df_scores.loc[i, ["Acc", "Rec", "Prec", "F1"]] = [acc, rec, prec, f1]
            df_scores.loc[i, ["loss train", "loss test"]] = loss_train[-1], loss_test[-1]


print(df_scores.shape)
print(df_scores.round(3))
df_scores_avg = df_scores.groupby(["Hidden dims", "%N_train"]).mean().reset_index()

print(df_scores_avg.sort_values(by=["Num neurons", "%N_train"]).round(3))

# PLOT MODEL PERFORMANCES WITH DECREASING AVALIABLE TRAINING DATA
fig, ax = plt.subplots()
# sns.lineplot(df_scores_avg, x="%N_train", y="loss test", hue="Hidden dims", ax=ax)
sns.lineplot(df_scores_avg, x="%N_train", y="loss test", hue="Hidden dims", ax=ax)

plt.show()
