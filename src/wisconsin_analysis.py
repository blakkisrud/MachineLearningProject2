import project_2_utils as utils
from neural_net import fnn

import numpy as np


from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


## TESTS MULTIPLE ITERATIONS OF NN ARCHITECTURE AGAINST EACH OTHER
## 1: HOW DOES THE TEST SCORE CHANGE WITH MODEL SIZE / COMPLEXITY?

RANDOM_STATE = 42

# NETWORK_ARCHITECTURES = ([], [4], [8], [16], [4, 2], [8, 4], [16, 4], [16, 8, 4], [8, 4, 2])
NETWORK_ARCHITECTURES = ([8, 4, 2],)


EPOCHS_MAX = 1000   # maximum number of epochs to evaluate in HP-grid search by 3-fold cross-validation

LOSS_FUNC_NAME = "cross-entropy"
SCALE_INPUT = True
OUTCOME_ACTIVATION = utils.sigmoid
OUTCOME_ACTIVATION_DERIV = utils.sigmoid_derivated

HIDDEN_ACTIVATION = utils.sigmoid
# HIDDEN_ACTIVATION = utils.RELU
# HIDDEN_ACTIVATION = utils.LRELU

# SCHEDULER = utils.ConstantScheduler
SCHEDULER = utils.MomentumScheduler(0.1, 0.9)


HIDDEN_ACTIVATION_DERIV = utils.derivate(HIDDEN_ACTIVATION)


# LOAD DATA
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(as_frame=True, return_X_y=True)
print("Shape x / y", X.shape, y.shape)
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

output_dim = 1
input_dim = X.shape[1]


for dims_hiddens in NETWORK_ARCHITECTURES:
    print("\n\n--- NEW NETWORK ARCHETICTURE (hidden layers):", dims_hiddens, "---")


    # SET UP NEURAL NETWORK
    net = fnn(dim_input=input_dim, dim_output=output_dim, dims_hiddens=dims_hiddens, loss_func_name=LOSS_FUNC_NAME,
              activation_func=HIDDEN_ACTIVATION, activation_func_deriv=HIDDEN_ACTIVATION_DERIV,
              outcome_func=OUTCOME_ACTIVATION, outcome_func_deriv=OUTCOME_ACTIVATION_DERIV,
              scheduler=SCHEDULER, random_state=RANDOM_STATE, normalize_outcome=True
              )

    net.init_random_weights_biases(verbose=True)


    # HYPER-PARAMTER TUNING USING KFOLD CROSS-VALIDATION
    # HPs: learning rate, epochs, batches, lambda, dropout
    lr = 0.01
    epochs = 100
    batches = 4
    lmb = 0.0
    dropout = 1.0
    hyper_parameters_optimal = {
        "scheduler.eta":0.001, "epochs":100, "batches":3, "lmb":0.0, "dropout":1.0
    }

    hp_keys = list(hyper_parameters_optimal.keys())
    print(list(filter(lambda k: "." not in k, hp_keys)))

    print(list(filter(lambda a: type(net.__getattribute__(a)) == int and a[:2] != "__", dir(net))))
    print([(aa, net.__getattribute__("scheduler").__getattribute__(aa)) for aa in list(filter(lambda a: a[:2] != "__", dir(net.__getattribute__("scheduler"))))])
    net.update_parameters(hyper_parameters_optimal)
    # print(list(filter(lambda a: type(net.__getattribute__(a)) == int and a[:2] != "__", dir(net))))
    print([(aa, net.__getattribute__("scheduler").__getattribute__(aa)) for aa in list(filter(lambda a: a[:2] != "__", dir(net.__getattribute__("scheduler"))))])
    # net.__setattr__("epochs", 100)


    # TRAINING using optimal HPs

    # print(net.__getattribute__("epochs"))
    # net.train(X, y)
    # net.train(X, y, epochs=5)

    # print(dir(net.__getattribute__("scheduler")))
    # TESTING AND MODEL EVALUTATION -> accuracy, precision, recall, F1, ...
    break
