import sys
import os

import project_2_utils as utils
from project_2_utils import ConstantScheduler
from project_2_utils import MomentumScheduler
from project_2_utils import AdagradScheduler
from project_2_utils import RMS_propScheduler
from project_2_utils import AdamScheduler

from copy import deepcopy, copy

import autograd.numpy as np

from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from matplotlib import pyplot as plt
                    
from sklearn.utils import resample


class fnn():
    def __init__(self, dim_input, dim_output, dims_hiddens,
                 activation_func, outcome_func, 
                 activation_func_deriv, outcome_func_deriv,
                 loss_func_name="MSE",
                 learning_rate=1e-4,
                 max_iterations=1000,
                 epsilon = 1.0e-8,
                 batches = 1,
                 scheduler=None, random_state=42):

        self.dim_input = dim_input
        self.dim_output = dim_output

        self.random_state = random_state

        self.loss_func_name = loss_func_name


        # INITIALIZING LOSS FUNCTION
        if loss_func_name.upper() == "MSE":
            self.loss_func = utils.mse_loss
            self.loss_func_deriv = utils.mse_loss_deriv

        elif loss_func_name.upper() == "CROSS-ENTROPY":
            self.loss_func = utils.cross_entropy_loss
            self.loss_func_deriv = utils.cross_entropy_loss_deriv

        else:
            print("LOSS FUNCTION", loss_func_name, "NOT IMPLEMENTED...")
            sys.exit()


        self.batches = batches
        self.scheduler = scheduler if not scheduler == None else utils.ConstantScheduler(eta=learning_rate)

        self.dims_hiddens = dims_hiddens    # tuple of neurons per hidden layer, e.g. (8) for single layer or (16, 8, 8) for three hidden layers
        self.num_hidden_layers = len(dims_hiddens)
        self.activation_func = activation_func
        self.activation_func_deriv = activation_func_deriv
        self.outcome_func = outcome_func
        self.outcome_func_deriv = outcome_func_deriv

        # Schedule list for Weidghts and biases

        self.schedulers_weights = []
        self.schedulers_biases = []

        self.max_iterations = max_iterations
        self.epsilon = epsilon

        self.layer_sizes = [dim_input] + self.dims_hiddens + [dim_output]
        self.num_layers = len(self.layer_sizes)


        self.weights = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) for i in range(len(self.dims_hiddens) + 1)]
        self.biases = [np.random.randn(self.layer_sizes[i+1]) for i in range(len(self.dims_hiddens) + 1)]

        self.init_random_weights_biases()

        self.learning_rate = learning_rate # TODO: make dynamic

    def init_random_weights_biases(self):
        print(f"INITIALIZING RANDOM VALUES FOR LAYERS ({self.num_hidden_layers} hidden)", [self.dim_input, *self.dims_hiddens, self.dim_output])

        # store arrays of weights and biases for edges between all layers
        weights = []
        biases = []

        dims_prev = self.dim_input

        for dims in [*self.dims_hiddens, self.dim_output]:
            # weights.append(np.random.randn(dims_prev, dims))
            # biases.append(np.random.randn(dims))

            weights.append(np.random.normal(size=(dims_prev, dims), loc = 0.01, scale = 1e-3))
            biases.append(np.random.normal(size=(dims)) + 1e-3)

            dims_prev = dims

        self.weights = weights
        self.biases = biases

        pass

    def predict_feed_forward(self, X, **kwargs):
        print(f"PREDICTING BY FEED-FORWARDING INPUT {X.shape} THROUGH NETWORK") if kwargs.get("verbose", 0) else 0
        self.activations = [X]
        self.weighted_inputs = []

        i = 0
        Al = X
        for Wl, bl in zip(self.weights, self.biases):
            Zl = Al @ Wl + bl
            # todo: linear activation function for final layer? (i.e. the identity function)
            i += 1

            if i > self.num_hidden_layers:  # FINAL LAYER
                Al = self.outcome_func(Zl)

            else:                           # HIDDEN LAYERS
                Al = self.activation_func(Zl)

            self.activations.append(Al)
            self.weighted_inputs.append(Zl)

        self.A = Al
        return Al


    def backpropagate(self, y, **kwargs):
        # print("activations", [np.shape(a) for a in self.activations])
        # print("weights", [np.shape(w) for w in self.weights])
        verbose = kwargs.get("verbose", False)

        num_obs = len(y)

        # loss = np.square(self.activations[-1] - y)     # prediction - ground truth squared
        # dC = 2 * (self.activations[-1] - y)     # derivative of squared error
        loss = self.loss_func(self.activations[-1], y)
        dC = self.loss_func_deriv(self.activations[-1], y)

        # print(loss.shape, np.mean(loss))
        # print(dC.shape, np.mean(dC))
        # sys.exit()


        if verbose:
            print("num layers\t", self.num_layers)
            print("weights\t", [np.shape(w) for w in self.weights])
            print("biases\t", [np.shape(b) for b in self.biases])
            print("activations\t\t", [np.shape(a) for a in self.activations])
            print("weighted inputs\t", [np.shape(z) for z in self.weighted_inputs])


        for l in range(self.num_layers - 2, -1, -1):
            print("l=", l) if verbose else 0

            # FINAL / OUTPUT LAYER
            if l == self.num_layers - 2:
                # print("FINAL LAYER")
                f_deriv_zl = self.outcome_func_deriv(self.weighted_inputs[l])


            # HIDDEN LAYER
            else:
                # print("HIDDEN LAYER")
                f_deriv_zl = self.activation_func_deriv(self.weighted_inputs[l])

            delta_l = dC * f_deriv_zl


            # cost rate of change with respect to weights and biases in layer l

            dW = np.dot(self.activations[l].T, delta_l)
            db = np.sum(delta_l, axis=0)

            # Would dW and db be the gradiants?

            # print("dW, db", dW.shape, db.shape)
            # print(dW)

            # update weights and biases here

            change_weights = (self.schedulers_weights[l].update_change(dW))/num_obs
            change_biases = (self.schedulers_biases[l].update_change(db))/num_obs

            self.weights[l] -= change_weights
            self.biases[l] -= change_biases

            dC = np.dot(delta_l, self.weights[l].T)

        return loss


    def train(self, X, y, X_test=[], y_test=[], scheduler=None, epochs=100, **kwargs):
        if scheduler == None:
            scheduler = self.scheduler
        verbose = kwargs.get("verbose", False)

        loss_for_epochs = []
        loss_for_epochs_test = []   # only returns if y_val is not None

        batch_size = X.shape[0] // self.batches

        print("TRAINING NETWORK using the scheduler\t", scheduler)
        print(f"\twith {self.batches} batches of size {batch_size}")

        for n in range(len(self.weights)):
            self.schedulers_weights.append(copy(scheduler))
            self.schedulers_biases.append(copy(scheduler))

        #X, y = resample(X, y) # Resample the data for the mini-batches

        for e in range(epochs):

            print(f"epoch {e}", end="\r") 
            loss_for_batches = []
            loss_for_batches_test = []
            for n in range(self.batches):

                if n == self.batches - 1:

                    X_batch = X[n*batch_size:,:]
                    y_batch = y[n*batch_size:]

                else:

                    X_batch = X[n*batch_size:(n+1)*batch_size,:]
                    y_batch = y[n*batch_size:(n+1)*batch_size]

                self.predict_feed_forward(X_batch)
                loss = self.backpropagate(y_batch, **kwargs)
                loss_for_batches.append(np.mean(loss))
                # print(loss.shape, loss_for_batches)
                # sys.exit()

                if any(y_test):
                    self.predict_feed_forward(X_test)
                    loss_test = self.loss_func(self.activations[-1], y_test)
                    loss_for_batches_test.append(loss_test)


            # After the epoch is done, we can reset the scheduler
            for n in range(len(self.weights)):
                self.schedulers_weights[n].reset()
                self.schedulers_biases[n].reset()

            print(e, loss.shape, f"{np.mean(loss):.3e}, {np.median(loss):.3e}") if verbose else 0
            # loss_for_epochs.append(np.mean(loss))
            loss_for_epochs.append(np.mean(loss_for_batches))

            if any(y_test):
                loss_for_epochs_test.append(np.mean(loss_for_batches_test))

        if not any(y_test):
            return loss_for_epochs
        else:
            return loss_for_epochs, loss_for_epochs_test

    def find_optimal_epochs_kfold(self, X, y, epochs_max=int(2e3), k=3, **kwargs):
        from sklearn.model_selection import KFold

        return_loss_values = kwargs.get("return_loss_values", False)
        plot = kwargs.get("plot", True)

        print(f"FINDING optimal number of epochs using {k}-fold validation")

        # Training / validation average loss over epochs for each fold k
        loss_training = []
        loss_validation = []

        for ind_train, ind_val in KFold(n_splits=k, shuffle=True, random_state=self.random_state).split(X, y):
            x_train, y_train = X[ind_train], y[ind_train]
            x_val, y_val = X[ind_val], y[ind_val]
            print(x_train.shape, x_val.shape)

            self.init_random_weights_biases()   # reset weights and biases

            loss_train_k, loss_val_k = self.train(x_train, y_train, X_test=x_val, y_test=y_val, epochs=epochs_max, **kwargs)

            loss_training.append(loss_train_k)
            loss_validation.append(loss_val_k)

        epochs_opt = np.argmin(np.mean(loss_validation, axis=0)) + 1
        print("optimal number of epochs = ", epochs_opt)

        if plot:
            epochs = list(range(1, epochs_max + 1))
            fig_ep, ax_ep = plt.subplots(ncols=2)
            ax_ep[0].set_title("Training loss")
            ax_ep[1].set_title("Validation loss")

            for i in range(k):
                ax_ep[0].plot(epochs, loss_training[i], c=f"C{i}")
                ax_ep[1].plot(epochs, loss_validation[i], c=f"C{i}")

            plt.show()
        if not return_loss_values:
            return epochs_opt
        else:
            return epochs_opt, loss_training, loss_validation

    def evaluate(self, validation_data_X, validation_data_y):
        pass

if __name__ == "__main__":

    input_mode = 2
    valid_inputs = [1, 2]

    while input_mode not in valid_inputs:
        input_mode = input("SELECT INPUT: 1 = simple one-dimensional function, 2 = single MNIST digits image, 3 = franke function (not implemented), 4 = terrain image (not implemented)\n")
        try:
            input_mode = int(input_mode)
        except Exception as e:
            print(*e.args)

        if not input_mode in valid_inputs:
            print("Invalid input, please try again...")

    # LOAD ONE-DIM FUNCTION R1 -> R1
    if input_mode == 1:
        print("LOADING SIMPLE ONE-DIMENSIONAL FUNCTION")
        x = np.arange(0, 10, 0.01)
        X = utils.one_d_design_matrix(x, n=1)
        X = X[:, 1]     # remove bias from design matrix
        X = X.reshape(-1, 1)

        y = utils.simple_func(x, 1, 5, 3, noise_sigma=2.0)
        # y = y.reshape(1, -1)
        output_dim = 1   # each observation have only one dimension
        input_dim = 1
        # plt.plot(x, y)
        # plt.show()

    # LOAD SINGLE MNIST-IMAGE R2 -> R2
    if input_mode == 2:
        print("LOADING SINGLE MNIST IMAGE")
        dataset_digits = load_digits()
        print(dataset_digits.images.shape)
        img = dataset_digits.images[0]
        del dataset_digits
        print("IMAGE SHAPE:", img.shape)
        y = img.ravel()
        X = np.arange(0, len(y)).reshape(1, -1)

        # use all pixels for both training and testing
        # TODO: implement 2-dimensional design matrix
        input_dim = np.prod(img.shape)
        output_dim = np.prod(img.shape)


    y = MinMaxScaler(feature_range=(0, 1)).fit_transform(y.reshape(-1, 1)).reshape(-1, 1)
    num_obs = len(y)
    print("SHAPE x / y:", X.shape, y.shape)

    # activation_func = np.vectorize(lambda z: 1 / (1 + np.exp(-z)))  # sigmoidal activation
    # outcome_func = np.vectorize(lambda z: z)    # identity function

    activation_func = utils.sigmoid
    activation_func_deriv = utils.sigmoid_derivated

    outcome_func = utils.identity
    outcome_func_deriv = utils.identity_derived

    # dims_hidden = [8, 8, 8]
    dims_hidden = [4]
    # dims_hidden = []

    lr = 0.1
    epochs = 100

    net = fnn(dim_input=input_dim, dim_output=output_dim, dims_hiddens=dims_hidden, learning_rate=lr,
              activation_func=activation_func, outcome_func=outcome_func, activation_func_deriv=activation_func_deriv, 
              outcome_func_deriv=outcome_func_deriv,
              batches=32,
              scheduler=AdamScheduler(lr, 0.9, 0.999))

    net.find_optimal_epochs_kfold(X, y)

    # Plot MSE for epochs for repeated random initialization

    nrand = 1
    plot = True

    fig_folder = r"figs\neural_test"
    if not os.path.exists(fig_folder):
        os.mkdir(fig_folder)

    title = f"hidden dims = {net.dims_hiddens}, repeated with random initiation {nrand} times, eta={net.learning_rate:.3e}, N_obs={num_obs}"
    savename = f"nn={net.layer_sizes}_lr={net.learning_rate}.png"
    fig_path = os.path.join(fig_folder, savename)

    from sklearn.metrics import mean_squared_error

    if plot:
        fig, ax = plt.subplots(ncols=2, figsize=(12, 8))
        ax, ax1 = ax
        ax1.set_ylim(0, 1)
        # fig1, ax1 = plt.subplots()

        net.learning_rate = lr
        linewidth = 4.0
        max_loss = 0

        for i in range(nrand):
            net.init_random_weights_biases()

            loss_epochs = net.train(X, y, epochs=epochs, 
                                    scheduler=AdamScheduler(lr, 0.9, 0.999),
                                    plot=False, figax=(fig, ax), showplot=False, plot_title=f"MSE lr = {net.learning_rate}", verbose=False)
            # print(net.weights, net.biases)

            loss_final = loss_epochs[-1]
            max_loss = loss_final if loss_final > max_loss else max_loss

            yhat = net.predict_feed_forward(X)
            mse = mean_squared_error(y, yhat)
            mse2 = mean_squared_error(y, net.activations[-1])
            print(i, f"mse={mse:.2e}, {mse2:.2e}")
            print(f"weights", net.weights, "biases", net.biases)


            ax.plot(list(range(len(loss_epochs))), loss_epochs, c=f"C{i}", label=i)
            ax1.plot(x, yhat, c=f"C{i}", label=i)


        ax1.plot(x, y, linewidth=linewidth, c="black", zorder=0)
        # ax.plot([0, epochs], [0, 0], linewidth=linewidth, c="black", zorder=0)
        ax.set_title(f"MSE during training")
        ax.set_xlabel("Epochs")

        ax1.set_title(f"predictions post-training")
        ax1.set_xlabel("x")
        ax.set_ylim(0, max_loss*1.1)

        ax.legend()
        ax1.legend()

        fig.suptitle(title)
        fig.savefig(fig_path)
        plt.show()


    for wi, bi in zip(net.weights, net.biases):
        print(wi.shape, bi.shape)

    yhat = net.predict_feed_forward(X)
    print(yhat.shape)

    if input_mode == 1:
        fig, ax = plt.subplots()
        ax.plot(x, y, label="y")
        ax.plot(x, yhat, label="yhat")

        #for j in range(5):
        #    net.init_random_weights_biases()
        #    yhat = net.predict_feed_forward(X)
        #    ax.plot(x, yhat, color="C1")

    if input_mode == 2:
        img_pred = yhat.reshape(img.shape)
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(img, cmap="gray");
        ax[0].set_title("Ground truth")
        ax[1].imshow(img_pred, cmap="gray");
        ax[1].set_title("Prediction")

    # plt.show()
    plt.close()