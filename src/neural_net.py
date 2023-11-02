import sys

import project_2_utils as utils
import autograd.numpy as np

from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from matplotlib import pyplot as plt

## ITERATION 1: predict a single MNIST image by overfitting a feed-forward neural network


class fnn():
    def __init__(self, dim_input, dim_output, dims_hiddens, activation_func, outcome_func, learning_rate=1e-4):
        self.dim_input = dim_input
        self.dim_output = dim_output

        self.dims_hiddens = dims_hiddens    # tuple of neurons per hidden layer, e.g. (8) for single layer or (16, 8, 8) for three hidden layers
        self.num_hidden_layers = len(dims_hiddens)
        self.activation_func = activation_func
        self.outcome_func = outcome_func

        self.layer_sizes = [dim_input] + self.dims_hiddens + [output_dim]

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
            weights.append(np.random.randn(dims_prev, dims))
            biases.append(np.random.randn(dims))

            dims_prev = dims

        self.weights = weights
        self.biases = biases

        pass


    def predict_feed_forward(self, X, **kwargs):
        print(f"PREDICTING BY FEED-FORWARDING INPUT {X.shape} THROUGH NETWORK") if kwargs.get("verbose", 0) else 0
        self.activations = [X]
        i = 0
        Al = X
        for Wl, bl in zip(self.weights, self.biases):
            Zl = Al @ Wl + bl
            # todo: linear activation function for final layer? (i.e. the identity function)
            i += 1
            if i > self.num_hidden_layers:
                Al = self.outcome_func(Zl)
            else:
                Al = self.activation_func(Zl)
            self.activations.append(Al)

        self.A = Al
        return Al


    def backpropagate(self, y, **kwargs):
        print("activations", [np.shape(a) for a in self.activations])
        print("weights", [np.shape(w) for w in self.weights])


        loss = np.square(self.activations[-1] - y)     # prediction - ground truth

        dA = 2 * (self.activations[-1] - y)     # derivative of squared error
        # dZ = dA * self.sigmoid_derivatives(self.A)
        self.sigmoid_derivatives = []
        for i in range(len(self.weights)-1, -1, -1):
            sd = utils.sigmoid_derivated(self.A)
            self.sigmoid_derivatives.append(sd)
            dZ = dA * utils.sigmoid_derivated(self.activations[i+1])
            dW = np.dot(self.activations[i].T, dZ)
            db = np.sum(dZ, axis=0)
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
            dA = np.dot(dZ, self.weights[i].T)

        return loss


    def train(self, X, y, stochastic=True, epochs=10, **kwargs):
        plot = kwargs.get("plot")
        figax = kwargs.get("figax", (0, 0))
        if plot or figax:
            fig, ax = figax

            showplot = kwargs.get("showplot", False)
            plot = True

        # fig, ax = plt.subplots() if plot else (0, 0)
        errs = []
        for e in range(epochs):
            if stochastic:
                from sklearn.utils import resample
                X_samp, y_samp = resample(X, y)
                self.predict_feed_forward(X_samp)
                loss = self.backpropagate(y_samp)
            else:
                self.predict_feed_forward(X)
                loss = self.backpropagate(y)
            # ax.plot(list(range(len(loss))), loss, "x", label=f"{e}") if plot else 0
            errs.append(np.mean(loss))
            print(loss.shape) if kwargs.get("verbose", 0) else 0
        ax.plot(list(range(epochs)), errs) if plot else 0
        ax.set_title(kwargs.get("plot_title")) if plot else 0
        ax.set_xlabel(kwargs.get("plot_xlabel")) if plot else 0

        plt.show() if showplot else 0
        return errs


if __name__ == "__main__":

    input_mode = 1
    valid_inputs = [1, 2]

    while input_mode not in valid_inputs:
        input_mode = input("SELECT INPUT: 1 = simple one-dimensional function, 2 = single MNIST digits image, 3 = franke function (not implemented), 4 = terrain image (not implemented)\n")
        try:
            input_mode = int(input_mode)
        except Exception as e:
            print(*e.args)

        if not input_mode in valid_inputs:
            print("Invalid input, please try again...")

    if input_mode == 1:
        print("LOADING SIMPLE ONE-DIMENSIONAL FUNCTION")
        x = np.arange(0, 10, 0.1)
        X = utils.one_d_design_matrix(x, n=1)
        X = X[:, 1]     # remove bias from design matrix
        X = X.reshape(-1, 1)

        y = utils.simple_func(x, 1, 5, 3, noise_sigma=2.0)
        # y = y.reshape(1, -1)
        output_dim = 1   # each observation have only one dimension
        input_dim = 1
        # plt.plot(x, y)
        # plt.show()


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
    print("SHAPE x / y:", X.shape, y.shape)


    activation_func = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))  # sigmoidal activation
    outcome_func = np.vectorize(lambda x: x)    # identity function

    # dims_hidden = [8, 8, 8]
    dims_hidden = [4]
    net = fnn(dim_input=input_dim, dim_output=output_dim, dims_hiddens=dims_hidden, activation_func=activation_func, outcome_func=outcome_func)

    # net.predict_feed_forward(X)
    # net.backpropagate(y)


    # Plot MSE for epochs for repeated random initialization
    plot = True
    stochastic = True

    if plot:
        fig, ax = plt.subplots(ncols=2)
        ax, ax1 = ax
        ax.set_ylim(-1, 1)
        ax1.set_ylim(0, 1)
        # fig1, ax1 = plt.subplots()

        lr = 5e-3
        epochs = 10
        net.learning_rate = lr
        linewidth = 4.0

        nrand = 50
        for i in range(nrand):
            net.init_random_weights_biases()

            # if stochastic:
                # from sklearn.utils import resample
                # X_samp, y_samp = resample(X, y)
                # errs = net.train(X_samp, y_samp, plot=False, figax=(fig, ax), showplot=False, plot_title=f"MSE lr = {net.learning_rate}")

            # else:
            errs = net.train(X, y, stochastic=True, plot=False, figax=(fig, ax), showplot=False, plot_title=f"MSE lr = {net.learning_rate}")

            ax.plot(list(range(len(errs))), errs)

            yhat = net.predict_feed_forward(X)
            ax1.plot(x, y, linewidth=linewidth, c="black")
            ax1.plot(x, yhat, ":")

        ax.plot([0, epochs], [0, 0], linewidth=linewidth, c="black")
        ax.set_title(f"MSE post-training with random initiation {nrand} times for eta={net.learning_rate}")
        ax.set_xlabel("Epochs")

        ax1.set_title(f"predictions post-training with random initiation {nrand} times for eta={net.learning_rate}")
        ax1.set_xlabel("x")

        fig.suptitle(f"hidden dims = {net.dims_hiddens}")

        plt.show()


    for wi, bi in zip(net.weights, net.biases):
        print(wi.shape, bi.shape)

    yhat = net.predict_feed_forward(X)
    print(yhat.shape)

    if input_mode == 1:
        fig, ax = plt.subplots()
        ax.plot(x, y, label="y")
        ax.plot(x, yhat, label="yhat")

        for j in range(5):
            net.init_random_weights_biases()
            yhat = net.predict_feed_forward(X)
            ax.plot(x, yhat, color="C1")

    if input_mode == 2:
        img_pred = yhat.reshape(img.shape)
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(img, cmap="gray");
        ax[0].set_title("Ground truth")
        ax[1].imshow(img_pred, cmap="gray");
        ax[1].set_title("Prediction")

    # plt.show()
    plt.close()