import sys

import project_2_utils as p2utils
import autograd.numpy as np

from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from matplotlib import pyplot as plt

## ITERATION 1: predict a single MNIST image by overfitting a feed-forward neural network


class fnn():
    def __init__(self, dim_input, dim_output, dims_hiddens, activation_func):
        self.dim_input = dim_input
        self.dim_output = dim_output

        self.dims_hiddens = dims_hiddens    # tuple of neurons per hidden layer, e.g. (8) for single layer or (16, 8, 8) for three hidden layers
        self.num_hidden_layers = len(dims_hiddens)
        self.activation_func = activation_func

        self.weights, self.biases = self.init_random_weights_biases()

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

        return weights, biases

    def predict_feed_forward(self, X):
        print(f"PREDICTING BY FEED-FORWARDING INPUT {X.shape} THROUGH NETWORK")
        Al = X
        for Wl, bl in zip(self.weights, self.biases):
            Zl = Al @ Wl + bl
            Al = self.activation_func(Zl)
            # todo: linear activation function for final layer? (i.e. the identity function)
        return Al


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

    if input_mode == 1:
        print("LOADING SIMPLE ONE-DIMENSIONAL FUNCTION")
        x = np.arange(0, 10, 0.1)
        X = p2utils.one_d_design_matrix(x, n=1)
        X = X[:, 1]     # remove bias from design matrix
        X = X.reshape(1, -1)

        y = p2utils.simple_func(x, 1, 5, 3, noise_sigma=2.0)
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

    net = fnn(dim_input=input_dim, dim_output=output_dim, dims_hiddens=[8], activation_func=activation_func)

    for wi, bi in zip(net.weights, net.biases):
        print(wi.shape, bi.shape)

    yhat = net.predict_feed_forward(X)
    print(yhat.shape)

    if input_mode == 1:
        fig, ax = plt.subplots()
        ax.plot(x, y, label="y")
        ax.plot(x, yhat, label="yhat")

    if input_mode == 2:
        img_pred = yhat.reshape(img.shape)
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(img, cmap="gray");
        ax[0].set_title("Ground truth")
        ax[1].imshow(img_pred, cmap="gray");
        ax[1].set_title("Prediction")

    plt.show()

