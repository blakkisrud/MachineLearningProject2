import sys
import os
import types

import project_2_utils as utils
from project_2_utils import ConstantScheduler
from project_2_utils import MomentumScheduler
from project_2_utils import AdagradScheduler
from project_2_utils import RMS_propScheduler
from project_2_utils import AdamScheduler

from sklearn.linear_model import Ridge
import seaborn as sns    
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

from copy import deepcopy, copy
from sklearn.metrics import mean_squared_error

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
                 lmbd = 0.0,
                 max_iterations=1000,
                 epsilon = 1.0e-8,
                 batches = 1,
                 scheduler=None, random_state=42):

        self.dim_input = dim_input
        self.dim_output = dim_output

        self.random_state = random_state

        self.loss_func_name = loss_func_name
        self.lmbd = lmbd

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

        self.epsilon = epsilon

        self.layer_sizes = [dim_input] + self.dims_hiddens + [dim_output]
        self.num_layers = len(self.layer_sizes)

        # self.weights = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) for i in range(len(self.dims_hiddens) + 1)]
        # self.biases = [np.random.randn(self.layer_sizes[i+1]) for i in range(len(self.dims_hiddens) + 1)]

        self.init_random_weights_biases()

        self.learning_rate = learning_rate # TODO: make dynamic

        self.is_trained = False

    def init_random_weights_biases(self, **kwargs):
        """
        Initialize weights and biases with random values
        """

        verbose = kwargs.get("verbose", False)
        print(f"INITIALIZING RANDOM VALUES FOR LAYERS ({self.num_hidden_layers} hidden)", [self.dim_input, *self.dims_hiddens, self.dim_output]) if verbose else 0

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
        """
        Feed-forward prediction of input X through network
        """
        print(f"PREDICTING BY FEED-FORWARDING INPUT {X.shape} THROUGH NETWORK") if kwargs.get("verbose", 0) else 0
        self.activations = [X]
        self.weighted_inputs = []

        i = 0
        Al = X
        for Wl, bl in zip(self.weights, self.biases):
            Zl = Al @ Wl + bl
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

        """
        Backpropagate error through network
        """
        # print("activations", [np.shape(a) for a in self.activations])
        # print("weights", [np.shape(w) for w in self.weights])
        verbose = kwargs.get("verbose", False)

        num_obs = y.shape[0]

        # loss = np.square(self.activations[-1] - y)     # prediction - ground truth squared
        # dC = 2 * (self.activations[-1] - y)     # derivative of squared error

        loss = self.loss_func(self.activations[-1], y, self.lmbd, self.weights)
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
            pass

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
            dW = np.dot(self.activations[l].T, delta_l)+self.lmbd*self.weights[l]
            db = np.sum(delta_l, axis=0)


            # Would dW and db be the gradiants?

            # print("dW, db", dW.shape, db.shape)
            # print(dW)

            # update weights and biases here

            change_weights = (self.schedulers_weights[l].update_change(dW))/num_obs
            change_biases = (self.schedulers_biases[l].update_change(db))/num_obs

            if self.dropout_state:
                change_weights *= self.dropout_retain_proba
                change_biases *= self.dropout_retain_proba

            self.weights[l] -= change_weights
            self.biases[l] -= change_biases

            dC = np.dot(delta_l, self.weights[l].T)

        return loss

    def train(self, X, y, X_test=[], y_test=[], scheduler=None, epochs=100, **kwargs):
        """
        Train network on input data X with ground truth y
        """
        if scheduler == None:
            scheduler = self.scheduler

        self.dropout_retain_proba = kwargs.get("dropout_retain_proba", 1.0)  # fraction of neurons to dropout for each epoch of training


        verbose = kwargs.get("verbose", False)
        kwargs["verbose"] = False


        loss_for_epochs = []
        loss_for_epochs_test = []   # only returns if y_val is not None

        batch_size = X.shape[0] // self.batches
        self.num_obs_train = X.shape[0]

        print("TRAINING NETWORK using the scheduler\t", scheduler) if verbose else 0
        print(f"\twith {self.batches} batches of size {batch_size}") if verbose else 0

        for n in range(len(self.weights)):
            self.schedulers_weights.append(copy(scheduler))
            self.schedulers_biases.append(copy(scheduler))

        #X, y = resample(X, y) # Resample the data for the mini-batches

        for e in range(epochs):

            loss_for_batches = []
            loss_for_batches_test = []
            for n in range(self.batches):

                if n == self.batches - 1:

                    X_batch = X[n*batch_size:,:]
                    y_batch = y[n*batch_size:]

                else:

                    X_batch = X[n*batch_size:(n+1)*batch_size,:]
                    y_batch = y[n*batch_size:(n+1)*batch_size]

                if self.dropout_retain_proba < 1.0:
                    self.dropout_layers()
                else:
                    self.dropout_state = False

                self.predict_feed_forward(X_batch)
                loss = self.backpropagate(y_batch, **kwargs)
                loss_for_batches.append(np.mean(loss))
                # print(loss.shape, loss_for_batches)
                # sys.exit()

                if self.dropout_state:
                    self.dropout_reset()

                if np.any(y_test):
                    self.predict_feed_forward(X_test)
                    loss_test = self.loss_func(self.activations[-1], y_test, self.lmbd, self.weights)
                    loss_for_batches_test.append(loss_test)


            # After the epoch is done, we can reset the scheduler
            for n in range(len(self.weights)):
                self.schedulers_weights[n].reset()
                self.schedulers_biases[n].reset()

            if verbose and not e%10:
                print("epoch", e, loss.shape, f"loss mean / median = {np.mean(loss):.1e} / {np.median(loss):.1e}")

                nonzero_percent = [np.count_nonzero(w.reshape(-1)) / np.prod(w.shape) * 100 for w in self.weights]
                # print("\tnonzero weights:", [f"{np.count_nonzero(np.mean(w, axis=0))} of {len(w)}" for w in self.weights])
                print(f"\tnonzero weights: {np.round(nonzero_percent, 1)} %")

            # loss_for_epochs.append(np.mean(loss))
            loss_for_epochs.append(np.mean(loss_for_batches))

            if np.any(y_test):
                loss_for_epochs_test.append(np.mean(loss_for_batches_test))

        self.is_trained = True
        self.loss_for_epochs_train = loss_for_epochs

        if not np.any(y_test):
            self.loss_for_epochs_test = []
            return loss_for_epochs
        else:
            self.loss_for_epochs_test = loss_for_epochs_test
            self.num_obs_test = X_test.shape[0]
            return loss_for_epochs, loss_for_epochs_test

    def find_optimal_epochs_kfold(self, X, y, epochs_max=int(2e3), k=3, **kwargs):
        from sklearn.model_selection import KFold

        return_loss_values = kwargs.get("return_loss_values", False)
        plot = kwargs.get("plot", True)
        verbose = kwargs.get("verbose", False)
        print(f"FINDING optimal number of epochs using {k}-fold validation")

        # Training / validation average loss over epochs for each fold k
        loss_training = []
        loss_validation = []

        for ind_train, ind_val in KFold(n_splits=k, shuffle=True, random_state=self.random_state).split(X, y):
            x_train, y_train = X[ind_train], y[ind_train]
            x_val, y_val = X[ind_val], y[ind_val]

            self.init_random_weights_biases()   # reset weights and biases

            loss_train_k, loss_val_k = self.train(x_train, y_train, X_test=x_val, y_test=y_val, epochs=epochs_max, **kwargs)

            loss_training.append(loss_train_k)
            loss_validation.append(loss_val_k)

        epochs_opt = np.argmin(np.mean(loss_validation, axis=0)) + 1
        print("\toptimal number of epochs = ", epochs_opt)

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

    def evaluate(self, evaluation_X, evaluation_y):

        yhat = self.predict_feed_forward(evaluation_X)

        y_hat_binary = np.zeros((yhat.shape[0],1))
        y_hat_binary[yhat > 0.5] = 1

        mse = mean_squared_error(evaluation_y, yhat)
        accuracy = accuracy_score(evaluation_y, y_hat_binary)

        #TODO: Do accuracy instead

        return mse, accuracy

    def dropout_layers(self):
        '''
        Retains self.dropout_retain_proba fraction of neurons in each layer, rest is dropped out temporarily.
        '''

        self.neurons_to_retain = [np.random.choice(range(dh), int(dh * self.dropout_retain_proba), replace=False) for dh in
                             self.dims_hiddens]
        # print([dh for dh in self.dims_hiddens])
        # print([len(nrt) for nrt in self.neurons_to_retain])
        # sys.exit()

        self.weights_before_dropout = copy(self.weights)
        self.biases_before_dropout = copy(self.biases)

        # print(self.neurons_to_retain)
        # print([np.shape(w) for w in self.weights])
        # print([np.shape(b) for b in self.biases])

        self.weights = []
        self.biases = []

        for l in range(len(self.dims_hiddens) + 1):

            if not l == len(self.dims_hiddens):
                ns_rt = self.neurons_to_retain[l]
                self.weights.append(self.weights_before_dropout[l][:, ns_rt])
                self.biases.append(self.biases_before_dropout[l][ns_rt])

                if not l == 0:
                    self.weights[l] = self.weights[l][ns_rt_prev, :]

            else:
                self.weights.append(self.weights_before_dropout[l][ns_rt_prev, :])
                self.biases.append(self.biases_before_dropout[-1])

            ns_rt_prev = ns_rt

        # print([np.shape(w) for w in self.weights])
        # print([np.shape(b) for b in self.biases])
        self.dropout_state = True

        pass


    def dropout_reset(self):
        '''
        Sets the weights tuned after dropping out some neurons,
        while resetting dropout neurons to weight and bias values from before dropout was initialized
        Normalization to dropout probability is done in the backpropagation algorithm, when self.dropout_state is True
        '''
        # self.weights_dropout = [w * self.dropout_retain_proba for w in self.weights]
        # self.biases_dropout = [b * self.dropout_retain_proba for b in self.biases]
        self.weights_dropout = self.weights
        self.biases_dropout = self.biases


        self.weights = self.weights_before_dropout
        self.biases = self.biases_before_dropout

        for l in range(len(self.dims_hiddens) + 1):

            if not l == len(self.dims_hiddens):
                ns_rt = self.neurons_to_retain[l]
                self.biases[l][ns_rt] = self.biases_dropout[l]
                if l == 0:
                    self.weights[l][:, ns_rt] = self.weights_dropout[l]
                else:
                    self.weights[l][ns_rt_prev[:, None], ns_rt[None, :]] = self.weights_dropout[l]
            else:   # final layer
                self.biases[-1] = self.biases_dropout[-1]
                self.weights[l][ns_rt_prev, :] = self.weights_dropout[l]

            ns_rt_prev = ns_rt

        self.dropout_state = False
        pass


    def save_state(self, name, folder="results", overwrite=False):
        # SAVES: dictionary of metadata, loss for epochs during training, current weights & biases
        savepath_meta = os.path.join(folder, name + "_meta.npy")
        savepath_weightsbiases = os.path.join(folder, name + "_wb.npy")
        savepath_loss = os.path.join(folder, name + "_loss.npy")
        print("--- SAVING NN STATE", name, "---")
        attr = dir(self)
        attr = list(filter(lambda a: a[:2] != "__", attr))

        meta_dict = {}
        for att_nm in attr:
            att = self.__getattribute__(att_nm)
            try:
                att = int(att)
            except Exception:
                pass

            if type(att) in [str, int, float, bool]:
                meta_dict[att_nm] = att

            elif type(att) == types.FunctionType:
                meta_dict[att_nm] = att.__name__

            elif att_nm == "scheduler":
                meta_dict[att_nm] = type(att)
                attr_sch = list(filter(lambda a: a[:2] != "__", dir(att)))
                for att_sch_nm in attr_sch:
                    att_sch = att.__getattribute__(att_sch_nm)
                    # print(att_sch_nm, type(att_sch))
                    if not(type(att_sch)) == types.MethodType:
                        meta_dict["scheduler." + att_sch_nm] = att_sch

            elif att_nm == "dims_hiddens":
                meta_dict[att_nm] = att

        print("SAVING meta_dict in folder", folder)
        print(meta_dict)
        save_state = False

        if os.path.exists(savepath_meta) and not overwrite:
            print("FILE", savepath_meta, "ALREADY EXISTS. DO YOU WANT TO OVERWRITE?")
            inp = input("y / n?\n")
            # TODO: load meta from file, compare, create new file if not equal overwrite if equal
            if inp == "y":
                np.save(savepath_meta, meta_dict)
                save_state = True
        else:
            np.save(savepath_meta, meta_dict)
            save_state = True

        if save_state:
            # save weights, biases, loss during training
            print("SAVING WEIGHTS", [np.shape(w) for w in self.weights],
            "BIASES", [np.shape(b) for b in self.biases], "LOSS DURING TRAINING EPOCHS", np.shape(self.loss_for_epochs_train))

            self.weights = np.array([*self.weights], dtype="object")
            self.biases = np.array([*self.biases], dtype="object")
            wb = np.array([self.weights,  self.biases], dtype="object")
            np.save(savepath_weightsbiases, wb)

            loss = [self.loss_for_epochs_train]
            loss.append(self.loss_for_epochs_test) if any(self.loss_for_epochs_test) else None

            loss = np.array(loss)
            np.save(savepath_loss, loss)

            print("\tSAVE SUCCESSFUL")

        else:
            print("\tQUITTING SAVE STATE")
        pass


    def load_state(self, name, folder="results"):
        path_meta = os.path.join(folder, name + "_meta.npy")
        path_wb = os.path.join(folder, name + "_wb.npy")
        path_loss = os.path.join(folder, name + "_loss.npy")
        print("--- LOADING NN STATE", name, "---")

        meta_dict = np.load(path_meta, allow_pickle=True).item()

        for attrs in meta_dict.items():
            errs = ""
            try:
                self.__setattr__(*attrs)
            except Exception as e:
                errs += e + "\n"
        del meta_dict

        weights, biases = np.load(path_wb, allow_pickle=True)

        self.weights = weights
        self.biases = biases
        print([np.shape(w) for w in weights])
        del weights, biases

        loss = np.load(path_loss)
        print(loss.shape)
        if loss.shape[0] == 2:
            self.loss_for_epochs_train, self.loss_for_epochs_test = loss
        else:
            self.loss_for_epochs_train = loss
        del loss

        if any(errs):
            print("\tLOADED NN-STATE with some errors:")
            print(errs)
        else:
            print("\tLOADED NN-STATE with no errors :)")
        pass


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

    #dims_hidden = [8, 8, 8]
    #dims_hidden = [4]
    dims_hidden = []

    lr = 0.1
    epochs = 100
    search_values = False

    if search_values == True:
        eta_vals = np.logspace(-5, 1, 7)
        lmbd_vals = np.logspace(-5, 1, 7)
        # store the models for later use
        DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
        mse_list = np.zeros((len(eta_vals), len(lmbd_vals)))
        fig, ax = plt.subplots(figsize = (10, 10))    

        # grid search
        for i, eta in enumerate(eta_vals):
            for j, lmbd in enumerate(lmbd_vals):
                dnn = fnn(dim_input=input_dim, dim_output=output_dim, dims_hiddens=dims_hidden, learning_rate=eta,
                activation_func=activation_func, outcome_func=outcome_func, activation_func_deriv=activation_func_deriv, 
                outcome_func_deriv=outcome_func_deriv,
                lmbd = lmbd,
                batches=32,
                scheduler=AdamScheduler(eta, 0.9, 0.999))
                dnn.train(X, y, epochs=epochs, 
                                        scheduler=AdamScheduler(eta, 0.9, 0.999),
                                        plot=False, figax=(fig, ax), showplot=False, plot_title=f"MSE lr = {dnn.learning_rate}", verbose=False)
        
                DNN_numpy[i][j] = dnn
                yhat = dnn.predict_feed_forward(X)
                mse = mean_squared_error(y, yhat)
                mse_list[i][j] = mse
        sns.heatmap(mse_list, annot=True, ax=ax, cmap="viridis")
        ax.set_title("Training Accuracy")
        ax.set_ylabel("$\eta$")
        ax.set_xlabel("$\lambda$")
        plt.savefig('Accuracy board.png')
        plt.show()



    net = fnn(dim_input=input_dim, dim_output=output_dim, dims_hiddens=dims_hidden, learning_rate=lr,
              activation_func=activation_func, outcome_func=outcome_func, activation_func_deriv=activation_func_deriv, 
              outcome_func_deriv=outcome_func_deriv,
              batches=32,
              lmbd = 0.001, 
              scheduler=AdamScheduler(lr, 0.9, 0.999))

    #net.find_optimal_epochs_kfold(X, y)

    # Plot MSE for epochs for repeated random initialization

    nrand = 1
    plot = True

    fig_folder = r"figs\neural_test"
    if not os.path.exists(fig_folder):
        os.mkdir(fig_folder)

    title = f"hidden dims = {net.dims_hiddens}, repeated with random initiation {nrand} times, eta={net.learning_rate:.3e}, N_obs={num_obs}"
    savename = f"nn={net.layer_sizes}_lr={net.learning_rate}.png"
    fig_path = os.path.join(fig_folder, savename)

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

        # Compare to Ridge regression    
        clf = Ridge(alpha=0.1)
        x1 = x.reshape(-1,1)
        y1 = y.reshape(-1,1)
        clf.fit(x1, y1)
        y_pred = clf.predict(x1)
        mse = mean_squared_error(y_pred, y)
        print(f'MSE Ridge regression: {mse:.2}')
        ax1.plot(x, y_pred, linewidth=linewidth, c="orange", zorder=1, label='Ridge regression')

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