from project_2_utils import *
from neural_net import fnn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml


import sys
import os

# Load data from https://www.openml.org/d/554
X, y_orig = fetch_openml(
    "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
)
print(X.shape, y_orig.shape)

X = X[:1000]
y_orig = y_orig[:1000]
y_orig = np.array([int(yi) for yi in y_orig])
print(X.shape, y_orig.shape)
# print(y_orig[:3])


# mnist_data = load_digits()    # this is not mnist
# mnist_images = mnist_data.images
# out_classes = mnist_data.target
# del mnist_data


eta = 0.01
epochs = 3000
# dropout_proba = 1.0
dropout_proba = 0.9
loss_name = "cross-entropy"

# dims_hidden = [32]
dims_hidden = [64]
# dims_hidden = []

scheduler = MomentumScheduler(eta, momentum=0.9)

activation = sigmoid
activation_deriv = sigmoid_derivated
# activation = softmax
# activation_deriv = derivate(softmax)

outcome = sigmoid
outcome_deriv = derivate(sigmoid)



savename = f"dims={dims_hidden}_eta={eta:.1e}_{activation.__name__}_{outcome.__name__}_dropout={dropout_proba:.2f}"


folder_save = os.path.join("figs", "mnist classif")
folder_save_results = os.path.join("results", "mnist classif")

if not os.path.exists(folder_save):
    os.mkdir(folder_save)
if not os.path.exists(folder_save_results):
    os.mkdir(folder_save_results)


num_obs = len(X)
img_dims = [int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))]
dim_input = np.prod(img_dims)

# X = mnist_images.reshape(num_obs, dim_input)
# print(X.shape)

# plt.imshow(X[0].reshape(img_dims))
# plt.show()


# Encode outcome to vector of shape (n_obs, 10) with one 1 at index of correct image label, and remaining zeros
dim_output = 10
# print(out_classes)

y = np.zeros(shape=(num_obs, 10))
for i, i_num in enumerate(y_orig.ravel()):
    y[i, int(i_num)] = 1


print(y.shape, y.shape)
# print(np.count_nonzero(y.reshape(-1)))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
out_train, out_test = train_test_split(y_orig, test_size=0.30, random_state=42)

print(y_train[:3])
print(out_train[:3])
# sys.exit()

# fig, ax = plt.subplots(ncols=2)
# ax[0].imshow(X_train[0].reshape(img_dims))

# scaler = StandardScaler()
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# ax[1].imshow(X_train[0].reshape(img_dims))
# plt.show()
# plt.close()
# sys.exit()

net = fnn(dim_input=dim_input, dim_output=dim_output, dims_hiddens=dims_hidden, learning_rate=eta, batches=3,
          loss_func_name=loss_name, scheduler=scheduler, normalize_outcome=True,
          outcome_func=outcome, outcome_func_deriv=outcome_deriv,
          activation_func=activation, activation_func_deriv=activation_deriv)

net.init_random_weights_biases(verbose=True)
loss_per_epoch, loss_per_epoch_test = net.train(X_train, y_train, X_test, y_test, dropout_retain_proba=dropout_proba, epochs=epochs, verbose=True)


net.save_state(savename, folder=folder_save_results, overwrite=True)
net.load_state(savename, folder=folder_save_results)


fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(list(range(1, epochs+1)), loss_per_epoch, label="Train")
ax.plot(list(range(1, epochs+1)), loss_per_epoch_test, label="Test")
ax.set_xlabel("epoch")
ax.set_ylabel(loss_name)
ax.legend()

yhat_test = net.predict_feed_forward(X_test)

yhat_train = net.predict_feed_forward(X_train)


mse_test = np.mean(mse_loss(yhat_test, y_test))
mse_train = np.mean(mse_loss(yhat_train, y_train))

print(f"MSE train / test = {mse_train:.2e} / {mse_test:.2e}")

yhat_train_binary = np.copy(yhat_train)
yhat_train_binary[yhat_train_binary > 0.5] = 1
yhat_train_binary[yhat_train_binary < 0.5] = 0
yhat_test_binary = np.copy(yhat_test)
yhat_test_binary[yhat_test_binary > 0.5] = 1
yhat_test_binary[yhat_test_binary < 0.5] = 0


acc_train = accuracy_score(y_train, yhat_train_binary)
acc_test = accuracy_score(y_test, yhat_test_binary)

fig.suptitle(
    f"lr={eta:.2e}, epoch={epochs:.1e}, scheduler={type(scheduler)}, activation={activation.__name__}, outcome={outcome.__name__},"
    f"\ndropout_proba={dropout_proba:.2f}, acc train / test = {np.mean(acc_train):.2f} / {np.mean(acc_test):.2f}")
fig.savefig(os.path.join(folder_save, savename + "_loss.png"))

print(f"Acc train / test = {np.mean(acc_train):.2f} / {np.mean(acc_test):.2f}")
print("\ttest accuracy per outcome class:", np.round(acc_test, 2))

auc_train = roc_auc_score(y_train, yhat_train)
auc_test = roc_auc_score(y_test, yhat_test)
print(f"ROC-AUC train / test = {np.mean(auc_train):.2f} / {np.mean(auc_test):.2f}")


print(yhat_test_binary[:3])
print(out_test[:3])

print([np.shape(w) for w in net.weights])

if not any(dims_hidden):
    fig2, ax2 = plt.subplots(ncols=5, nrows=2, figsize=(16, 8))
    ax2 = ax2.ravel()

    for i, w in enumerate(net.weights[0].T):
        ax2[i].imshow(w.reshape(img_dims))
        ax2[i].set_title(i)

    fig2.suptitle(f"lr={eta:.2e}, epoch={epochs:.1e}, scheduler={type(scheduler)}, activation={activation.__name__}, outcome={outcome.__name__},"
                 f"\ndropout_proba={dropout_proba:.2f}, acc train / test = {np.mean(acc_train):.2f} / {np.mean(acc_test):.2f}")
    fig2.tight_layout()

    fig2.savefig(os.path.join(folder_save, savename + "_weights.png"))
# plt.close()


# Compare to sklearn logistic regression
lr = LogisticRegression(multi_class="multinomial", penalty=None, tol=0.1, solver="saga")
lr.fit(X_train, out_train)

yhat_test_lr = lr.predict(X_test)
acc_test_lr = accuracy_score(out_test, yhat_test_lr)

print(f"Logistic regression on test: acc={acc_test_lr:.2f}")

plt.show()
