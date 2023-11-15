from project_2_utils import *
from neural_net import fnn
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
import os

mnist_data = load_digits()
mnist_images = mnist_data.images
out_classes = mnist_data.target
del mnist_data


eta = 0.1
epochs = 100
dropout_proba = 1.0
loss_name = "cross-entropy"

# dims_hidden = [32]
dims_hidden = []
scheduler = MomentumScheduler(eta, momentum=0.9)

savename = f"dims={dims_hidden}_eta={eta:.1e}_dropout={dropout_proba:.2f}.png"

# dims_hidden = [32]
# scheduler = ConstantScheduler(eta)
folder_save = os.path.join("figs", "mnist classif")
if not os.path.exists(folder_save):
    os.mkdir(folder_save)


print(mnist_images.shape)
print(out_classes.shape)
print(out_classes)

num_obs = len(out_classes)
img_dims = mnist_images.shape[1:]
dim_input = np.prod(img_dims)

X = mnist_images.reshape(num_obs, dim_input)
print(X.shape)

# plt.imshow(X[0].reshape(img_dims))
# plt.show()


# Encode outcome to vector of shape (n_obs, 10) with one 1 at index of correct image label, and remaining zeros
dim_output = 10
print(out_classes)
y = np.zeros(shape=(num_obs, 10))
for i, i_num in enumerate(out_classes):
    y[i, i_num] = 1

print(y.shape, y.reshape(-1).shape)
print(np.count_nonzero(y.reshape(-1)))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(X_train[0].reshape(img_dims))

# scaler = StandardScaler()
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

ax[1].imshow(X_train[0].reshape(img_dims))
# plt.show()
plt.close()
# sys.exit()



net = fnn(dim_input=dim_input, dim_output=dim_output, dims_hiddens=dims_hidden, learning_rate=eta, batches=3,
          loss_func_name=loss_name, scheduler=scheduler,
          outcome_func=sigmoid, outcome_func_deriv=sigmoid_derivated,
          activation_func=sigmoid, activation_func_deriv=sigmoid_derivated)

net.init_random_weights_biases(verbose=True)
loss_per_epoch = net.train(X_train, y_train, dropout_retain_proba=dropout_proba, epochs=epochs, verbose=True)
# print(np.shape(loss_per_epoch))

fig, ax = plt.subplots()
ax.plot(list(range(1, epochs+1)), loss_per_epoch)
ax.set_xlabel("epoch")
ax.set_ylabel(loss_name)

# sys.exit()

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

print(f"Acc train / test = {np.mean(acc_train):.2f} / {np.mean(acc_test):.2f}")
print("\ttest accuracy per outcome class:", np.round(acc_test, 2))

print([np.shape(w) for w in net.weights])

fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(16, 8))
ax = ax.ravel()

for i, w in enumerate(net.weights[0].T):
    ax[i].imshow(w.reshape(img_dims))
    ax[i].set_title(i)

fig.suptitle(f"lr={eta:.2e}, epoch={epochs:.1e}, scheduler={type(scheduler)}, dropout_proba={dropout_proba:.2f}\n"
             f"acc train / test = {np.mean(acc_train):.2f} / {np.mean(acc_test):.2f}")
fig.tight_layout()

fig.savefig(os.path.join(folder_save, savename))
plt.show()
