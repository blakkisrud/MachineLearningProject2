"""

Code/junk-code to test

"""

import numpy as np
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

import project_2_utils as p2utils

fig_path = "figs/"

x = np.arange(0, 10, 0.1)
n = len(x)


X = p2utils.one_d_design_matrix(x, 2)


y = p2utils.simple_func(x, 1, 5, 3, noise_sigma=2.0)
y = y.reshape(-1, 1)

suggested_eta = p2utils.eta_from_hessian(X)
print("Suggested eta: ", suggested_eta)

print("Shape of x and y: ", x.shape, y.shape)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, 'o', label='Data points', alpha=0.5)

plt.savefig(fig_path + "data_points.png")

# Gradient descent from function

beta_from_function = p2utils.gradient_descent(X, y, np.random.randn(3,1), 
                                              suggested_eta)

print("Beta from function: ", beta_from_function)

# Below the comparison

beta_linreg = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

ax.plot(x, X.dot(beta_from_function), label='Gradient descent')
ax.plot(x, X.dot(beta_linreg), label='OLS')

plt.legend()

plt.savefig(fig_path + "gradient_descent.png")

print("Final beta values: ", beta_from_function)
print("OLS values: ", beta_linreg)

print("With momentum")

(beta, beta_list, scores) = p2utils.gradient_descent_with_momentum(X, y, np.random.randn(3,1), 
                                              suggested_eta,
                                              gamma=0.9)

print("Going into minibatch")

(beta_mini, scores) = p2utils.gradient_descent_with_minibatches(X, y, np.random.randn(3,1),
                                                suggested_eta,
                                                minibatch_size=10)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, 'o', label='Data points', alpha=0.5)
ax.plot(x, X.dot(beta), label='Momentum')
ax.plot(x, X.dot(beta_mini), label='Minibatch')
ax.plot(x, X.dot(beta_linreg), label='OLS')

plt.legend()

plt.savefig(fig_path + "momentum_minibatch.png")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(scores, label='Minibatch')
ax.set_xlabel("Epochs")

plt.savefig(fig_path + "minibatch_scores.png")

print("Final results")
print(beta)
print(beta_linreg)
print(beta_mini)




