"""

Code/junk-code to test

"""

import numpy as np
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import sys

import project_2_utils as p2utils

from project_2_utils import ConstantScheduler
from project_2_utils import MomentumScheduler
from project_2_utils import AdagradScheduler
from project_2_utils import RMS_propScheduler
from project_2_utils import AdamScheduler

fig_path = "figs/"

x = np.arange(0, 10, 0.1)
n = len(x)

X = p2utils.one_d_design_matrix(x, 2)

y = p2utils.simple_func(x, 1, 5, 3, noise_sigma=4.0)
y = y.reshape(-1, 1)

suggested_eta = p2utils.eta_from_hessian(X)
print("Suggested eta: ", suggested_eta)

print("Shape of x and y: ", x.shape, y.shape)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, 'o', label='Data points', alpha=0.5)

plt.savefig(fig_path + "data_points.png")

fig_iterations = plt.figure()
ax_iterations = fig_iterations.add_subplot(111)

scheduler = AdamScheduler(suggested_eta, 0.9, 0.999)

all_schedulers = [ConstantScheduler(suggested_eta),
                    MomentumScheduler(suggested_eta, 0.9),
                    AdagradScheduler(suggested_eta),
                    RMS_propScheduler(suggested_eta, 0.9),
                    AdamScheduler(suggested_eta, 0.9, 0.999)]

scheduler_names = ["Constant", "Momentum", "Adagrad", "RMS_prop", "Adam"]

for scheduler, name in zip(all_schedulers, scheduler_names):

    output = p2utils.general_gradient_descent(X, y, 
                                     np.random.randn(3,1), 
                                     scheduler=scheduler,
                                     max_iterations=100000,
                                     cost_func=p2utils.simple_cost_func,
                                     gradient_cost_func=p2utils.gradient_simple_function,
                                     return_diagnostics=True)
    
    ax_iterations.plot(output["iteration"], output["mse"], label=name)

    beta = output["beta"]
    
    ax.plot(x, X.dot(beta), label=name)


print("Beta from general function: ", beta)

ax.plot(x, X.dot(beta), label='Gradient descent')

ols_beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

ax.plot(x, X.dot(ols_beta), label='OLS')

plt.legend()

fig_iterations.savefig(fig_path + "gradient_descent_iterations.png")
fig.savefig(fig_path + "gradient_descent_all.png")


sys.exit()

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

# Try different eta-values

etas = [suggested_eta, 0.0001]

fig = plt.figure()
ax = fig.add_subplot(111)

for eta in etas:

    (beta_mini, scores) = p2utils.gradient_descent_with_minibatches(X, y, np.random.randn(3,1),
                                                eta,
                                                minibatch_size=10)

    ax.plot(scores, label=f"eta = {eta}")

ax.set_xlabel("Epochs")
ax.legend()

plt.savefig(fig_path + "minibatch_scores_eta.png")

print("Going into time decay")

(beta_time_decay, scores) = p2utils.gradient_descent_with_time_decay(X, y, np.random.randn(3,1),
                                                suggested_eta,
                                                minibatch_size=10)
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(scores, label=f"Time decay")

ax.set_xlabel("Epochs")

plt.savefig(fig_path + "time_decay.png")

print("Final results")
print(beta)
print(beta_linreg)
print(beta_mini)
print(beta_time_decay)




