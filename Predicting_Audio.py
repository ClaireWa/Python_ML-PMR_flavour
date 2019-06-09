"""
Script for MLPR Assignment 1 - Predicting Audio
"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

# Set random state
seed = 17
np.random.seed(seed)

# Load data
data_path = '/afs/inf.ed.ac.uk/group/teaching/mlprdata/audio/amp_data.mat'
amp_data = loadmat(data_path)['amp_data']  # Load data

# Plot line graph of amp data
plt.plot(amp_data[::30], linewidth=0.1)  # subsample if there's any trouble plotting but not too much
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# Plot histogram of amplitudes
plt.hist(amp_data, bins=1000, linewidth=0)
plt.xlabel('Amplitude')
plt.show()

# Create dataset of consecutive amplitudes
n_data = amp_data.size
chunk_width = 21
n_chunks = n_data // chunk_width  # Floor divison
X = np.reshape(amp_data[:chunk_width*n_chunks], (n_chunks, chunk_width))

# Split into training, validation and test sets
X_shuf = np.random.permutation(X)
train_prop, val_prop = 0.7, 0.15
train_prop_ind = int(np.floor(train_prop*n_chunks))
val_prop_ind = train_prop_ind + int(np.floor(val_prop*n_chunks))
X_shuf_train = X_shuf[:train_prop_ind, :-1]
y_shuf_train = X_shuf[:train_prop_ind, -1]
X_shuf_val = X_shuf[train_prop_ind:val_prop_ind, :-1]
y_shuf_val = X_shuf[train_prop_ind:val_prop_ind, -1]
X_shuf_test = X_shuf[val_prop_ind:, :-1]
y_shuf_test = X_shuf[val_prop_ind:, -1]

# Select a training row
row_id = np.random.randint(X_shuf_train.shape[0])
x_row = X_shuf_train[row_id]
y_row = y_shuf_train[row_id]
tt = np.arange(0, 1, 1/20)

# Plot training row
plt.scatter(tt, x_row, label='Row amplitudes')
plt.scatter(1, y_row, color='r', label='Target amplitude')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend(fontsize='small')
plt.show()

# Fit a straight line
t_linear = np.array([tt**0, tt**1]).T
params = np.linalg.lstsq(t_linear, x_row)[0]

# Plot fitted line with points
t_plot = np.arange(0, 1+1/20, 1/20)
t_plot_linear = np.vstack([t_linear, np.ones(2)])
x_pred = t_plot_linear.dot(params)
plt.scatter(tt, x_row, label='Row amplitudes')
plt.scatter(1, y_row, color='r', label='Target amplitude')
plt.plot(t_plot, x_pred, 'g--', label='Fitted line')
plt.scatter(1, x_pred[-1], color='g', label='Target prediction')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend(fontsize='small')
plt.show()

# Fit quartic polynomial
t_quartic = np.array([tt**0, tt**1, tt**2, tt**3, tt**4]).T
params = np.linalg.lstsq(t_quartic, x_row)[0]

# Plot fitted quartic with points
t_plot_quartic = np.vstack([t_quartic, np.ones(5)])
x_pred = t_plot_quartic.dot(params)
plt.scatter(tt, x_row, label='Row amplitudes')
plt.scatter(1, y_row, color='r', label='Target amplitude')
plt.plot(t_plot, x_pred, 'g--', label='Fitted line')
plt.scatter(1, x_pred[-1], color='g', label='Target prediction')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend(fontsize='small')
plt.show()


# Construct design matrix
def construct_design_mat(context, degree):
    tt = np.arange(0, 1, 1/20)
    Phi = np.tile(tt[-context:][:, np.newaxis], (1, degree))
    for col in range(degree):
        Phi[:, col] = Phi[:, col]**col
    return Phi

# Create v predictor
C = 20
K = 5
Phi = construct_design_mat(C, K)
phi_1 = np.ones([K, 1])
V = Phi.dot(np.linalg.lstsq(Phi.T.dot(Phi), phi_1)[0])

# Check predictions
y_pred = V.T.dot(x_row)
if np.isclose(y_pred, x_pred[-1]):
    print('Predictions match')

# Optimize over context lengths and polynomial degree
max_C = 20
max_K = 20
train_errs = np.zeros([max_C, max_K])
for C in range(1, max_C + 1):
    X_shuf_train_context = X_shuf_train[:, -C:]
    for K in range(1, max_K + 1):
        Phi = construct_design_mat(C, K)
        phi_1 = np.ones([K, 1])
        V = Phi.dot(np.linalg.lstsq(Phi.T.dot(Phi), phi_1)[0])
        y_pred = X_shuf_train_context.dot(V)[:, 0]
        train_errs[C-1, K-1] = np.mean((y_shuf_train - y_pred)**2)
min_val = np.min(train_errs)
res = np.argwhere(train_errs == min_val)[0] + 1
print('Training error minimised for C = {} and K = {}'.format(res[0], res[1]))
print('Mean squared error = {}'.format(min_val))

# Evaluate on validation and test sets
best_C, best_K = res[0], res[1]
X_shuf_val_context = X_shuf_val[:, -best_C:]
X_shuf_test_context = X_shuf_test[:, -best_C:]
Phi = construct_design_mat(best_C, best_K)  # Get design matrix
phi_1 = np.ones([best_K, 1])
V = Phi.dot(np.linalg.solve(Phi.T.dot(Phi), phi_1))
y_pred_val = X_shuf_val_context.dot(V)[:, 0]
y_pred_test = X_shuf_test_context.dot(V)[:, 0]
val_err = np.mean((y_shuf_val - y_pred_val)**2)
test_err_poly = np.mean((y_shuf_test - y_pred_test)**2)
print('Validation Mean squared error = {}'.format(val_err))
print('Test Mean squared error = {}'.format(test_err_poly))

# Fit general vectors v for each context length
max_C = 20
train_errs = np.zeros(max_C)
val_errs = np.zeros(max_C)
for C in range(1, max_C + 1):
    X_shuf_train_context = X_shuf_train[:, -C:]
    X_shuf_val_context = X_shuf_val[:, -C:]
    V = np.linalg.lstsq(X_shuf_train_context, y_shuf_train)[0][:, np.newaxis]
    y_pred_train = X_shuf_train_context.dot(V)[:, 0]
    y_pred_val = X_shuf_val_context.dot(V)[:, 0]
    train_errs[C - 1] = np.mean((y_shuf_train - y_pred_train)**2)
    val_errs[C - 1] = np.mean((y_shuf_val - y_pred_val)**2)
best_C_train = np.argmin(train_errs) + 1
best_C_val = np.argmin(val_errs) + 1
print('Best context length on training set is: {}'.format(best_C_train))
print('Mean squared error = {}'.format(np.min(train_errs)))
print('Best context length on validation set is: {}'.format(best_C_val))
print('Mean squared error = {}'.format(np.min(val_errs)))

# Plot training and validation errors against context length
plt.plot(range(1, max_C + 1), train_errs)
plt.plot(range(1, max_C + 1), val_errs)
plt.xlabel('Context length')
plt.ylabel('Mean squared error')
plt.legend(['Train', 'Validation'])
plt.show()

# Evaluate test performance against best polynomial model
X_shuf_train_context = X_shuf_train[:, -best_C_val:]
X_shuf_test_context = X_shuf_test[:, -best_C_val:]
X_shuf_val_context = X_shuf_val[:, -best_C_val:]
V = np.linalg.lstsq(X_shuf_train_context, y_shuf_train)[0][:, np.newaxis]
y_pred_test = X_shuf_test_context.dot(V)[:, 0]
test_err_lin = np.mean((y_shuf_val - y_pred_val)**2)
print('test error for best polynomial: {}'.format(test_err_poly))
print('test error for best linear predictor: {}'.format(test_err_lin))

# Plot histogram of residuals on the validation data
y_pred_val = X_shuf_val_context.dot(V)[:, 0]
plt.figure(figsize=(8, 3))
plt.subplot(2, 1, 1)
plt.hist(amp_data, bins=1000, linewidth=0, alpha=0.5, color='r')
plt.xlim([-0.2, 0.2]);
plt.xlabel('Amplitude')
plt.subplot(2, 1, 2)
plt.hist(y_pred_val - y_shuf_val, bins=1000, linewidth=0, alpha=0.5)
plt.xlim([-0.2, 0.2]);
plt.xlabel('Residual')
plt.show()
