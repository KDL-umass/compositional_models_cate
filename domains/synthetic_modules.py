from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import torch

# script to generate synthetic data for the causal effect estimation experiments
coeffs = {
 '1': {0: [ 0.21667   , -0.63903564,  9.00922865],
  1: [0.12936905, 0.2799703 , 6.57174534]},
 '2': {0: [ 0.30264717, -1.25032706,  9.94655612],
  1: [0.16302046, 1.00160131, 4.64935063]},
 '3': {0: [0.01939968, 0.02394197, 9.47095968],
  1: [0.02878614, 0.03671678, 8.68403881]},
 '4': {0: [ 0.22816887, -0.77317096,  9.38126938],
  1: [0.14832781, 0.1123053 , 6.89397014]},
 '5': {0: [ 0.21146381, -0.56553982,  8.49733234],
  1: [ 0.29130791, -0.41326397,  8.49733234]},
 '6': {0: [0.13604885, 0.41398028, 9.26971561],
  1: [ 0.27753554, -0.14679641,  9.01205926]},
 '7': {0: [ 0.21259568, -0.43276121, 10.05956736],
  1: [0.20716442, 0.60688892, 6.5829069 ]},
 '8': {0: [ 0.19895964, -0.46852413, 10.8638669 ],
  1: [ 0.33923464, -1.1632891 , 11.31139315]},
 '9': {0: [ 0.03409934, -0.05973316,  8.11280246],
  1: [-0.03727203,  0.68611027,  5.90218791]},
 '10': {0: [0.04776045, 1.34701657, 8.06038158],
  1: [0.16713525, 0.98506548, 7.36645402]}
}

def f1_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    return np.dot(X, w[:Mj]) + w[-1]

def f2_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    return np.dot(X**2, w[:Mj]) + np.dot(X, w[Mj:2*Mj]) + w[-1]

def f3_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    return w[0] * (np.sin(np.pi * np.dot(X, w[1:Mj+1])) / 2 + 0.5) + w[-1]

def f4_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    return w[0] / (1 + np.exp(-np.dot(X, w[1:Mj+1]))) + w[-1]

def f5_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    return w[0] * np.sqrt(np.dot(X, w[1:Mj+1])) + w[-1]

def f6_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    return w[0] * (1 - np.dot(X**3, w[1:Mj+1])) + w[-1]

def f7_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    return w[0] * (0.5 * np.cos(2 * np.pi * np.dot(X, w[1:Mj+1])) + 0.5) + w[-1]

def f8_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)

    # have non-exponential function
    return np.polyval(w, inputs)[0]

def f9_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    return w[0] * np.log(np.dot(X, w[1:Mj+1]) + 1) / np.log(2) + w[-1]

def f10_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    return w[0] * (0.5 * np.tanh(np.dot(X, w[1:Mj+1]) - 2) + 0.5) + w[-1]

def polyval_module(*inputs, w):
    return np.polyval(w, inputs)[0]

def cubic_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    w = np.array(w)
    y = np.dot(X**3, w[:Mj]) + np.dot(X**2, w[Mj:2*Mj]) + np.dot(X, w[2*Mj:3*Mj]) + w[-1]
    return y

def quadratic_module(*inputs, w):
    # take all the inputs
    X = np.array(inputs)
    Mj = len(X)
    w = np.array(w)
    
    # quadratic outcome function
    x_squared = X**2
    y = np.dot(X**2, w[:Mj]) 
    y += np.dot(X, w[Mj:2*Mj])
    y += w[-1]
    return y

def linear_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    # linear outcome function
    y = np.dot(X, w[:Mj]) + w[-1]
    return y

def mlp_module(*inputs, model):
    X = np.array(inputs)
    X = X.reshape(1, -1)
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(X, dtype=torch.float32))
    return output.item()

def logarithmic_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    # take log of absolute value of X + 1
    X_log = np.log(np.abs(X) + 1)
    w = np.array(w)
    y = np.dot(X_log, w[:Mj]) + w[-1]
    # exponential function
    y = np.exp(y)
    return y

def sigmoid_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    y = 1 / (1 + np.exp(-np.dot(X, w[:Mj]) - w[-1]))
    return y

def exponential_module(*inputs, w):
    X = np.array(inputs)
    Mj = len(X)
    y = np.exp(np.dot(X, w[:Mj])) + w[-1]
    return y

def polynomial_module(*inputs, w):
    X = np.array(inputs)  # Convert inputs to array
    degree = len(w) - 1  # Number of coefficients minus the intercept
    return np.polyval(w, X[0])  # Use the first input element with the polynomial coefficients

def affine_module(*inputs, w):
    X = np.array(inputs)  # Convert inputs to array
    Mj = len(X)  # Number of input features
    matrix = np.array(w[:Mj])  # Use first Mj weights as matrix coefficients
    vector = np.array(w[-1])  # Last weight as the intercept
    return np.dot(matrix, X) + vector
