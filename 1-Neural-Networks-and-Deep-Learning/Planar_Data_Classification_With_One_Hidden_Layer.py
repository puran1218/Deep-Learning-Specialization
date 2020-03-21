import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1) # set a seed so that the results are consistent

# Load data via helper function planar_utils.py
X, Y = load_planar_dataset()

# Define the neural network structure
def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)

# Initialize the neural network parameters
def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2= np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters


# Gradient Descent method

# 1. Forward propagation
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Implement forward propagation
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache

# Compute cost
def compute_cost(A2, Y, parameters):
    logprobs = np.multiply(Y, np.log(A2))
    cost = -np.sum(logprobs)
    print(cost)

    cost = float(np.squeeze(cost))

    return cost

# Implement the function backward_propagation()
def back_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A2.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T * dZ2 * (1 - np.power(A1, 2))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

# Update parameters
def update_parameters(parameters, grads, learning_rate = 1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation.
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function.
        cost = compute_cost(A2, Y, parameters)
 
        # Backpropagation.
        grads = back_propagation(parameters, cache, X, Y)
 
        # Gradient descent parameter update.
        parameters = update_parameters(parameters, grads)
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def predict(parameters, X):
    # Computes probabilities using forward propagation
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2) #  Classifies to 0/1 using 0.5 as the threshold
    
    return predictions  

# size = layer_sizes(X, Y)
# print(size)

# para = initialize_parameters(size[0], size[1], size[2])
# print(para)

# forward = forward_propagation(X, para)
# print(forward[0].shape)

# cost = compute_cost(forward[0], Y, para)
# print(cost)

# back = back_propagation(para, forward[1], X, Y)
# print(back)

# update = update_parameters(para, back)
# print(update)

# print(X.shape)
# print(Y.shape)

parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

predictions = predict(parameters, X)
# print ('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
print (np.dot(Y, predictions.T))
print (np.dot(1-Y,1-predictions.T))