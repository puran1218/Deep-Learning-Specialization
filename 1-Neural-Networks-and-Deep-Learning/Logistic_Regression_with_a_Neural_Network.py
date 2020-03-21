import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage, misc
from matplotlib.pyplot import imread
from lr_utils import load_dataset

# Loading the dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Number of training examples:  = 209
m_train = train_set_x_orig[0]
# Number of test examples:  = 50
m_test = test_set_x_orig[0]
# Height/Width of each image:  = 64
num_px = train_set_x_orig[1]

# train_set_x_orig shape: (209, 64, 64, 3)
# train_set_y_orig shape: (1, 209)
# test_set_x_orig shape: (50, 64, 64, 3)
# test_set_y_orig shape: (1, 50)

# ------
# Pre-processing data
# ------

# Reshape images of shape (num_px, num_px, 3) in a numpy-array of shape (num_px * num_px ∗ 3, 1)
train_set_x_orig_flatten = train_set_x_orig.reshape(train_set_x_orig[:].shape[0], -1).T # (12288, 209)
test_set_x_orig_flatten = test_set_x_orig.reshape(test_set_x_orig[:].shape[0], -1).T # (12288, 50)

# Standardize the data
train_set_x = train_set_x_orig_flatten/255
test_set_x = test_set_x_orig_flatten/255

# ------
# Helper function for learning algorithm
# ------

# 1. Sigmoid function
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

# 2. Initializing parameters
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0

    return w, b

# 3. Forward and backward propagation
def propagate(w, b, X, Y):
    m = X.shape[1]

    # Forward propagation
    # Compute activation
    A = sigmoid(np.dot(w.T, X) + b)

    # Compute cost
    cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # Backward propagation
    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)

    cost = np.squeeze(cost)

    grads = {"dw": dw,
             "db": db}

    return grads, cost

# 4. Optimization, using gradient descent to update parameters
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the cost
        if i % 100 == 0:
            costs.append(cost)

        # Print cost every 100 iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))


    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

# 5. Predict
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    return Y_prediction

# ------
# Model
# ------

def model(X_train, Y_train, X_test, Y_test, num_iterations = 200, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve w and b from parameters
    w = parameters["w"]
    b = parameters["b"]

    # Predict test and train set examples
    Y_predict_test = predict(w, b, X_test)
    Y_predict_train = predict(w, b, X_train)

    # Print train/test errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_predict_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_predict_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_predict_test": Y_predict_test,
         "Y_predict_train": Y_predict_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

# Plot learning curve (with costs)
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
