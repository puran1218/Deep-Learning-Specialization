import numpy as np

def sigmoid(x):
    # Compute sigmoid of x
    # Return s = sigmoid(x)

    s = 1 / (1 + np.exp(-x))

    return s

def sigmoid_derivative(x):
    # Compute the gradient of the sigmoid function with respect to its input x
    # Return computed gradient ds = s(1-s)

    s = sigmoid(x)
    ds = s * (1 - s)

    return ds

def image2vector(image):
    # takes an input of shape (length, height, 3)
    # returns a vector of shape (length*height*3, 1)

    # Image -- a numpy array of shape (length, height, depth)
    # Return v, a vector of shape (length * height * depth, 1)

    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)

    return v

def normalizeRows(x):
    # Implement a function that normalizes each row of the matrix x

    x_norm = np.linalg.norm(x, ord=2, axis=0, keepdims=True)
    x = x / x_norm

    print(x_norm)

    return x

x = np.array([1, 2, 3])
s = sigmoid(x)
ds = sigmoid_derivative(x)

# image is a 3 by 3 by 2 array
image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

v = image2vector(image)

print(s) 
print(ds)
print("\n image shape is: ", image.shape)
# print(v)

print("normalizeRows(x) = " + str(normalizeRows(x)))