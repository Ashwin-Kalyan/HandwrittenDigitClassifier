# This will be a handwritten digit classifier using a 
# Feedforward neural network built from scratch using only NumPy and mathematics.
# The dataset consists of 28x28 images, each pixel being a number between 0 (white) and 255 (black).
# Thus, each image can be represented as a matrix with 784 numbers between 0-255 to represent each pixel value.
# We will have one matrix, with m (m being the number of images in the dataset) columns, 
# each column representing one image in the dataset. We will call this matrix A. 
# Within each column, we get another matrix with 784 rows to represent each pixel in the image it represents.
# We will have 10 classes of digits, 0-9, that the neural network will try to classify the image data into.
# The first layer of the neural network--the input layer--will consist of 784 nodes for each pixel of the given image.
# The second layer is a hidden layer with 10 nodes, and, finally, 
# the output layer is going to consist of 10 nodes representing the digits.
# FORWARD PROPAGATION--a recognition-inference architechture--in the first layer will be done by 
# taking the dot product of a new 10x784 WEIGHT (W) matrix 
# by the A matrix to make a 10xm Z matrix. The values within the W matrix will represent the 
# 7840 connections between the input layer nodes and hidden layer. 
# The Z matrix will also have a BIAS (B) matrix added to the dot product of A and W 
# to indicate patterns beyond those that pass through the origin (think y = mx+b!).
# Next we need an ACTIVATION FUNCTION to take the weighted sum of inputs and apply
# a transformation to produce an output. We will use a ReLU--rectified linear unit. 
# ReLU outputs the input directly if it's positive, or 0 otherwise. ReLU is advantageous because it
# avoids saturation, alleviating the VANISHING GRADIENT PROBLEM--
# when the gradient of the error/loss function become so small that it cannot be used to adjust the weights and bias. 
# ReLU applied to the Z will comprise the hidden layer. The output layer will be equal to a second W matrix, 
# comprised of values representing the connection between the hidden layer 
# and output layer's nodes, dot producted by the hidden layer matrix--ReLU(Z)--added to another bias vector. 
# The final output layer matrix will have the SOFTMAX function--which takes a tuple of K real numbers 
# and turns them into a probability distribution of K possible outcomes
# --as its activation function to take the output matrix and produce probabilities from it, describing how likely each
# final value is to be a class of digit. We will use BACKPROPAGATION to find the error, 
# and see how the previous weights and biases can be adjusted to minimize the error. 
# This will allow the model to "learn" which images correspond to which digits. 
# Once it adjusts, it will be passed back into the forward propagation, 
# then back into backpropagation to minimize the error further. Repeat until fully accurate!

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("train.csv")
data = np.array(data)

m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.0

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.0
_, m_train = X_train.shape

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    B1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    B2 = np.random.rand(10, 1) - 0.5
    return W1, B1, W2, B2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z))  # Improve numerical stability
    return exp_Z / exp_Z.sum(axis=0, keepdims=True)
 
def forward_propagation(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def derivative_ReLU(Z):
    return Z > 0

def back_propagation(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * derivative_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha):
    W1 = W1 - alpha * dW1
    B1 = B1 - alpha * dB1
    W2 = W2 - alpha * dW2
    B2 = B2 - alpha * dB2
    return W1, B1, W2, B2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return (np.sum(predictions == Y) / Y.size)

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_propagation(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

W1, B1, W2, B2 = gradient_descent(X_train, Y_train, 5000, 0.01)