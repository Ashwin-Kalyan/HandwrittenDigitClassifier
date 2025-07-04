import pandas as pd 
import numpy as np
import time
import math
from matplotlib import pyplot as plt

# Load and prepare data
data = pd.read_csv(r"C:\Users\ashwi\OneDrive\Desktop\GitHub\HandwrittenDigitClassifier\HandwrittenDigitClassifier\train.csv")
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# Split into dev and training sets
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n] / 255.0

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.0
_, m_train = X_train.shape

# Optimized initialization with bigger hidden layer
def init_params():
    W1 = np.random.randn(64, 784) * np.sqrt(2.0/848.0)  # Increased neurons, Xavier initialization
    B1 = np.zeros((64, 1))
    W2 = np.random.randn(10, 64) * np.sqrt(2.0/74.0)   # Adjusted to match
    B2 = np.zeros((10, 1))
    return W1, B1, W2, B2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z))
    return exp_Z / exp_Z.sum(axis=0)

def forward_propagation(W1, B1, W2, B2, X):
    Z1 = W1 @ X + B1  # Using @ for matrix multiplication
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + B2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def derivative_ReLU(Z):
    return Z > 0

# Optimized backpropagation with vectorized operations
def back_propagation(Z1, A1, Z2, A2, W1, W2, X, Y, lamda_reg):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (dZ2 @ A1.T) / m + ((lamda_reg / m) * W2)
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = W2.T @ dZ2 * derivative_ReLU(Z1)
    dW1 = (dZ1 @ X.T) / m + ((lamda_reg / m) * W1)
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return dW1, db1, dW2, db2

def update_params(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha):
    W1 -= alpha * dW1
    B1 -= alpha * dB1
    W2 -= alpha * dW2
    B2 -= alpha * dB2
    return W1, B1, W2, B2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

# Memory-efficient batch generator
def batch_generator(X, Y, batch_size):
    m = X.shape[1]
    for i in range(0, m, batch_size):
        yield X[:, i:i+batch_size], Y[i:i+batch_size]

# Main training function with all optimizations
def gradient_descent(X, Y, iterations, batch_size, initial_alpha):
    W1, b1, W2, b2 = init_params()
    m = X.shape[1]
    start_time = time.time()
    
    # Early stopping variables
    best_dev_acc = 0
    no_improve = 0
    patience = 10
    min_improvement = 0.0001
    
    for i in range(iterations):
        # Shuffle data at epoch boundaries
        if i % (m // batch_size) == 0:
            perm = np.random.permutation(m)
            X_shuffled = X[:, perm]
            Y_shuffled = Y[perm]
        
        # Learning rate decay
        n_min = 0.01
        t = i
        T = iterations
        alpha = n_min + 0.5 * (initial_alpha - n_min) * (1 + math.cos((math.pi * t) / T))
        
        # Process batches
        for X_batch, Y_batch in batch_generator(X_shuffled, Y_shuffled, batch_size):
            Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X_batch)
            dW1, db1, dW2, db2 = back_propagation(Z1, A1, Z2, A2, W1, W2, X_batch, Y_batch, 0.01)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        # Progress monitoring
        if i % 20 == 0:
            train_pred = get_predictions(A2)
            train_acc = get_accuracy(train_pred, Y_batch)
            
            _, _, _, A2_dev = forward_propagation(W1, b1, W2, b2, X_dev)
            dev_pred = get_predictions(A2_dev)
            dev_acc = get_accuracy(dev_pred, Y_dev)
            
            elapsed = time.time() - start_time
            remaining = (iterations-i) * (elapsed/(i+1))
            
            print(f"Iter {i}: {elapsed:.1f}s | ~{remaining:.1f}s remaining | "
                  f"Train: {train_acc:.3f} | Dev: {dev_acc:.3f} | LR: {alpha:.5f}")
            
            # Early stopping check
            if dev_acc > best_dev_acc + min_improvement:
                best_dev_acc = dev_acc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\nEarly stopping at iteration {i}")
                    print(f"Best dev accuracy: {best_dev_acc:.4f}\n")
                    return W1, b1, W2, b2
    
    return W1, b1, W2, b2

# Start training with optimized parameters
print("Starting training...")
W1, B1, W2, B2 = gradient_descent(X_train, Y_train, 1000, 64, 0.05)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

for i in range(10):
    test_prediction(i, W1, B1, W2, B2)