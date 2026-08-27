import numpy as np

def get_arrays(*args):
    arrays = []

    for arg in args:
        arrays.append(np.array(arg))

    return arrays

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(grad_H, Z):
    return grad_H * (Z > 0)

def compute_error(Y, P):
    return Y - P

def forward(X, W1, B1, W2, B2, Y): 
    Z1 = X @ W1 + B1
    H = relu(Z1)
    Z2 = H @ W2 + B2
    loss = compute_error(Y, Z2)**2

    return Z1, H, Z2, loss

def backward(Y, Z2, H, W2, Z1, X):
    grad_Z2 = -2 * compute_error(Y, Z2)
    grad_W2 = H.T @ grad_Z2
    grad_B2 = np.sum(grad_Z2, axis=0, keepdims=True)
    grad_H = grad_Z2 @ W2.T
    grad_Z1 = relu_derivative(grad_H, Z1)
    grad_W1 = X.T @ grad_Z1
    grad_B1 = np.sum(grad_Z1, axis=0, keepdims=True)

    return grad_W1, grad_B1, grad_W2, grad_B2

def update_weights(W1, grad_W1, B1, grad_B1, W2, grad_W2, B2, grad_B2, learning_rate):
    W1 = W1 - (learning_rate * grad_W1)
    B1 = B1 - (learning_rate * grad_B1)
    W2 = W2 - (learning_rate * grad_W2)
    B2 = B2 - (learning_rate * grad_B2)

    return W1, B1, W2, B2

def train(X, starting_W1, starting_B1, starting_W2, starting_B2, Y, learning_rate, epochs):
    X, W1, B1, W2, B2, Y = get_arrays(X, starting_W1, starting_B1, starting_W2, starting_B2, Y)

    for _ in range(epochs):
        Z1, H, Z2, loss = forward(X, W1, B1, W2, B2, Y)
        grad_W1, grad_B1, grad_W2, grad_B2 = backward(Y, Z2, H, W2, Z1, X)
        W1, B1, W2, B2 = update_weights(W1, grad_W1, B1, grad_B1, W2, grad_W2, B2, grad_B2, learning_rate)

    return W1, B1, W2, B2

