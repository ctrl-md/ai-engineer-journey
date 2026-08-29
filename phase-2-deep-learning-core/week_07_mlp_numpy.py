"""
Week 7 — A multi-layer perceptron from scratch, in NumPy.
Same two-layer network as week 5, but matrix-based: trains on a whole
batch of examples at once via matrix multiplication, instead of one
number at a time.
"""

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
    loss = compute_error(Y, Z2) ** 2
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


def train(
    X, starting_W1, starting_B1, starting_W2, starting_B2, Y, learning_rate, epochs
):
    X, W1, B1, W2, B2, Y = get_arrays(
        X, starting_W1, starting_B1, starting_W2, starting_B2, Y
    )
    for _ in range(epochs):
        Z1, H, Z2, loss = forward(X, W1, B1, W2, B2, Y)
        grad_W1, grad_B1, grad_W2, grad_B2 = backward(Y, Z2, H, W2, Z1, X)
        W1, B1, W2, B2 = update_weights(
            W1, grad_W1, B1, grad_B1, W2, grad_W2, B2, grad_B2, learning_rate
        )
    return W1, B1, W2, B2


if __name__ == "__main__":
    # 3 examples, 2 features each, 4 hidden neurons, 1 output neuron
    X = [[1, 2], [3, -1], [0, 5]]
    W1 = [[1, 0, -1, 2], [2, 1, 1, -1]]
    B1 = [0, 1, -1, 0]
    W2 = [[1], [2], [1], [-1]]
    B2 = [0]
    Y = [[10], [5], [20]]

    W1f, B1f, W2f, B2f = train(X, W1, B1, W2, B2, Y, learning_rate=0.001, epochs=5000)
    Xn, _, _, _, _, Yn = get_arrays(X, W1, B1, W2, B2, Y)
    Z1, H, Z2, loss = forward(Xn, W1f, B1f, W2f, B2f, Yn)

    print("final predictions:\n", Z2)
    print("actual Y:\n", Yn)
    print("final loss per example:\n", loss)
