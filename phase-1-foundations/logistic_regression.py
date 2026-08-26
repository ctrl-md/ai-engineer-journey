from math import exp

def sigmoid(z):
    return 1/(1 + exp(-z))

def predict(x, w, b):
    return sigmoid(x * w + b)

def compute_error(y, prediction):
    return y - prediction

def compute_gradients(error, x):
    w_grad = -error * x
    b_grad = -error
    return w_grad, b_grad

def update_weights(w, w_grad, b, b_grad, learning_rate):
    new_w = w - (learning_rate * w_grad)
    new_b = b - (learning_rate * b_grad)
    return new_w, new_b

def train(x, y, starting_w, starting_b, learning_rate, epochs):
    w = starting_w
    b = starting_b

    for _ in range(epochs):
        y_pred = predict(x, w, b)
        error = compute_error(y, y_pred)
        w_grad, b_grad = compute_gradients(error, x)
        w, b = update_weights(w, w_grad, b, b_grad, learning_rate)

    return w, b