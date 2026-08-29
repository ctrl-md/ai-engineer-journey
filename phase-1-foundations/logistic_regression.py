"""
Week 4 — Logistic regression, from scratch.
Predicts a probability (via sigmoid) from a single input, trained via
gradient descent on cross-entropy loss.
"""

from math import exp


def sigmoid(z):
    return 1 / (1 + exp(-z))


def predict(x, w, b):
    return sigmoid(x * w + b)


def compute_error(y, prediction):
    return y - prediction


def compute_gradients(x, error):
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
        w_grad, b_grad = compute_gradients(x, error)
        w, b = update_weights(w, w_grad, b, b_grad, learning_rate)
    return w, b


if __name__ == "__main__":
    # actual answer is "yes" (y=1)
    w1, b1 = train(
        x=3, y=1, starting_w=0.5, starting_b=0.5, learning_rate=0.1, epochs=1000
    )
    print(f"[y=1 case] predicted probability: {predict(3, w1, b1):.6f}")

    # actual answer is "no" (y=0)
    w2, b2 = train(
        x=3, y=0, starting_w=0.5, starting_b=0.5, learning_rate=0.1, epochs=1000
    )
    print(f"[y=0 case] predicted probability: {predict(3, w2, b2):.6f}")
