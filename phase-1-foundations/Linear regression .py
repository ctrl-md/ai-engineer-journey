"""
Week 4 — Linear regression, from scratch.
Predicts a raw number (no activation) from a single input, trained via
gradient descent on squared error.
"""

def predict(x, w, b):
    return x * w + b

def compute_error(y, prediction):
    return y - prediction

def compute_gradients(error, x):
    grad_w = -2 * error * x
    grad_b = -2 * error
    return grad_w, grad_b

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
        grad_w, grad_b = compute_gradients(error, x)
        w, b = update_weights(w, grad_w, b, grad_b, learning_rate)
    return w, b


if __name__ == "__main__":
    final_w, final_b = train(x=4, y=90, starting_w=10, starting_b=40, learning_rate=0.01, epochs=1000)
    final_prediction = predict(4, final_w, final_b)
    print(f"final w: {final_w:.4f}")
    print(f"final b: {final_b:.4f}")
    print(f"final prediction: {final_prediction:.4f} (target: 90)")