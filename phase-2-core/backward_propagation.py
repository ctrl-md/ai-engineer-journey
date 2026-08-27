def relu(z):
    if z <= 0:
        return 0
    return z

def relu_derivative(z):
    if z <= 0:
        return 0
    return 1

def compute_error(y, prediction):
    return y - prediction

def forward(x, w1, b1, w2, b2, y):
    z1 = x * w1 + b1 
    h = relu(z1)
    z2 = h * w2 + b2
    loss = compute_error(y, z2)**2

    return z1, h, z2, loss

def backward(y, z2, h, w2, z1, x):
    error = compute_error(y, z2)
    grad_z2 = -2 * error
    grad_w2 = grad_z2 * h
    grad_b2 = grad_z2 * 1
    grad_h = grad_z2 * w2
    grad_z1 = grad_h * relu_derivative(z1)
    grad_w1 = grad_z1 * x
    grad_b1 = grad_z1 * 1

    return grad_w1, grad_b1, grad_w2, grad_b2

def update_weights(w1, grad_w1, b1, grad_b1, w2, grad_w2, b2, grad_b2, learning_rate):
    new_w1 = w1 - (learning_rate * grad_w1)
    new_b1 = b1 - (learning_rate * grad_b1)
    new_w2 = w2 - (learning_rate * grad_w2)
    new_b2 = b2 - (learning_rate * grad_b2)

    return new_w1, new_b1, new_w2, new_b2

def train(x, starting_w1, starting_b1, starting_w2, starting_b2, y, learning_rate, epochs):
    w1, b1, w2, b2 = starting_w1, starting_b1, starting_w2, starting_b2

    for _ in range(epochs):
        z1, h, z2, loss = forward(x, w1, b1, w2, b2, y)
        grad_w1, grad_b1, grad_w2, grad_b2 = backward(y, z2, h, w2, z1, x)
        w1, b1, w2, b2 = update_weights(w1, grad_w1, b1, grad_b1, w2, grad_w2, b2, grad_b2, learning_rate)

    return w1, b1, w2, b2
