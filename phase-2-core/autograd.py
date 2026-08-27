"""
Week 6 — A tiny autograd engine from scratch.
A Value class that tracks how every number was created, so gradients for
any combination of operations can be computed automatically via a single
generic backward() pass, instead of hand-deriving a new backward function
for every new network shape.
"""

def build_topo(v, visited, topo):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child, visited, topo)
        topo.append(v)

def relu(v):
    if v <= 0:
        return 0
    return v

def relu_derivative(v):
    if v <= 0:
        return 0
    return 1


class Value():
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        out = Value(-self.data, (self,))
        def _backward():
            self.grad += -1 * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def relu(self):
        out = Value(relu(self.data), (self,))
        def _backward():
            self.grad += relu_derivative(self.data) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        visited = set()
        topo = []
        build_topo(self, visited, topo)
        self.grad = 1
        for item in reversed(topo):
            item._backward()


if __name__ == "__main__":
    # same tiny network as week 5, but trained using ONLY the autograd
    # engine's operators -- no hand-written backward logic anywhere
    x = Value(4)
    y = Value(1)
    w1, b1, w2, b2 = Value(0.5), Value(1), Value(-1), Value(3)
    learning_rate = 0.01

    for _ in range(2000):
        z1 = x * w1 + b1
        h = z1.relu()
        z2 = h * w2 + b2
        error = y - z2
        loss = error * error

        w1.grad = 0
        b1.grad = 0
        w2.grad = 0
        b2.grad = 0
        loss.backward()

        w1.data -= learning_rate * w1.grad
        b1.data -= learning_rate * b1.grad
        w2.data -= learning_rate * w2.grad
        b2.data -= learning_rate * b2.grad

    print(f"final prediction: {z2.data:.4f} (target: 1)")
    print(f"final loss: {loss.data:.6f}")