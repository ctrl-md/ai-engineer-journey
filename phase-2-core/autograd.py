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

    def backward(self):
        visited = set()
        topo = []
        build_topo(self, visited, topo)
        self.grad = 1

        for item in reversed(topo):
            item._backward()

    def __neg__(self):
        out = Value(-self.data, (self,))

        def _backward():
            self.grad += -1 * out.grad

        out._backward = _backward

        return out

    def __sub__(self, other):
        out = self + (-other)

        return out

    def relu(self):
        out = Value(relu(self.data), (self,))

        def _backward():
            self.grad += relu_derivative(self.data) * out.grad

        out._backward = _backward

        return out
        