"""Chapter 10: a scalar automatic-differentiation engine in plain Python."""

import math


class Value:
    """A number that remembers how it was computed, so that gradients can flow back."""

    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0                     # dL/d(this value), filled by backward()
        self.parents = parents              # the Values this one was computed from
        self.op = op                        # the operation, for printing only
        self._backward = lambda: None       # sends self.grad on to the parents

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():                    # d(a+b)/da = d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():                    # d(ab)/da = b, d(ab)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):                   # k is an ordinary number
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """Visit the graph in topological order, then apply the chain rule from the output back."""
        order, seen = [], set()

        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for parent in v.parents:
                    visit(parent)
                order.append(v)             # every value after everything it depends on
        visit(self)
        for v in order:                     # intermediate results start from zero;
            if v.parents:                   # only leaves (inputs, weights) accumulate
                v.grad = 0.0
        if self.parents:
            self.grad = 1.0                 # seed an operation output
        else:
            self.grad += 1.0                # a leaf output also accumulates
        for v in reversed(order):
            v._backward()

    # The remaining operations are built from the ones above.
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return Value(other) - self
