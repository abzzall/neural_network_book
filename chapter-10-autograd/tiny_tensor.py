"""Chapter 10: a small tensor autograd engine that follows PyTorch's rules.

Supported: leaf tensors and requires_grad, grad_fn, gradient accumulation,
detach, no_grad, freeing saved tensors after backward (retain_graph),
backward with an explicit gradient, retain_grad, hooks, and version counters
that detect harmful in-place changes. Error categories follow PyTorch; messages are shorter.

Only real float64 arrays, elementwise addition/multiplication, two-dimensional
matrix multiplication, ReLU and full reduction are supported. No higher-order
derivatives, concurrency, view tracking, or general in-place operation history.
Mutate through add_ only: direct writes to data bypass version checks.
"""

import contextlib

import numpy as np

_grad_enabled = True


@contextlib.contextmanager
def no_grad():
    """Inside this block, operations are not recorded."""
    global _grad_enabled
    previous, _grad_enabled = _grad_enabled, False
    try:
        yield
    finally:
        _grad_enabled = previous


class Node:
    """One recorded operation: its inputs, the tensors it saved, and its backward rule."""

    def __init__(self, name, inputs, backward, saved):
        self.name, self.inputs, self.backward = name, inputs, backward
        self.saved = list(saved)
        self.saved_versions = [t._version for t in saved]

    def __repr__(self):
        return f"<{self.name}>"


class HookHandle:
    def __init__(self, hooks, fn):
        self.hooks, self.fn = hooks, fn

    def remove(self):
        if self.fn in self.hooks:
            self.hooks.remove(self.fn)


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad = None                    # filled only for leaves (or with retain_grad)
        self.grad_fn = None                 # the Node that produced this tensor, if any
        self._counter = [0]                 # shared with detached views of the same data
        self._retain = False
        self._hooks = []

    @property
    def is_leaf(self):
        return self.grad_fn is None

    @property
    def _version(self):
        return self._counter[0]

    def __repr__(self):
        fn = f", grad_fn={self.grad_fn}" if self.grad_fn else ""
        return f"Tensor({self.data.tolist()}{fn})"

    def detach(self):
        """Same data and version counter, cut off from the graph."""
        out = Tensor.__new__(Tensor)
        out.data, out.requires_grad, out.grad, out.grad_fn = self.data, False, None, None
        out._counter, out._retain, out._hooks = self._counter, False, []
        return out

    def retain_grad(self):
        if not self.requires_grad:
            raise RuntimeError("cannot retain_grad on a tensor with requires_grad=False")
        self._retain = True

    def register_hook(self, fn):
        if not self.requires_grad:
            raise RuntimeError("cannot register a hook on a tensor that does not require grad")
        self._hooks.append(fn)
        return HookHandle(self._hooks, fn)

    def add_(self, other):
        """In-place addition; it changes data that a recorded operation may have saved."""
        if self.is_leaf and self.requires_grad and _grad_enabled:
            raise RuntimeError("a leaf Variable that requires grad is being used in an in-place operation.")
        self.data += other.data if isinstance(other, Tensor) else other
        self._counter[0] += 1
        return self

    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __matmul__(self, other):
        return matmul(self, other)

    def sum(self):
        return reduce_sum(self)

    def backward(self, gradient=None, retain_graph=False):
        _backward(self, gradient, retain_graph)


def _as_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x)


def _result(data, name, inputs, backward, saved=()):
    """Record a Node only when grad mode is on and some input requires a gradient."""
    out = Tensor(data)
    if _grad_enabled and any(t.requires_grad for t in inputs):
        out.requires_grad = True
        out.grad_fn = Node(name, inputs, backward, saved)
    return out


def _unbroadcast(grad, shape):
    """Sum a gradient over the dimensions that broadcasting created."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def add(a, b):
    a, b = _as_tensor(a), _as_tensor(b)
    return _result(a.data + b.data, "AddBackward", (a, b),
                   lambda g: (_unbroadcast(g, a.data.shape), _unbroadcast(g, b.data.shape)))


def mul(a, b):
    a, b = _as_tensor(a), _as_tensor(b)
    return _result(a.data * b.data, "MulBackward", (a, b),
                   lambda g: (_unbroadcast(g * b.data, a.data.shape),
                              _unbroadcast(g * a.data, b.data.shape)), saved=tuple(t for t, needed in ((a, b.requires_grad), (b, a.requires_grad)) if needed))


def matmul(a, b):
    a, b = _as_tensor(a), _as_tensor(b)
    if a.data.ndim != 2 or b.data.ndim != 2:
        raise ValueError("matmul supports two-dimensional matrices only")
    return _result(a.data @ b.data, "MmBackward", (a, b),
                   lambda g: (g @ b.data.T, a.data.T @ g), saved=(a, b))


def relu(a):
    out = _result(np.maximum(a.data, 0.0), "ReluBackward", (a,),
                  lambda g: (g * (out.data > 0),))
    if out.grad_fn:
        out.grad_fn.saved = [out]
        out.grad_fn.saved_versions = [out._version]
    return out


def reduce_sum(a):
    return _result(a.data.sum(), "SumBackward", (a,), lambda g: (np.ones_like(a.data) * g,))


def _backward(root, gradient, retain_graph):
    if not root.requires_grad:
        raise RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")
    if gradient is None:
        if root.data.size != 1:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        gradient = np.ones_like(root.data)
    if np.shape(gradient) != root.data.shape:
        raise RuntimeError("Mismatch in shape between gradient and output")
    order, seen = [], set()                 # topological order, as for Value

    def visit(t):
        if id(t) not in seen:
            seen.add(id(t))
            for parent in (t.grad_fn.inputs if t.grad_fn else ()):
                visit(parent)
            order.append(t)
    visit(root)
    grads = {id(root): np.asarray(gradient, dtype=float)}
    for t in reversed(order):
        g = grads.pop(id(t), None)
        if g is None:
            continue
        for hook in t._hooks:               # a hook may replace the gradient
            replaced = hook(g)
            g = replaced if replaced is not None else g
        if t.is_leaf or t._retain:          # only these keep .grad, and they accumulate
            t.grad = g.copy() if t.grad is None else t.grad + g
        if t.is_leaf:
            continue
        node = t.grad_fn
        if node.saved is None:
            raise RuntimeError("Trying to backward through the graph a second time; "
                               "specify retain_graph=True the first time.")
        for s, version in zip(node.saved, node.saved_versions):
            if s._version != version:
                raise RuntimeError("one of the variables needed for gradient computation "
                                   "has been modified by an inplace operation")
        input_grads = node.backward(g)
        if not retain_graph and node.saved:
            node.saved = None               # free what was saved for this backward pass
        for inp, ig in zip(node.inputs, input_grads):
            if inp.requires_grad:
                grads[id(inp)] = grads.get(id(inp), 0) + ig
