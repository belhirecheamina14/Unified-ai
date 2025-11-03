"""
Enhanced Autodiff Engine with Float64 Support
"""

import numpy as np
from typing import Union, Tuple, List, Optional
from abc import ABC, abstractmethod

class Node:
    """Enhanced Node with float64 support"""

    def __init__(self, data: np.ndarray, requires_grad: bool = True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set()

    def backward(self, grad: Optional[np.ndarray] = None):
        """Backpropagation with gradient clipping"""
        if grad is None:
            grad = np.ones_like(self.data, dtype=np.float64)

        self.grad = grad if self.grad is None else self.grad + grad

        # Gradient clipping
        max_norm = 1.0
        norm = np.linalg.norm(self.grad)
        if norm > max_norm:
            self.grad = self.grad * (max_norm / norm)

        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for prev in node._prev:
                    build_topo(prev)
                topo.append(node)

        build_topo(self)

        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Node(shape={self.data.shape}, dtype={self.data.dtype})"

class Parameter(Node):
    """Trainable parameter"""

    def __init__(self, data: np.ndarray):
        super().__init__(data, requires_grad=True)

class Module(ABC):
    """Base class for all neural network modules"""

    def __init__(self):
        self._parameters = {}
        self._modules = {}

    @abstractmethod
    def forward(self, x: Node) -> Node:
        pass

    def __call__(self, x: Node) -> Node:
        return self.forward(x)

    def parameters(self) -> List[Parameter]:
        """Returns all parameters"""
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def zero_grad(self):
        """Zero all gradients"""
        for p in self.parameters():
            p.grad = None

# Improved operations
def add(a: Node, b: Node) -> Node:
    out = Node(a.data + b.data)

    def _backward():
        if a.requires_grad:
            a.grad = out.grad if a.grad is None else a.grad + out.grad
        if b.requires_grad:
            b.grad = out.grad if b.grad is None else b.grad + out.grad

    out._backward = _backward
    out._prev = {a, b}
    return out

def matmul(a: Node, b: Node) -> Node:
    out = Node(a.data @ b.data)

    def _backward():
        if a.requires_grad:
            grad_a = out.grad @ b.data.T
            a.grad = grad_a if a.grad is None else a.grad + grad_a
        if b.requires_grad:
            grad_b = a.data.T @ out.grad
            b.grad = grad_b if b.grad is None else b.grad + grad_b

    out._backward = _backward
    out._prev = {a, b}
    return out

def relu(x: Node) -> Node:
    out = Node(np.maximum(0, x.data))

    def _backward():
        if x.requires_grad:
            grad = out.grad * (x.data > 0)
            x.grad = grad if x.grad is None else x.grad + grad

    out._backward = _backward
    out._prev = {x}
    return out
