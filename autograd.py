import numpy as np

class Value:
    def __init__(self, data, _child=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_child)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        ans = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * ans.grad 
            other.grad += 1.0 * ans.grad
        ans._backward = _backward

        return ans
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        ans = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * ans.grad
            other.grad += self.data * ans.grad
        ans._backward = _backward

        return ans
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "support int/float power only."
        ans = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad = other * (self.data ** (other-1)) * ans.grad
        ans._backward = _backward

        return ans
    
    def __rmul__(self, other): 
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def exp(self):
        ans = Value(np.exp(self.data), (self, ), 'exp')

        def _backward():
            self.grad += ans.data * ans.grad
        ans._backward = _backward

        return ans
    
    def tanh(self):
        ans = Value(np.tanh(self.data), (self, ), 'tanh')

        def _backward():
            self.grad += (1 - ans.data**2) * ans.grad
        ans._backward = _backward
        return ans
    
    def backward(self):
        self.grad = 1.0
 
        topo = []
        vis = set()
        def dfs(v):
            if v not in vis:
                vis.add(v)
                for u in v._prev:
                    dfs(u)
                topo.append(v)
        dfs(self)

        for node in reversed(topo):
            node._backward()
        
