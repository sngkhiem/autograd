# Autograd

## Gradient computation and the chain rule

The gradients are computed via reverse-mode automatic differentiation using the chain rule. For any \(x_i\) in the computational graph with final output \(L\):

$$
\frac{dL}{dx_{i}} = \sum_{j \in \mathrm{children}(i)} \frac{dL}{dz_{j}} \times \frac{\partial z_{j}}{\partial x_{i}}
$$

## Backward pass

### Addition
Given:

$$
z = x + y
$$

Then:

$$
\frac{\partial z}{\partial x} = 1,\qquad \frac{\partial z}{\partial y} = 1
$$

Backward step (accumulate):

$$
x.\text{grad} = 1 \cdot z.\text{grad},\qquad y.\text{grad} = 1 \cdot z.\text{grad}
$$

### Multiplication
Given:

$$
z = x \times y
$$

Then:

$$
\frac{\partial z}{\partial x} = y,\qquad \frac{\partial z}{\partial y} = x
$$

Backward step (accumulate):

$$
x.\text{grad} = y \cdot z.\text{grad},\qquad y.\text{grad} = x \cdot z.\text{grad}
$$

### Power
Given:

$$
z = x^n
$$

Then:

$$
\frac{\partial z}{\partial x} = n \cdot x^{\,n-1}
$$

Backward step (accumulate):

$$
x.\text{grad} = n \cdot x^{\,n-1} \cdot z.\text{grad}
$$

### Exponential
Given:

$$
z = e^{x}
$$

Then:

$$
\frac{\partial z}{\partial x} = e^{x} = z
$$

Backward step (accumulate):

$$
x.\text{grad} = z.\text{data} \cdot z.\text{grad}
$$

### Tanh
Given:

$$
z = \tanh(x)
$$

Then:

$$
\frac{\partial z}{\partial x} = 1 - \tanh^{2}(x) = 1 - z^{2}
$$

Backward step (accumulate):

$$
x.\text{grad} = (1 - z^{2}) \cdot z.\text{grad}
$$

**Note on accumulation:** a variable may influence the output through multiple downstream paths, so gradients are **summed** (using `+=`) during backprop to implement the multivariable chain rule.

**Note on topological sort:** We need to perform topological sort (just a DFS) to make sure our backward pass go through from the right to the left of an expression.

## Reference:
[Micrograd](https://github.com/karpathy/micrograd)


