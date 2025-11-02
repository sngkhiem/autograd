import Value
import random

class Neuron:
    def __init__(self, inputSize):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(inputSize)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = activation.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, inputSize, outputSize):
        self.neurons = [Neuron(inputSize) for _ in range(outputSize)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        params = []
        for neuron in self.neurons:
            curParam = neuron.parameters()
            params.extend(curParam)
        return params
    
class MLP:
    def __init__(self, inputSize, outputSizes):
        size = [inputSize] + outputSizes
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(outputSizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            for neuron in layer.neurons:
                curParam = neuron.parameters()
                params.extend(curParam)
        return params
    