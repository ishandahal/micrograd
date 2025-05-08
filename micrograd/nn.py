import random
from micrograd import Value

# Simple Neural Network
class Neuron:
    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.random())
        self.parameters = self.weights + [self.bias]

    def __call__(self, xs):
        assert len(xs) == len(self.weights), "inputs and weights have to be same size"
        out = sum( (w * x for w, x in zip(self.weights, xs)) ) + self.bias
        out = out.relu()
        return out

    def __repr__(self):
        return f"Neuron: {len(self.weights)}"

class Layer:
    def __init__(self, nin, n_neurons):
        self.layer = [Neuron(nin) for _ in range(n_neurons)]
        self.parameters = [param for neuron in self.layer for param in neuron.parameters]

    def __call__(self, xs):
        outs = [neuron(xs) for neuron in self.layer]
        return outs[0] if len(outs) == 1 else outs 

    def __repr__(self):
        return f"Layer of ({', '.join(str(n) for n in self.layer)})"

class MLP:
    # def __init__(self, nin, n_layers=1, n_neurons=[5]):
    #     self.nin = nin
    #     self.n_neurons = n_neurons
    #     self.n_layers = n_layers
    #     self.model = self._init_model()
    #     # self.model = [Layer(nin, n_neurons) for _ in range(n_layers)]
    #     self.parameters = [params for layer in self.model for params in layer.parameters]
    
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(sz) - 1)]
        self.parameters = [params for layer in self.layers for params in layer.parameters]

    def __call__(self, xs):
        for layer in self.layers:
            xs = layer(xs)
        return xs

    def _init_model(self):
        model = []
        nin = self.nin
        for i in range(self.n_layers):
            out = self.n_neurons[i]
            model.append(Layer(nin, out))
            nin = out
        # Final layer to create one output
        final_layer = Layer(nin, 1)
        model.append(final_layer)
        return model

    def zero_grad(self):
        # Zero the gradient
        for param in self.parameters:
            param.grad = 0

    def __repr__(self):
        return f"MLP: \n {', '.join(str(layer) for layer in self.layers)}"
        
