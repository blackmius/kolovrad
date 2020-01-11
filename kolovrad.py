import numpy as np

def sigm(x):
    return 1 / (1 + np.exp(-x))

def dsigm(x):
    return x * (1 - x)

def relu(x):
    return .01*x

def drelu(x):
    return .01*np.ones(x.shape)

class Layer:
    def __init__(self, count, parent, learning_rate, activation):
        self.count = count
        self.parent = parent
        self.learning_rate = learning_rate
        if activation == 'sigm':
            self.activation = sigm
            self.deritiv = dsigm
        elif activation == 'relu':
            self.activation = relu
            self.deritiv = drelu
        if self.parent:
            self.weights = np.random.random((self.parent.count, self.count))

    def predict(self):
        self.X = self.activation(np.dot(self.parent.X, self.weights))
    
    def adjust(self, error):
        delta = self.learning_rate * error * self.deritiv(self.X)
        error = np.dot(delta, self.weights.T)
        self.weights += np.dot(self.parent.X.T, delta)
        return error

class Kolovrad:
    def __init__(self, layers, learning_rate=0.5, activation='sigm'):
        self.layers = [ Layer(layers[0], None, learning_rate, activation) ]
        prev = self.layers[0]
        for layer in layers[1:]:
            prev = Layer(layer, prev, learning_rate, activation)
            self.layers.append(prev)
            
    def __repr__(self):
        return 'KOLOVRAD-'+'-'.join([str(layer.count) for layer in self.layers])

    def print_weights(self):
        for layer in self.layers[1:]:
            print(layer.weights)

    def predict(self, X):
        self.layers[0].X = X
        for layer in self.layers[1:]:
            layer.predict()
        return self.layers[-1].X

    def fit(self, X, Y, count=100000):
        for i in range(count):
            y = self.predict(X)
            error = Y - y
            for layer in self.layers[1:][::-1]:
                error = layer.adjust(error)
