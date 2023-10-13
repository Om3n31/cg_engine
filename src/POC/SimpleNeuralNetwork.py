import numpy as np


class SimpleNeuralNetwork:

    def __init__(self, shape, function):
        self.shape = shape
        self.function = function

    def predict(self, data):
        return self.function(data)
