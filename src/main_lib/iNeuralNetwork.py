
from abc import abstractclassmethod
from enum import Enum

from .tools import are_lists_equal

class Position(Enum):
    FIRST = 0
    MIDDLE = 1
    LAST = 2

    @classmethod
    def format(cls):
        return [(member.value, member.name) for member in cls]

class State(Enum):
    IDLE = 0
    WORKING = 1


class INeuralNetwork:
    
    # input = []
    # output = []
    # readyPreviousNN = []

    def __init__(self, position=Position.MIDDLE) -> None:
        # self.shape = (inputShape, outputShape)
        
        self.input = []
        self.output = []
        self.position = position
        self.next_NN = [] # Next neural network list
        self.previous_NN = []
        self.readyPreviousNN = []

    def predict_network(self, data, previous_nn=None):
        self.input.extend(data[:])
        
        if previous_nn is not None:
            self.readyPreviousNN.append(previous_nn)
        
        if not are_lists_equal(self.readyPreviousNN, self.previous_NN):
            return

        self.output = self.execute(self.input)
        
        for nn in self.next_NN:
            nn.predict_network(self.output, self)
        # self.output = [i + 1 for i in data]

    @abstractclassmethod
    def execute(self, data):
        return data
