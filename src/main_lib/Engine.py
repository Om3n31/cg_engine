from .iNeuralNetwork import INeuralNetwork, Position
# Local creation

# from iNeuralNetwork import INeuralNetwork
# from iNeuralNetwork import Position
# from main_lib.INeuralNetwork import INeuralNetwork
from .tools import *

class Engine:

    def __init__(self, NeuralNetList) -> None:
        self.neuralNetList = NeuralNetList
        self.layers = []
        # for nnLayer in NNetMatrix:
        #     self.layers.append(Layer(nnLayer))

    # def set_layer(self, neural_net, layer_index):
    #     if self.layers[layer_index] != None:
    #         self.layers.append(Layer([neural_net]))
    #     else :
    #         self.layers[layer_index].NNList.append(neural_net)

    def run(self, inputData):
        data = inputData
        output_data = []
        first_nn = [i for i in self.neuralNetList if i.position is Position.FIRST.value]
        
        for index, nn in enumerate(first_nn):
            nn.predict_network(data[index])
            output_data.extend(nn.output)

        return [i.output for i in self.neuralNetList if i.position is Position.LAST.value]

if __name__ == '__main__':
    a = INeuralNetwork(position=Position.FIRST)
    b = INeuralNetwork(position=Position.FIRST)
    c = INeuralNetwork(position=Position.LAST)
    a.name = "a"
    b.name = "b"
    c.name = "c"
    a.next_NN.append(c)
    b.next_NN.append(c)
    c.previous_NN.append(b)
    c.previous_NN.append(a)

    NNetMatrix = [a, b, c]
    engine = Engine(NNetMatrix)
    result = engine.run([[1,2,3], [4,5,6,7]])
    print(result)
