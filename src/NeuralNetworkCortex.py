from SimpleNeuralNetwork  import SimpleNeuralNetwork
from backend.CortexBack.src.main_lib.iNeuralNetwork import INeuralNetwork
# from main_libINeuralNetwork import INeuralNetwork


class NeuralNetworkCortex(SimpleNeuralNetwork, INeuralNetwork):
    """
    This class inherit from the Neural Network which is the network from the lib.
    It's also inherit from the INeuralNetwork which is the interface to work with the CortexGroove engine.
    """

    def __init__(self, shape, training_function=None):
        super(NeuralNetworkCortex, self).__init__(shape, training_function)
        self.training_function = training_function
        # super(INeuralNetwork, self).__init__()

    # def train(self, train_input, train_output, epochs=10):
    #     """
    #     Might be useful for later, if we want to train the data dynamically
    #     :param epochs:
    #     :param train_input:
    #     :param train_output:
    #     :return:
    #     """
    #     super(NeuralNetworkCortex, self).train(train_input, train_output, epochs)

    def execute(self, data):
        """
        Return the prediction, but most importantly set the output value
        :param data: is a list of multiple input to test
        :return:
        """
        prediction = super(NeuralNetworkCortex, self).predict(data)
        print(f"Input data : {data}, result : {prediction.tolist()}, expected result : {self.training_function(data)}")
        return prediction.tolist()

    # def execute(self, data):
    #     self.output = self.predict([data])

