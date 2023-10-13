import numpy as np

from main_lib import Engine
from NeuralNetworkCortex import NeuralNetworkCortex

def functionNN1(input_data):
    return (input_data[0] / 2) + input_data[1] - input_data[2], input_data[1] * 2

def functionNN2(input_data):
    return input_data[0] - input_data[1]

def functionNN3(input_data):
    return input_data[0] + input_data[1] - input_data[2]

def createSet(function, size):
    # Generate a data set
    num_samples = 10000
    input_data = np.random.rand(num_samples, size)
    output_data = np.array([function(x) for x in input_data])

    # Split the data into train and test
    train_input = input_data[:8000]
    train_output = output_data[:8000]

    test_input = input_data[8000:]
    test_output = output_data[8000:]
    return train_input, train_output, test_input, test_output

math_functions = [functionNN1, functionNN2, functionNN3]
# shapes = [[3,2], [2,1], [3,1]]
shapes = [[3,2], [2,1], [3,1]]
nn_list = []

for index, function in enumerate(math_functions):
    shape = shapes[index] # It means that the model will take a list of 3 element in input and will output one element
    train_input, train_output, test_input, test_output = createSet(function, shape[0])
    nn = NeuralNetworkCortex(shape)
    # nn.train(train_input, train_output, epochs=5)
    # nn.test(test_input, test_output)
    # prediction_data = [0,1,1]
    # print(f"Result of {prediction_data} is {nn.predict([prediction_data])}")
    nn_list.append(nn)

NNetMatrix = [[nn_list[0], nn_list[1]], [nn_list[2]]]
engine = Engine(NNetMatrix)
test_data = [[1, 2, 3], [4, 5]]
result_nn1 = functionNN1(test_data[0])
result = functionNN3([result_nn1[0], result_nn1[1], functionNN2(test_data[1])])

print(f"Result : {engine.run(test_data)}, expected result : {result}")

# # Création de l'orchestrateur
# engine = Engine()

# # Création des première couches
# engine.set_layer(shape: [3, 2], layer_index: 0)
# engine.set_layer(shape: [2, 1], layer_index: 0)

# # Création de la dernière couches
# engine.set_layer(shape: [3, 1], layer_index: 1)

# test_data = [[1, 2, 3], [4, 5]]
# engine.run(test_data)