import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_ih = np.random.randn(self.hidden_nodes, self.input_nodes)
        self.weights_ho = np.random.randn(self.output_nodes, self.hidden_nodes)

        # Initialize biases
        self.bias_h = np.random.randn(self.hidden_nodes, 1)
        self.bias_o = np.random.randn(self.output_nodes, 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_sigmoid(self, y):
        return y * (1 - y)

    def feedforward(self, inputs):
        inputs = np.array(inputs).reshape((self.input_nodes, 1))

        hidden = np.dot(self.weights_ih, inputs) + self.bias_h
        hidden = self.sigmoid(hidden)

        output = np.dot(self.weights_ho, hidden) + self.bias_o
        output = self.sigmoid(output)

        return output

    def train(self, inputs, targets, learning_rate=0.1):
        inputs = np.array(inputs).reshape((self.input_nodes, 1))
        targets = np.array(targets).reshape((self.output_nodes, 1))

        # Feedforward
        hidden = np.dot(self.weights_ih, inputs) + self.bias_h
        hidden = self.sigmoid(hidden)

        output = np.dot(self.weights_ho, hidden) + self.bias_o
        output = self.sigmoid(output)

        # Calculate output error
        output_errors = targets - output

        # Compute the mean squared error
        mse = np.mean(np.square(output_errors))

        # Calculate gradient
        gradients = learning_rate * output_errors * self.derivative_sigmoid(output)

        # Adjust weights and biases
        self.weights_ho += np.dot(gradients, hidden.T)
        self.bias_o += gradients

        # Calculate hidden layer errors
        hidden_errors = np.dot(self.weights_ho.T, output_errors)

        # Calculate hidden gradient
        hidden_gradients = learning_rate * hidden_errors * self.derivative_sigmoid(hidden)

        # Adjust hidden weights and biases
        self.weights_ih += np.dot(hidden_gradients, inputs.T)
        self.bias_h += hidden_gradients

        return mse  # Return the mean squared error

    def predict(self, inputs):
        return self.feedforward(inputs)

# Define the training data
training_data = [
    {"inputs": [0, 0], "targets": [0]},
    {"inputs": [0, 1], "targets": [0]},
    {"inputs": [1, 0], "targets": [0]},
    {"inputs": [1, 1], "targets": [1]},
]

if __name__ == '__main__':
    # Create a new neural network
    nn = NeuralNetwork(2, 2, 1)

    # Train the network
    for epoch in range(50000):
        mse_total = 0
        for data in training_data:
            mse = nn.train(data["inputs"], data["targets"])
            mse_total += mse
        if epoch % 5000 == 0:
            print(f"Epoch {epoch}, Average MSE: {mse_total / len(training_data)}")

    # Test the network
    for data in training_data:
        print(f"For inputs {data['inputs']} prediction is {nn.predict(data['inputs'])}")
