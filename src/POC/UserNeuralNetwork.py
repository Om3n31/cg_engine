import numpy as np
from tensorflow import keras


class UserNeuralNetwork:

    def __init__(self, shape):
        neurons_quantity = 32
        self.shape = shape
        # Build a model
        self.model = keras.Sequential([
            keras.layers.Dense(neurons_quantity, activation='relu', input_shape=[shape[0]]),
            keras.layers.Dense(neurons_quantity, activation='relu'),
            keras.layers.Dense(shape[1])
        ])

        # Compile the model
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, train_input, train_output, epochs=10):
        # Train the model
        self.model.fit(train_input, train_output, epochs=epochs, batch_size=32)

    def test(self, train_input, train_output):
        # Test the model
        loss = self.model.evaluate(train_input, train_output)
        print("Evaluation result on Test Data : Loss = {}".format(loss))
        return loss

    def predict(self, data):
        return self.model.predict(data)
