import os
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Define two neural network models
def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(1,)),
        Dense(1)
    ])
    return model

first_model = create_model()
second_model = create_model()

# Set the learning rate and the loss function
learning_rate = 0.001
optimizer1 = tf.keras.optimizers.Adam(learning_rate=learning_rate)
optimizer2 = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_function = tf.keras.losses.MeanSquaredError()

# Create log directories
log_dir1 = "logs/first_model"
log_dir2 = "logs/second_model"
os.makedirs(log_dir1, exist_ok=True)
os.makedirs(log_dir2, exist_ok=True)

# Set up summary writers for TensorBoard
summary_writer1 = tf.summary.create_file_writer(log_dir1)
summary_writer2 = tf.summary.create_file_writer(log_dir2)

# Custom training loop with TensorBoard integration
num_epochs = 500
batch_size = 1

for epoch in range(num_epochs):
    # Generate a random input and target
    input_data = tf.random.normal(shape=(batch_size, 1))
    target_data = input_data * 2 + 3

    # Train the first model
    with tf.GradientTape() as tape:
        first_model_output = first_model(input_data)
        loss1 = loss_function(target_data, first_model_output)
    gradients1 = tape.gradient(loss1, first_model.trainable_variables)
    optimizer1.apply_gradients(zip(gradients1, first_model.trainable_variables))

    # Train the second model with the output of the first model
    with tf.GradientTape() as tape:
        second_model_input = first_model_output
        second_model_output = second_model(second_model_input)
        loss2 = loss_function(target_data, second_model_output)
    gradients2 = tape.gradient(loss2, second_model.trainable_variables)
    optimizer2.apply_gradients(zip(gradients2, second_model.trainable_variables))

    # Write losses and weights to TensorBoard
    with summary_writer1.as_default():
        tf.summary.scalar('loss', loss1, step=epoch)
        for layer in first_model.layers:
            for w in layer.get_weights():
                tf.summary.histogram(f"{layer.name}/weights", w, step=epoch)

    with summary_writer2.as_default():
        tf.summary.scalar('loss', loss2, step=epoch)
        for layer in second_model.layers:
            for w in layer.get_weights():
                tf.summary.histogram(f"{layer.name}/weights", w, step=epoch)

    if epoch % 50 == 0:
        print(f'Epoch: {epoch}, Loss1: {loss1.numpy()}, Loss2: {loss2.numpy()}')

# Test the combined model
test_input = tf.constant([[5.0]])
first_model_output = first_model(test_input)
second_model_output = second_model(first_model_output)
print(f"Input: {test_input.numpy()[0][0]}, First model output: {first_model_output.numpy()[0][0]}, Second model output: {second_model_output.numpy()[0][0]}")
