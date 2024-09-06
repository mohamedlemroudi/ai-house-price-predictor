import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Input data (square meters) and output data (price in thousands of dollars)
square_meters = np.array([50, 75, 100, 125, 150, 200, 250], dtype=float)
prices = np.array([150, 200, 250, 300, 350, 450, 500], dtype=float)

# Define the layers of the neural network
hidden1 = tf.keras.layers.Dense(units=3, input_shape=[1])
hidden2 = tf.keras.layers.Dense(units=3)
output = tf.keras.layers.Dense(units=1)

# Create the sequential model
model = tf.keras.Sequential([hidden1, hidden2, output])

# Compile the model with optimizer and loss function
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# Train the model
print("Starting training...")
history = model.fit(square_meters, prices, epochs=1000, verbose=False)
print("Model trained!")

# Make a prediction
print("Let's make a prediction!")
result = model.predict(np.array([[180.0]]))  # Prediction for a 180 square meter house
print("The estimated price is $" + str(result[0][0] * 1000) + " dollars")

import matplotlib.pyplot as plt

# Create a new figure with a black background
plt.figure(facecolor='black')

# Plot the loss over the epochs
plt.xlabel("Epoch", color='white')
plt.ylabel("Loss Magnitude", color='white')
plt.plot(history.history["loss"], color='white')  # Set the line color to white for contrast

# Set the background color of the axes
plt.gca().set_facecolor('black')

# Display the plot
plt.show()

# Print internal model weights
print("Internal model weights")
print(hidden1.get_weights())
print(hidden2.get_weights())
print(output.get_weights())
