"""
    Project Setup and Execution

    This project was completed on Ubuntu 26.04 LTS (Resolute) and follows the CUDA Neural Network Tutorial.

    1. Set Up the Python Environment

    Create and activate a Python virtual environment, then install all the required packages for the Python script.

    2. Download the MNIST Dataset

    Follow this guide to download the MNIST dataset:

    https://www.digitalocean.com/community/tutorials/mnist-dataset-in-python

    The downloaded file, mnist.npz, is cached at:

    ~/.keras/datasets/mnist.npz

    3. Extract the Dataset

    Extract mnist.npz into the project directory:

    unzip ~/.keras/datasets/mnist.npz -d <project_directory>


    Replace <project_directory> with the path to your project folder.

    4. Generate the Required Files

    Run the provided Python script to generate the test data, weights, and biases required for the neural network's forward pass.

    5. Run the Project

    Once the required files have been generated, follow the CUDA Neural Network Tutorial to run the forward pass implementation.
"""

import numpy as np
from pathlib import Path
from tensorflow import keras
from matplotlib import pyplot

# Get the directory where this Python script is located
SCRIPT_DIR = Path(__file__).resolve().parent


# Download/load the MNIST dataset
(train_X, train_y), (test_X, test_y) = keras.datasets.mnist.load_data()

# Save each array as a separate .npy file
np.save(SCRIPT_DIR / "x_train.npy", train_X)
np.save(SCRIPT_DIR / "x_test.npy", test_X)
np.save(SCRIPT_DIR / "y_train.npy", train_y)
np.save(SCRIPT_DIR / "y_test.npy", test_y)

print("MNIST files saved to:", SCRIPT_DIR)

# Preprocess the MNIST dataset 
x_train = np.load(SCRIPT_DIR / "x_train.npy").reshape(-1, 784).astype(np.float32) / 255.0
x_test  = np.load(SCRIPT_DIR / "x_test.npy").reshape(-1, 784).astype(np.float32) / 255.0
y_train = np.load(SCRIPT_DIR / "y_train.npy")
y_test  = np.load(SCRIPT_DIR / "y_test.npy")

# Convert one-hot encoded labels to scalar digit indices if necessary
if y_train.ndim > 1: 
    y_train = y_train.argmax(axis=1)
if y_test.ndim > 1:  
    y_test  = y_test.argmax(axis=1)

# Build and train a simple Multi-Layer Perceptron (MLP)
model = keras.Sequential([
    keras.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
print("Keras test accuracy:", model.evaluate(x_test, y_test)[1])

# Extract weights and biases
W1, b1 = model.layers[0].get_weights()   # W1: (784, 128), b1: (128,)
W2, b2 = model.layers[1].get_weights()   # W2: (128, 10),  b2: (10,)

#  Export to raw binary format for CUDA consumption
W1.astype(np.float32).tofile("W1.bin")
b1.astype(np.float32).tofile("b1.bin")
W2.astype(np.float32).tofile("W2.bin")
b2.astype(np.float32).tofile("b2.bin")
y_test.astype(np.int32).tofile("y_test.bin")
# Export raw pixels (0-255); the CUDA forward pass divides by 255 itself
test_X.reshape(-1, 784).astype(np.float32).tofile("x_test.bin")

