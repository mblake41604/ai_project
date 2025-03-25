import numpy as np
import pandas as pd

# dataset
data = pd.read_csv("diabetes.csv")

# features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# normalization
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)  # input weight
        self.b1 = np.zeros((1, hidden_size))  # hidden layer bias
        self.W2 = np.random.randn(hidden_size, output_size)  # output weight
        self.b2 = np.zeros((1, output_size))  # output bias

    # sigmoid activation func
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # sigmoid deriv
    def sigmoid_derivative(self, z):
        return z * (1 - z)

    # forward propagation
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    # backpropagation
    def backward(self, X, y, learning_rate):
        m = X.shape[0]  # number of samples

        # error at the output layer
        dz2 = self.a2 - y.reshape(-1, 1)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # error at the hidden layer
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # ppdate weights/biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    # train
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # forward propagation
            self.forward(X)

            # backpropagation, weight updates
            self.backward(X, y, learning_rate)

            # print epoch/loss
            if epoch % 100 == 0:
                loss = np.mean(-(y * np.log(self.a2) + (1 - y) * np.log(1 - self.a2)))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # prediction
    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

# main

input_size = X.shape[1]  # features
hidden_size = 10  # neurons in hidden layer
output_size = 1  # output layer binary classification

nn = NeuralNetwork(input_size, hidden_size, output_size)

# training
epochs = 1000
learning_rate = 0.1
nn.train(X, y, epochs, learning_rate)

# predictions
predictions = nn.predict(X)

# accuracy
accuracy = np.mean(predictions == y.reshape(-1, 1)) * 100
print(f"Accuracy: {accuracy:.2f}%")
