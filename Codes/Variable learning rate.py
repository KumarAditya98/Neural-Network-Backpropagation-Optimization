#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


class NeuralNetwork:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        # initialize the weights and biases for the neural network
        self.weights1 = np.random.randn(input_layer_size, hidden_layer_size)
        self.biases1 = np.random.randn(hidden_layer_size)
        self.weights2 = np.random.randn(hidden_layer_size, output_layer_size)
        self.biases2 = np.random.randn(output_layer_size)
        
        # initialize the variables for the momentum and learning rate
        self.momentum1 = np.zeros((input_layer_size, hidden_layer_size))
        self.momentum2 = np.zeros((hidden_layer_size, output_layer_size))
        self.learning_rate = 0.1
    
    def forward_propagation(self, X):
        # perform forward propagation to calculate the predicted output
        self.z1 = np.dot(X, self.weights1) + self.biases1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.biases2
        self.output = self.z2
    
    def backward_propagation(self, X, y):
        # perform backward propagation to update the weights and biases
        self.error = y - self.output
        self.gradient2 = self.error * 1  # identity activation function
        self.gradient1 = (1 - np.square(self.a1)) * np.dot(self.gradient2, self.weights2.T)
        
        # update the weights and biases using the momentum and learning rate
        self.momentum2 = 0.9 * self.momentum2 + np.dot(self.a1.T, self.gradient2) * self.learning_rate
        self.weights2 += self.momentum2
        self.biases2 += np.sum(self.gradient2, axis=0) * self.learning_rate
        
        self.momentum1 = 0.9 * self.momentum1 + np.dot(X.T, self.gradient1) * self.learning_rate
        self.weights1 += self.momentum1
        self.biases1 += np.sum(self.gradient1, axis=0) * self.learning_rate
    
    def train(self, X, y, epochs):
        for i in range(epochs):
            self.forward_propagation(X)
            self.backward_propagation(X, y)
            
    def predict(self, X):
        self.forward_propagation(X)
        return self.output


# In[3]:


# define the input and target
p = np.linspace(-2, 2, 100)
g = np.exp(-np.abs(p)) * np.sin(np.pi * p)

# create an instance of the NeuralNetwork class
nn = NeuralNetwork(input_layer_size=1, hidden_layer_size=10, output_layer_size=1)

# train the neural network on the input and target
nn.train(X=p.reshape(-1, 1), y=g.reshape(-1, 1), epochs=3000)

# predict the output for the input
prediction = nn.predict(X=p.reshape(-1, 1))

import matplotlib.pyplot as plt

plt.plot(p, g, label='target')
plt.plot(p, prediction, label='prediction')
plt.legend()
plt.show()


# In[4]:


print(prediction)

