# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# class Generalized_NeuralNetwork_Backpropagation:
#     """
#     Generalized neural network with R - S1 - S2 - ... - Sm architecture
#     Default: Custom Activation function (Sigmoid, Tanh, Linear, Relu, Softmax) can be defined in Hidden Layer
#     """
#     def __init__(self,Input_neuron_list,activation_function_list,seed=6202):
#         self.n_layers = len(Input_neuron_list)
#         self.neurons = Input_neuron_list
#         self.activation_list = activation_function_list
#         self.seed = seed
#         np.random.seed(self.seed)
#         self.w_list = []
#         self.b_list = []
#         for i in range(len(self.neurons)-1):
#             setattr(self,f"w{i+1}",np.random.uniform(-0.5,0.5,(self.neurons[i+1],self.neurons[i])))
#             self.w_list.append(getattr(self,f"w{i+1}"))
#             setattr(self,f"b{i+1}",np.random.uniform(-0.5,0.5,(self.neurons[i+1],1)))
#             self.b_list.append(getattr(self,f"b{i+1}"))
#     def sigmoid(self,x, derivative = False):
#         sample = []
#         if not(derivative):
#             for i in range(len(x)):
#                 sample.append(1 / (1 + np.exp(-x[i])))
#         else:
#             for i in range(len(x)):
#                 sample.append((np.exp(-x[i]))/((1+np.exp(-x[i]))**2))
#         final = np.array(sample).reshape(len(x), 1)
#         return final
#     def tanh(self,x,derivative = False):
#         sample = []
#         if not(derivative):
#             for i in range(len(x)):
#                 sample.append((np.exp(x[i]) - np.exp(-x[i])) / (np.exp(x[i]) + np.exp(-x[i])))
#                 final = np.array(sample).reshape(len(x), 1)
#                 return final
#         else:
#             return 1.-np.tanh(x)**2
#     def relu(self,x,derivative=False):
#         sample = []
#         for i in range(len(x)):
#             if not(derivative):
#                 if x[i] < 0:
#                     sample.append(0)
#                 else:
#                     sample.append(x[i])
#             else:
#                 if x[i] <= 0:
#                     sample.append(0)
#                 else:
#                     sample.append(1)
#         final = np.array(sample).reshape(len(x), 1)
#         return final
#     def lin(self,x,derivative=False):
#         if not(derivative):
#             return x
#         else:
#             return np.array([1]).reshape(1,1)
#     def softmax(x,derivative=False):
#         # for stability, values shifted down so max = 0
#         exp_shifted = np.exp(x - x.max())
#         if not(derivative):
#             return exp_shifted / np.sum(exp_shifted, axis=0)
#         else:
#             return exp_shifted / np.sum(exp_shifted, axis=0) * (1 - exp_shifted / np.sum(exp_shifted, axis=0))
#     def activation_choice(self, function, x,derivative=False):
#         if function == 'tanh':
#             return self.tanh(x,derivative)
#         elif function == 'relu':
#             return self.relu(x,derivative)
#         elif function == 'sigmoid':
#             return self.sigmoid(x,derivative)
#         elif function == 'softmax':
#             return self.softmax(x,derivative)
#         else:
#             return self.lin(x,derivative)
#
#     def stochastic_train(self, train_data, target, learning_rate=0.1, epochs=750, momentum=0.9):
#         np.random.seed(self.seed)
#         epochs = epochs
#         self.epoch_error = np.empty((epochs, len(train_data), self.neurons[-1]))
#         alpha = learning_rate
#         for epoch in range(epochs):
#             error = np.empty((self.neurons[-1], len(train_data)))
#             zipped = list(zip(train_data, target))
#             np.random.shuffle(zipped)
#             input, output = zip(*zipped)
#             index = 0
#             for p, t in zip(input, output):
#                 n = {i + 1: None for i in range(len(self.activation_list))}
#                 n[1] = np.dot(self.w1, np.array(p).reshape(len(p), 1)) + self.b1
#                 a = {i: None for i in range(len(self.activation_list) + 1)}
#                 a[0] = np.array(p).reshape(len(p), 1)
#                 a[1] = self.activation_choice(self.activation_list[0], n[1])
#                 for i in range(len(self.activation_list) - 1):
#                     n[i + 2] = np.dot(getattr(self, f"w{i + 2}"), a[i + 1]) + getattr(self, f"b{i + 2}")
#                     a[i + 2] = self.activation_choice(self.activation_list[i + 1], n[i + 2])
#                 error[:, index] = np.ravel(np.array(t).reshape(len(t), 1) - a[list(a)[-1]])
#                 s = {i: None for i in range(1, len(self.activation_list) + 1)}
#                 F_n_last = self.activation_choice(self.activation_list[-1], n[list(n)[-1]], derivative=True)
#                 Fn = np.diag([element for row in F_n_last for element in row])
#                 s[list(s)[-1]] = -2 * np.dot(Fn, error[:, index].reshape(self.neurons[-1], 1))
#                 for i in range(len(self.activation_list) - 1):
#                     F_n = self.activation_choice(self.activation_list[-2 - i], n[list(n)[-2 - i]], derivative=True)
#                     Fn_ = np.diag([element for row in F_n for element in row])
#                     s[list(s)[-2 - i]] = np.dot(np.dot(Fn_, self.w_list[-1 - i].T), s[list(s)[-1 - i]])
#
#                 # Update weights and biases
#                 alpha = learning_rate / (1 + epoch / 1000)  # Variable learning rate
#                 for i in range(len(self.w_list)):
#                     dw = np.dot(s[list(s)[i]], a[i].T)
#                     self.v_w_list[i] = momentum * self.v_w_list[i] + alpha * dw  # Add momentum
#                     self.w_list[i] -= self.v_w_list[i]
#                     setattr(self, f"w{i + 1}", self.w_list[i])
#                     db = s[list(s)[i]]
#                     self.v_b_list[i] = momentum * self.v_b_list[i] + alpha * db  # Add momentum
#                     self.b_list[i] -= self.v_b_list[i]
#                     setattr(self, f"b{i + 1}", self.b_list[i])
#
#                 index += 1
#             self.epoch_error[epoch] = error.T ** 2
#
#     def batch_train(self, train_data, target, learning_rate=0.1, epochs=750, batch_size=None, beta=0.9, lr_decay=0.001):
#         np.random.seed(self.seed)
#         if batch_size is None:
#             batch_size = len(train_data)
#         alpha = learning_rate
#         epochs = epochs
#         v_w = {i + 1: np.zeros(getattr(self, f"w{i + 1}").shape) for i in range(len(self.w_list))}
#         v_b = {i + 1: np.zeros(getattr(self, f"b{i + 1}").shape) for i in range(len(self.b_list))}
#         self.epoch_error = np.empty((epochs, len(train_data), self.neurons[-1]))
#         for epoch in range(epochs):
#             error = np.empty((self.neurons[-1], len(train_data)))
#             zipped = list(zip(train_data, target))
#             np.random.shuffle(zipped)
#             batches = []
#             index = 0
#             for i in range(0, len(train_data), batch_size):
#                 batches.append(zipped[i:i + batch_size])
#             batch_error = np.empty((len(train_data), len(batches)))
#             for j in range(len(batches)):
#                 input, output = zip(*batches[j])
#                 grad_w = {i + 1: np.zeros(getattr(self, f"w{i + 1}").shape) for i in range(len(self.w_list))}
#                 grad_b = {i + 1: np.zeros(getattr(self, f"b{i + 1}").shape) for i in range(len(self.b_list))}
#                 for p, t in zip(input, output):
#                     n = {i + 1: None for i in range(len(self.activation_list))}
#                     n[1] = np.dot(self.w1, np.array(p).reshape(len(p), 1)) + self.b1
#                     a = {i: None for i in range(len(self.activation_list) + 1)}
#                     a[0] = np.array(p).reshape(len(p), 1)
#                     a[1] = self.activation_choice(self.activation_list[0], n[1])
#                     for i in range(len(self.activation_list) - 1):
#                         n[i + 2] = np.dot(getattr(self, f"w{i + 2}"), a[i + 1]) + getattr(self, f"b{i + 2}")
#                         a[i + 2] = self.activation_choice(self.activation_list[i + 1], n[i + 2])
#                     error[:, index] = np.ravel(np.array(t).reshape(len(t), 1) - a[list(a)[-1]])
#                     s = {i: None for i in range(1, len(self.activation_list) + 1)}
#                     F_n_last = self.activation_choice(self.activation_list[-1], n[list(n)[-1]], derivative=True)
#                     Fn = np.diag([element for row in F_n_last for element in row])
#                     s[list(s)[-1]] = -2 * np.dot(Fn, error[:, index].reshape(self.neurons[-1], 1))
#                     for i in range(len(self.activation_list) - 1):
#                         F_n = self.activation_choice(self.activation_list[-2 - i], n[list(n)[-2 - i]], derivative=True)
#                         Fn_ = np.diag([element for row in F_n for element in row])
#                         s[list(s)[-2 - i]] = np.dot(np.dot(Fn_, self.w_list[-1 - i].T), s[list(s)[-1 - i]])
#                     for i in range(len(self.w_list)):
#                         v_w[i + 1] = beta * v_w[i + 1] + (1 - beta) * grad_w[i + 1]
#                         v_b[i + 1] = beta * v_b[i + 1] + (1 - beta) * grad_b[i + 1]
#                         setattr(self, f"w{i + 1}", getattr(self, f"w{i + 1}") - alpha * (v_w[i + 1] / len(input)))
#                         setattr(self, f"b{i + 1}", getattr(self, f"b{i + 1}") - alpha * (v_b[i + 1] / len(input)))
#
#                     batch_error[:, j] = np.sum(error ** 2, axis=1)
#                     index += batch_size
#
#                     self.epoch_error[epoch] = np.sum(batch_error.reshape(-1,1), axis=0)
#
#                     alpha = alpha / (1 + lr_decay * epoch)
#
#                 return self.epoch_error[:epoch + 1]
#
#     def prediction(self,input):
#         output = np.empty((len(input),self.neurons[-1]))
#         index = 0
#         for row in input:
#             n = {i+1:None for i in range(len(self.activation_list))}
#             n[1] = np.dot(self.w1, np.array(row).reshape(len(row),1)) + self.b1
#             a = {i + 1: None for i in range(len(self.activation_list))}
#             a[1] = self.activation_choice(self.activation_list[0],n[1])
#             for i in range(len(self.activation_list)-1):
#                 n[i+2] = np.dot(self.w_list[i+1], a[i+1]) + self.b_list[i+1]
#                 a[i+2] = self.activation_choice(self.activation_list[i+1],n[i+2])
#             output[index] = a[list(a)[-1]].ravel()
#             index += 1
#         return output
#
#     def SSE_Epoch(self):
#         x_tick = np.arange(0, len(self.epoch_error))
#         series = pd.Series(np.sum(self.epoch_error,axis=1).ravel(), index=x_tick)
#         fig, ax = plt.subplots(figsize=(16,8))
#         ax.plot(x_tick, series, label='Sum Squared Error')
#         ax.set_title("SSE Error Plot")
#         ax.set_xlabel("Log Scale for SSE Error")
#         ax.set_ylabel("Log Scale for Epochs")
#         plt.xscale("log")
#         plt.yscale("log")
#         plt.grid()
#         plt.legend()
#         plt.tight_layout()
#         plt.show()
#
# # Feedforward test
# network = Generalized_NeuralNetwork_Backpropagation([1,10,1],['sigmoid','linear'])
# p = np.linspace(-2,2,100).reshape(100,1)
# g = np.exp(-np.abs(p))*np.sin(np.pi*p).reshape(100,1)
# # network.stochastic_train(p,g,learning_rate=0.2,epochs=1000)
# network.batch_train(p,g,learning_rate=0.01,epochs=150,batch_size=20)
# network.prediction(p)[:5]
# network.SSE_Epoch()
