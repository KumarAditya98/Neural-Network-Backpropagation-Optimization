import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Generalized_NeuralNetwork_Backpropagation:
    """
    Generalized neural network with R - S1 - S2 - ... - Sm architecture
    Default: Custom Activation function (Sigmoid, Tanh, Linear, Relu, Softmax) can be defined in Hidden Layer
    """
    def __init__(self,Input_neuron_list,activation_function_list,seed=6202):
        self.n_layers = len(Input_neuron_list)
        self.neurons = Input_neuron_list
        self.activation_list = activation_function_list
        self.seed = seed
        np.random.seed(self.seed)
        self.w_list = []
        self.b_list = []
        for i in range(len(self.neurons)-1):
            setattr(self,f"w{i+1}",np.random.uniform(-0.5,0.5,(self.neurons[i+1],self.neurons[i])))
            self.w_list.append(getattr(self,f"w{i+1}"))
            setattr(self,f"b{i+1}",np.random.uniform(-0.5,0.5,(self.neurons[i+1],1)))
            self.b_list.append(getattr(self,f"b{i+1}"))
    def sigmoid(self,x, derivative = False):
        sample = []
        if not(derivative):
            for i in range(len(x)):
                sample.append(1 / (1 + np.exp(-x[i])))
        else:
            for i in range(len(x)):
                sample.append((np.exp(-x[i]))/((1+np.exp(-x[i]))**2))
        final = np.array(sample).reshape(len(x), 1)
        return final
    def tanh(self,x,derivative = False):
        sample = []
        if not(derivative):
            for i in range(len(x)):
                sample.append((np.exp(x[i]) - np.exp(-x[i])) / (np.exp(x[i]) + np.exp(-x[i])))
                final = np.array(sample).reshape(len(x), 1)
                return final
        else:
            return 1.-np.tanh(x)**2
    def relu(self,x,derivative=False):
        sample = []
        for i in range(len(x)):
            if not(derivative):
                if x[i] < 0:
                    sample.append(0)
                else:
                    sample.append(x[i])
            else:
                if x[i] <= 0:
                    sample.append(0)
                else:
                    sample.append(1)
        final = np.array(sample).reshape(len(x), 1)
        return final
    def lin(self,x,derivative=False):
        if not(derivative):
            return x
        else:
            return 1
    def softmax(x,derivative=False):
        # for stability, values shifted down so max = 0
        exp_shifted = np.exp(x - x.max())
        if not(derivative):
            return exp_shifted / np.sum(exp_shifted, axis=0)
        else:
            return exp_shifted / np.sum(exp_shifted, axis=0) * (1 - exp_shifted / np.sum(exp_shifted, axis=0))
    def activation_choice(self, function, x,derivative=False):
        if function == 'tanh':
            return self.tanh(x,derivative)
        elif function == 'relu':
            return self.relu(x,derivative)
        elif function == 'sigmoid':
            return self.sigmoid(x,derivative)
        elif function == 'softmax':
            return self.softmax(x,derivative)
        else:
            return self.lin(x,derivative)
    def stochastic_train(self,train_data,target,learning_rate=0.1,epochs=750):
        alpha = learning_rate
        epochs = epochs
        self.NNOP = np.empty((len(train_data),self.neurons[-1]))
        self.t_plot = np.empty((len(train_data),self.neurons[-1]))
        self.epoch_error = np.empty((epochs,len(train_data),self.neurons[-1]))
        for epoch in range(epochs):
            error = np.empty((self.neurons[-1], len(train_data)))
            zipped = list(zip(train_data,target))
            np.random.shuffle(zipped)
            input, output = zip(*zipped)
            index = 0
            for p,t in zip(input,output):
                n = {i + 1: None for i in range(len(self.activation_list))}
                n[1] = np.dot(self.w1, np.array(p).reshape(len(p), 1)) + self.b1
                a = {i + 1: None for i in range(len(self.activation_list))}
                a[0] = np.array(p).reshape(len(p), 1)
                a[1] = self.activation_choice(self.activation_list[0], n[1])
                for i in range(len(self.activation_list) - 1):
                    n[i+2] = np.dot(self.w_list[i + 1], a[i + 1]) + self.b_list[i + 1]
                    a[i+2] = self.activation_choice(self.activation_list[i + 1], n[i + 2])
                self.NNOP[index] = a[list(a)[-1]].ravel()
                self.t_plot[index] = t.ravel()
                error[:,index] = np.ravel(np.array(t).reshape(len(t),1)-a[list(a)[-1]])
                s = {i:None for i in range(len(self.neurons)-1)}
                F_n_last = self.activation_choice(self.activation_list[-1],n[list(n)[-1]],derivative=True)
                Fn = np.diag([element for row in F_n_last for element in row])
                s[0] = -2 * np.dot(Fn,error[:,index].reshape(self.neurons[-1],1))
                for i in range(len(self.activation_list)-1):
                    F_n = self.activation_choice(self.activation_list[-1-(i+1)], n[list(n)[-1-(i+1)]], derivative=True)
                    Fn_ = np.diag([element for row in F_n for element in row])
                    s[i+1] = np.dot(np.dot(Fn_,self.w_list[-1-i].T),s[i])
                for i in range(len(self.w_list)):
                    self.w_list[i] -= alpha*np.dot(s[list(s)[-1-i]],a[i].T)
                    setattr(self,f"w{i+1}",self.w_list[i])
                    self.b_list[i] -= alpha*s[list(s)[-1-i]]
                    setattr(self, f"b{i + 1}", self.b_list[i])
                index += 1
            self.epoch_error[epoch] = error.T

    def prediction(self,input):
        output = np.empty((len(input),self.neurons[-1]))
        index = 0
        for row in input:
            n = {i+1:None for i in range(len(self.activation_list))}
            n[1] = np.dot(self.w1, np.array(row).reshape(len(row),1)) + self.b1
            a = {i + 1: None for i in range(len(self.activation_list))}
            a[1] = self.activation_choice(self.activation_list[0],n[1])
            for i in range(len(self.activation_list)-1):
                n[i+2] = np.dot(self.w_list[i+1], a[i+1]) + self.b_list[i+1]
                a[i+2] = self.activation_choice(self.activation_list[i+1],n[i+2])
            output[index] = a[list(a)[-1]].ravel()
            index += 1
        return output

    def SSE_Epoch(self):
        x_tick = np.arange(0, len(self.epoch_error))
        series = pd.Series(self.epoch_error, index=x_tick)
        fig, ax = plt.subplots(figsize=(16,8))
        ax.plot(x_tick, series, label='Sum Squared Error')
        ax.set_title("SSE Error Plot")
        ax.set_xlabel("Log Scale for SSE Error")
        ax.set_ylabel("Log Scale for Epochs")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    def NetworkOutput_Vs_Targets(self):
        x_tick = np.arange(len(self.t_plot)-100, len(self.t_plot))
        x_tick1 = np.arange(100)
        series1 = pd.Series(self.a[-100:], index=x_tick)
        series2 = pd.Series(self.t_plot[-100:], index=x_tick)
        series3 = pd.Series(self.a[:100], index=x_tick)
        series4 = pd.Series(self.t_plot[:100], index=x_tick)
        fig, ax = plt.subplots(2,1,figsize=(16,8))
        ax[0].plot(x_tick1, series3, label='Network Outputs')
        ax[0].plot(x_tick1, series4, label='Actual Targets')
        ax[0].set_title("Network Output vs Targets - First 100 Samples")
        ax[0].set_xlabel("Sample Count")
        ax[0].set_ylabel("Outputs")
        ax[0].grid()
        ax[0].legend()
        ax[1].plot(x_tick, series1, label='Network Outputs')
        ax[1].plot(x_tick, series2, label='Actual Targets')
        ax[1].set_title("Network Output vs Targets - Last 100 Samples")
        ax[1].set_xlabel("Sample Count")
        ax[1].set_ylabel("Outputs")
        ax[1].grid()
        ax[1].legend()
        plt.tight_layout()
        plt.show()

    def NN_Function_Approximation(self,p,g):
        a = self.prediction(p)
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(pd.Series(g), label="Actual Function")
        ax.plot(pd.Series(a), label="Function Approximation")
        plt.legend()
        plt.grid()
        plt.show()

# Feedforward test
network = Generalized_NeuralNetwork_Backpropagation([1,10,1],['sigmoid','linear'])
# p = np.array([[1, 1, 2, 2, -1, -2, -1, -2 ],
#               [1, 2, -1, 0, 2, 1, -1, -2]])
# t = np.array([[-1, -1, -1, -1, 1, 1, 1 , 1],
#               [-1, -1, 1 ,1, -1, -1, 1, 1]])

p = np.linspace(-2,2,100).reshape(100,1)
g = np.exp(-np.abs(p))*np.sin(np.pi*p).reshape(100,1)
network.stochastic_train(p,t,epochs=1000)
network.prediction(p)
