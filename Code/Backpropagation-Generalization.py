import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class NeuralNetwork_Backpropagation:
    """
    Generalized neural network with R - S1 - S2 - ... - Sm architecture
    Default: Custom Activation function (Sigmoid, Tanh, Linear, Relu) can be defined in Hidden Layer
    """
    def __init__(self,Input_neuron_list,activation_function_list,seed=6202):
        self.n_layers = len(Input_neuron_list)
        self.neurons = Input_neuron_list
        self.seed = seed
        self.a = np.array([]).reshape(self.neurons[-1],1)
        self.t_plot = np.array([]).reshape(self.neurons[-1],1)
        self.epoch_error = np.array([]).reshape(self.neurons[-1],1)
        np.random.seed(self.seed)
        for i in range(len(self.neurons)):
            setattr(self,f"w{i+1}",np.random.uniform(-0.5,0.5,(self.neurons[i+1],self.neurons[i])))
            setattr(self,f"b{i+1}",np.random.uniform(-0.5,0.5,(self.neurons[i+1],1)))

    def sigmoid(self,x):
        sample = []
        for i in range(len(x)):
            sample.append(1 / (1 + np.exp(-x[i])))
        final = np.array(sample).reshape(len(x), 1)
        return final
    def tanh(self,x):
        sample = []
        for i in range(len(x)):
            sample.append((np.exp(x[i]) - np.exp(-x[i])) / (np.exp(x[i]) + np.exp(-x[i])))
        final = np.array(sample).reshape(len(x), 1)
        return final
    def poslin(self,x):
        sample = []
        for i in range(len(x)):
            if x[i] < 0:
                sample.append(0)
            else:
                sample.append(x[i])
        final = np.array(sample).reshape(len(x), 1)
        return final
    def lin(self,x):
        return x

    # tan(h) = (1-a)*(1+a)
    def stochastic_train(self,train_data,target,learning_rate=0.1,epochs=750):
        alpha = learning_rate
        epochs = epochs
        self.a = np.array([]).reshape(self.neurons[-1], 1)
        self.t_plot = np.array([]).reshape(self.neurons[-1], 1)
        self.epoch_error = np.array([]).reshape(self.neurons[-1], 1)
        for epochs in range(epochs):
            error = np.array([])
            zipped = list(zip(train_data,target))
            np.random.shuffle(zipped)
            input, output = zip(*zipped)
            for p,t in zip(input,output):
                n1 = np.dot(self.w1, p) + self.b1
                a1 = self.sigmoid(n1)
                a2 = np.dot(self.w2, a1) + self.b2
                self.a = np.append(self.a,a2)
                self.t_plot = np.append(self.t_plot,t)
                error = np.append(error,(t-a2))
                S2 = -2 * error[-1]
                temp = [element for row in a1 for element in row]
                temp1 = [i*(1-i) for i in temp]
                fn1 = np.diag(temp1)
                S1 = np.dot(fn1,self.w2.T)*S2
                self.w2 = self.w2 - alpha*np.dot(S2,a1.T)
                self.b2 = self.b2 - alpha*S2
                self.w1 = self.w1 - alpha*np.dot(S1,p)
                self.b1 = self.b1 - alpha*S1
            self.epoch_error = np.append(self.epoch_error,np.sum(error**2))

    def prediction(self,input_matrix):
        output = np.array([]).resha
        for i in input:
            n1 = np.dot(self.w1, i) + self.b1
            a1 = self.sigmoid(n1)
            a2 = np.dot(self.w2, a1) + self.b2
            output = np.append(output,a2)
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



