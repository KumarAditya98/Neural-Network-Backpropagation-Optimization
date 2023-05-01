import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Generalized_NeuralNetwork_Backpropagation:
    """
    Generalized neural network with R - S1 - S2 - ... - Sm architecture
    Default: Custom Activation function (Sigmoid, Square, Tanh, Linear, Relu, Softmax) can be defined in Hidden Layer
    """
    def __init__(self,Input_neuron_list,activation_function_list,manual_input = False,seed=6202):
        self.n_layers = len(Input_neuron_list)
        self.neurons = Input_neuron_list
        self.activation_list = activation_function_list
        self.seed = seed
        params = 0
        for i in range(1,len(self.neurons)):
            params += self.neurons[i]*(self.neurons[i-1]+1)
        self.total_params = params
        np.random.seed(self.seed)
        self.w_list = []
        self.b_list = []
        if not(manual_input):
            for i in range(len(self.neurons)-1):
                setattr(self,f"w{i+1}",np.random.uniform(-0.5,0.5,(self.neurons[i+1],self.neurons[i])))
                self.w_list.append(getattr(self,f"w{i+1}"))
                setattr(self,f"b{i+1}",np.random.uniform(-0.5,0.5,(self.neurons[i+1],1)))
                self.b_list.append(getattr(self,f"b{i+1}"))
        else:
            for i in range(len(self.neurons)-1):
                setattr(self,f"w{i+1}",np.array(format(input(f"Enter array of weight matrix for w{i+1}")),dtype='float').reshape(self.neurons[i+1],self.neurons[i]))
                self.w_list.append(getattr(self, f"w{i + 1}"))
                setattr(self, f"b{i + 1}",np.array(format(input(f"Enter array of bias matrix for b{i + 1}")),dtype='float').reshape(self.neurons[i+1],1))
                self.b_list.append(getattr(self, f"b{i + 1}"))
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
            return np.array([1]).reshape(1,1)
    def softmax(x,derivative=False):
        # for stability, values shifted down so max = 0
        exp_shifted = np.exp(x - x.max())
        if not(derivative):
            return exp_shifted / np.sum(exp_shifted, axis=0)
        else:
            return exp_shifted / np.sum(exp_shifted, axis=0) * (1 - exp_shifted / np.sum(exp_shifted, axis=0))
    def square(self,x,derivative=False):
        if not(derivative):
            return x**2
        else:
            return 2*x
    def activation_choice(self, function, x,derivative=False):
        if function == 'tanh':
            return self.tanh(x,derivative)
        elif function == 'relu':
            return self.relu(x,derivative)
        elif function == 'sigmoid':
            return self.sigmoid(x,derivative)
        elif function == 'softmax':
            return self.softmax(x,derivative)
        elif function == 'square':
            return self.square(x,derivative)
        else:
            return self.lin(x,derivative)
    def stochastic_train(self,train_data,target,learning_rate=0.1,epochs=750):
        np.random.seed(self.seed)
        alpha = learning_rate
        epochs = epochs
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
                a = {i: None for i in range(len(self.activation_list)+1)}
                a[0] = np.array(p).reshape(len(p), 1)
                a[1] = self.activation_choice(self.activation_list[0], n[1])
                for i in range(len(self.activation_list) - 1):
                    n[i+2] = np.dot(getattr(self,f"w{i+2}"), a[i + 1]) + getattr(self,f"b{i+2}")
                    a[i+2] = self.activation_choice(self.activation_list[i + 1], n[i + 2])
                # self.NNOP[index] = a[list(a)[-1]].ravel()
                # self.t_plot[index] = t.ravel()
                error[:,index] = np.ravel(np.array(t).reshape(len(t),1)-a[list(a)[-1]])
                s = {i:None for i in range(1,len(self.activation_list)+1)}
                F_n_last = self.activation_choice(self.activation_list[-1],n[list(n)[-1]],derivative=True)
                Fn = np.diag([element for row in F_n_last for element in row])
                s[list(s)[-1]] = -2 * np.dot(Fn,error[:,index].reshape(self.neurons[-1],1))
                for i in range(len(self.activation_list)-1):
                    F_n = self.activation_choice(self.activation_list[-2-i], n[list(n)[-2-i]], derivative=True)
                    Fn_ = np.diag([element for row in F_n for element in row])
                    s[list(s)[-2-i]] = np.dot(np.dot(Fn_,self.w_list[-1-i].T),s[list(s)[-1-i]])
                for i in range(len(self.w_list)):
                    self.w_list[i] = self.w_list[i] - alpha*np.dot(s[list(s)[i]],a[i].T)
                    setattr(self,f"w{i+1}",self.w_list[i])
                    self.b_list[i] = self.b_list[i] - alpha*s[list(s)[i]]
                    setattr(self, f"b{i+1}", self.b_list[i])
                index += 1
            self.epoch_error[epoch] = error.T**2

    def batch_train(self,train_data,target,learning_rate=0.1,epochs=750,batch_size = None):
        np.random.seed(self.seed)
        if batch_size == None:
            batch_size = len(train_data)
        alpha = learning_rate
        epochs = epochs
        # self.NNOP = np.empty((len(train_data),self.neurons[-1]))
        # self.t_plot = np.empty((len(train_data),self.neurons[-1]))
        self.epoch_error = np.empty((epochs,len(train_data),self.neurons[-1]))
        for epoch in range(epochs):
            error = np.empty((self.neurons[-1], len(train_data)))
            zipped = list(zip(train_data,target))
            np.random.shuffle(zipped)
            batches = []
            index = 0
            for i in range(0, len(train_data), batch_size):
                batches.append(zipped[i:i + batch_size])
            for j in range(len(batches)):
                input, output = zip(*batches[j])
                grad_w = {i+1:np.zeros(getattr(self,f"w{i+1}").shape) for i in range(len(self.w_list))}
                grad_b = {i+1:np.zeros(getattr(self,f"b{i+1}").shape) for i in range(len(self.b_list))}
                for p,t in zip(input,output):
                    n = {i + 1: None for i in range(len(self.activation_list))}
                    n[1] = np.dot(self.w1, np.array(p).reshape(len(p), 1)) + self.b1
                    a = {i: None for i in range(len(self.activation_list)+1)}
                    a[0] = np.array(p).reshape(len(p), 1)
                    a[1] = self.activation_choice(self.activation_list[0], n[1])
                    for i in range(len(self.activation_list) - 1):
                        n[i+2] = np.dot(getattr(self,f"w{i+2}"), a[i + 1]) + getattr(self,f"b{i+2}")
                        a[i+2] = self.activation_choice(self.activation_list[i + 1], n[i + 2])
                    # self.NNOP[index] = a[list(a)[-1]].ravel()
                    # self.t_plot[index] = t.ravel()
                    error[:,index] = np.ravel(np.array(t).reshape(len(t),1)-a[list(a)[-1]])
                    s = {i:None for i in range(1,len(self.activation_list)+1)}
                    F_n_last = self.activation_choice(self.activation_list[-1],n[list(n)[-1]],derivative=True)
                    Fn = np.diag([element for row in F_n_last for element in row])
                    s[list(s)[-1]] = -2 * np.dot(Fn,error[:,index].reshape(self.neurons[-1],1))
                    for i in range(len(self.activation_list)-1):
                        F_n = self.activation_choice(self.activation_list[-2-i], n[list(n)[-2-i]], derivative=True)
                        Fn_ = np.diag([element for row in F_n for element in row])
                        s[list(s)[-2-i]] = np.dot(np.dot(Fn_,self.w_list[-1-i].T),s[list(s)[-1-i]])
                    for i in range(len(self.w_list)):
                        grad_w[i+1] += np.dot(s[list(s)[i]],a[i].T)
                        grad_b[i+1] += s[list(s)[i]]
                    index += 1
                for i in range(len(self.w_list)):
                    self.w_list[i] = self.w_list[i] - alpha*(grad_w[i+1]/len(input))
                    setattr(self,f"w{i+1}",self.w_list[i])
                    self.b_list[i] = self.b_list[i] - alpha*(grad_b[i+1]/len(input))
                    setattr(self, f"b{i+1}", self.b_list[i])
            output = self.prediction(train_data)
            errorr = target - output
            self.epoch_error[epoch] = np.sum(errorr.T**2)

    def LM_algo(self,train_data,target,mu=0.01,eta=10,max_iter=1000,epsilon=0.01,mu_max = 1e10):
        """
        :param train_data: training data
        :param target: output labels
        :param mu: Used to adjust the training speed vs convergence guarantee
        :param eta: Factor used to adjust mu
        :param max_iter: Number of iterations to find solution in
        :param epsilon: threshold for algorithm convergence
        :return:
        """
        np.random.seed(self.seed)
        mu = mu
        eta = eta
        max_iter = max_iter
        epsilon = epsilon
        mu_max = mu_max
        iteration = 0
        jakobian_rows = len(train_data)*self.neurons[-1]
        jakobian_cols = self.total_params
        self.epoch_error = np.empty((max_iter, len(train_data), self.neurons[-1]))
        while iteration < max_iter:
            output = self.prediction(train_data)
            SSE = np.sum((target - output)**2)
            self.epoch_error[iteration] = SSE
            error = np.empty((self.neurons[-1], len(train_data)))
            index = 0
            s_aug = {i:{j:None for j in range(len(train_data))} for i in range(1,len(self.activation_list)+1)}
            a_aug = {i:{j:None for j in range(len(train_data))} for i in range(len(self.activation_list)+1)}
            for p, t in zip(train_data, target):
                n = {i + 1: None for i in range(len(self.activation_list))}
                n[1] = np.dot(self.w1, np.array(p).reshape(len(p), 1)) + self.b1
                a = {i: None for i in range(len(self.activation_list) + 1)}
                a[0] = np.array(p).reshape(len(p), 1)
                a_aug[0][index]=a[0]
                a[1] = self.activation_choice(self.activation_list[0], n[1])
                a_aug[1][index]=a[1]
                for i in range(len(self.activation_list) - 1):
                    n[i + 2] = np.dot(getattr(self, f"w{i + 2}"), a[i + 1]) + getattr(self, f"b{i + 2}")
                    a[i + 2] = self.activation_choice(self.activation_list[i + 1], n[i + 2])
                    a_aug[i+2][index]=a[i+2]
                error[:, index] = np.ravel(np.array(t).reshape(len(t), 1) - a[list(a)[-1]])
                s = {i: None for i in range(1, len(self.activation_list) + 1)}
                F_n_last = self.activation_choice(self.activation_list[-1], n[list(n)[-1]], derivative=True)
                Fn = np.diag([element for row in F_n_last for element in row])
                s[list(s)[-1]] = -Fn
                s_aug[list(s_aug)[-1]][index] = s[list(s)[-1]]
                for i in range(len(self.activation_list) - 1):
                    F_n = self.activation_choice(self.activation_list[-2 - i], n[list(n)[-2 - i]], derivative=True)
                    Fn_ = np.diag([element for row in F_n for element in row])
                    s[list(s)[-2 - i]] = np.dot(np.dot(Fn_, self.w_list[-1 - i].T), s[list(s)[-1 - i]])
                    s_aug[list(s_aug)[-2-i]][index] = s[list(s)[-2 - i]]
                index += 1
            marquadt_s = {i:None for i in range(1,len(self.activation_list)+1)}
            a_final = {i:None for i in range(len(self.activation_list)+1)}
            for i in s_aug:
                marquadt_s[i] = np.hstack(list(s_aug[i].values()))
            for i in a_aug:
                a_final[i] = np.hstack(list(a_aug[i].values()))
            total_jakobian = np.empty((1,jakobian_cols))
            params = {i+1:self.w_list[i] for i in range(len(self.neurons)-1)}
            for i in range(int(jakobian_rows/self.neurons[-1])):
                    jakobian_row = np.array([])
                    for k in params:
                        s_w = np.array([])
                        s_b = np.array([])
                        a = np.array([])
                        for l in range(getattr(self,f"w{k}").shape[0]):
                            for m in range(getattr(self,f"w{k}").shape[1]):
                                s_w = np.append(s_w,marquadt_s[k][l][i])
                        for l in range(getattr(self,f"w{k}").shape[1]):
                            for m in range(getattr(self,f"w{k}").shape[0]):
                                a = np.append(a,a_final[k-1][l][i])
                        for n in range(getattr(self,f"w{k}").shape[0]):
                            s_b = np.append(s_b,marquadt_s[k][n][i])
                        weight_row = s_w*a
                        freq = int(len(s_w)/getattr(self,f"w{k}").shape[0])
                        indexes = np.arange(freq,len(weight_row)+len(s_b),freq)
                        for x, val in enumerate(s_b):
                            weight_row = np.insert(weight_row, indexes[x], val)
                        jakobian_row = np.hstack([jakobian_row,weight_row])
                    total_jakobian = np.vstack([total_jakobian,jakobian_row])
            total_jakobian = total_jakobian[1:,:]
            delta_x = -np.dot(np.linalg.inv((np.dot(total_jakobian.T,total_jakobian)+mu*np.identity(self.total_params))),np.dot(total_jakobian.T,error.T))
            delta_w = []
            delta_b = []
            w_temp = []
            b_temp = []
            for count in range(1,len(self.w_list)+1):
                ele_w_num = int(getattr(self,f"w{count}").shape[0]*getattr(self,f"w{count}").shape[1])
                delta_w.append(delta_x.ravel()[:ele_w_num].reshape(getattr(self,f"w{count}").shape))
                w_temp.append(getattr(self,f"w{count}")+delta_w[-1])
                delta_x = delta_x[ele_w_num:]
                ele_b_num = int(getattr(self, f"b{count}").shape[0] * getattr(self, f"b{count}").shape[1])
                delta_b.append(delta_x.ravel()[:ele_b_num].reshape(getattr(self, f"b{count}").shape))
                b_temp.append(getattr(self, f"b{count}") + delta_b[-1])
                delta_x = delta_x[ele_b_num:]
            output_new = self.prediction_custom(train_data,w_temp,b_temp)
            SSE_new = np.sum((target - output_new) ** 2)
            if SSE_new < SSE:
                #if np.linalg.norm(2*np.dot(total_jakobian.T,error.T)) < epsilon:
                if SSE_new < 1:
                    self.epoch_error[iteration] = SSE_new
                    print(f"Algorithm has converged in {iteration} epoch.")
                    for i in range(len(self.w_list)):
                        self.w_list[i] = w_temp[i]
                        setattr(self, f"w{i + 1}", self.w_list[i])
                        self.b_list[i] = b_temp[i]
                        setattr(self, f"b{i + 1}", self.b_list[i])
                    return None
                else:
                    for i in range(len(self.w_list)):
                        self.w_list[i] = w_temp[i]
                        setattr(self, f"w{i + 1}", self.w_list[i])
                        self.b_list[i] = b_temp[i]
                        setattr(self, f"b{i + 1}", self.b_list[i])
                    mu = mu/eta
            while SSE_new>=SSE:
                mu = mu*eta
                if mu > mu_max:
                    print(f"Algorithm has reached instability with mu value becoming too large. Saving current weights and biases")
                    print(iteration)
                    for i in range(len(self.w_list)):
                        self.w_list[i] = w_temp[i]
                        setattr(self, f"w{i + 1}", self.w_list[i])
                        self.b_list[i] = b_temp[i]
                        setattr(self, f"b{i + 1}", self.b_list[i])
                    return None
                delta_x = -np.dot(np.linalg.inv((np.dot(total_jakobian.T, total_jakobian) + mu * np.identity(self.total_params))),
                                      np.dot(total_jakobian.T, error.T))
                delta_w = []
                delta_b = []
                w_temp = []
                b_temp = []
                for count in range(1, len(self.w_list) + 1):
                    ele_w_num = int(getattr(self, f"w{count}").shape[0] * getattr(self, f"w{count}").shape[1])
                    delta_w.append(delta_x.ravel()[:ele_w_num].reshape(getattr(self, f"w{count}").shape))
                    w_temp.append(getattr(self, f"w{count}") + delta_w[-1])
                    delta_x = delta_x[ele_w_num:]
                    ele_b_num = int(getattr(self, f"b{count}").shape[0] * getattr(self, f"b{count}").shape[1])
                    delta_b.append(delta_x.ravel()[:ele_b_num].reshape(getattr(self, f"b{count}").shape))
                    b_temp.append(getattr(self, f"b{count}") + delta_b[-1])
                    delta_x = delta_x[ele_b_num:]
                output_new = self.prediction_custom(train_data, w_temp, b_temp)
                SSE_new = np.sum((target - output_new) ** 2)
            iteration += 1
            if iteration>=max_iter:
                print(f"Algorithm should've converged by now. However, maximum number of epoch trainings have been breached. Saving weights and biases.")
                for i in range(len(self.w_list)):
                    self.w_list[i] = w_temp[i]
                    setattr(self, f"w{i + 1}", self.w_list[i])
                    self.b_list[i] = b_temp[i]
                    setattr(self, f"b{i + 1}", self.b_list[i])
                return None
            for i in range(len(self.w_list)):
                self.w_list[i] = w_temp[i]
                setattr(self, f"w{i + 1}", self.w_list[i])
                self.b_list[i] = b_temp[i]
                setattr(self, f"b{i + 1}", self.b_list[i])

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

    def prediction_custom(self, input,w_list,b_list):
        output = np.empty((len(input), self.neurons[-1]))
        index = 0
        for row in input:
            n = {i + 1: None for i in range(len(self.activation_list))}
            n[1] = np.dot(w_list[0], np.array(row).reshape(len(row), 1)) + b_list[0]
            a = {i + 1: None for i in range(len(self.activation_list))}
            a[1] = self.activation_choice(self.activation_list[0], n[1])
            for i in range(len(self.activation_list) - 1):
                n[i + 2] = np.dot(w_list[i + 1], a[i + 1]) + b_list[i + 1]
                a[i + 2] = self.activation_choice(self.activation_list[i + 1], n[i + 2])
            output[index] = a[list(a)[-1]].ravel()
            index += 1
        return output

    def SSE_Epoch(self):
        x_tick = np.arange(0, len(self.epoch_error))
        series = pd.Series(np.sum(self.epoch_error,axis=1).ravel(), index=x_tick)
        fig, ax = plt.subplots(figsize=(16,8))
        ax.plot(x_tick, series, label='Sum Squared Error')
        ax.set_title("SSE Error Plot")
        ax.set_xlabel("Log Scale for SSE Epochs")
        ax.set_ylabel("Log Scale for Error")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

# Feedforward test
# network = Generalized_NeuralNetwork_Backpropagation([1,10,1],['sigmoid','linear'])

# # network.stochastic_train(p,g,learning_rate=0.2,epochs=1000)
# # network.batch_train(p,g,learning_rate=0.2,epochs=1000,batch_size=100)
# network.prediction(p)[:5]
# # network.SSE_Epoch()
# network.LM_algo(p,g)

p = np.linspace(-2,2,100).reshape(100,1)
g = np.exp(-np.abs(p))*np.sin(np.pi*p).reshape(100,1)
network = Generalized_NeuralNetwork_Backpropagation([1,10,1],['sigmoid','linear'],seed=6313)
network.stochastic_train(p,g,epochs = 300,learning_rate=0.2)
network.SSE_Epoch()

# network = Generalized_NeuralNetwork_Backpropagation([1,1,1],['square','linear'],manual_input=True)
# p = np.array([1,2]).reshape(2,1)
# network.LM_algo(p,p)

