import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append('../Final_integration.py')
from Final_integration import *

network = Generalized_NeuralNetwork_Backpropagation([1,10,1],['sigmoid','linear'])
p = np.linspace(-2,2,100).reshape(100,1)
g = np.exp(-np.abs(p))*np.sin(np.pi*p).reshape(100,1)
network.train(p,g,learning_rate=0.01,epochs=300,batch_size=20)
p1 = network.prediction(p)
e1 = g-p1
print(np.mean(e1**2))

network1 = Generalized_NeuralNetwork_Backpropagation([1,10,1],['sigmoid','linear'],seed=1234)
p = np.linspace(-2,2,100).reshape(100,1)
g = np.exp(-np.abs(p))*np.sin(np.pi*p).reshape(100,1)
network1.train(p,g,learning_rate=0.01,epochs=300,optimizer='lm')
p2 = network1.prediction(p)
e2 = g-p2
print(np.mean(e2**2))

network2 = Generalized_NeuralNetwork_Backpropagation([1,10,1],['sigmoid','linear'])
p = np.linspace(-2,2,100).reshape(100,1)
g = np.exp(-np.abs(p))*np.sin(np.pi*p).reshape(100,1)
network2.train(p,g,learning_rate=0.01,epochs=300,optimizer='conjugate_gradient',batch_size=20)
p3 = network2.prediction(p)
e3 = g-p3
print(np.mean(e3**2))

df1 = pd.read_csv("/Users/medhaswetasen/Documents/GitHub/Neural-Network-Backpropagation-Optimization/Dataset(Small).csv")
df2 = pd.read_csv("/Users/medhaswetasen/Documents/GitHub/Neural-Network-Backpropagation-Optimization/Dataset(Medium).csv")
df3 = pd.read_csv("/Users/medhaswetasen/Documents/GitHub/Neural-Network-Backpropagation-Optimization/Dataset(Large).csv")

Xdf1 = np.array(df1['PC1']).reshape(len(np.array(df1['PC1'])),1)
Xdf2 = np.array(df2['PC1']).reshape(len(np.array(df2['PC1'])),1)
Xdf3 = np.array(df3['PC1']).reshape(len(np.array(df3['PC1'])),1)
ydf1 = np.array(df1["y"])
ydf2 = np.array(df2["y"])
ydf3 = np.array(df3["y"])

# network12 = Generalized_NeuralNetwork_Backpropagation([1,10,1],['sigmoid','linear'])
# network12.train(Xtrain1,ytrain1,learning_rate=0.01,epochs=300,optimizer='lm')

Xtrain1, Xtest1, ytrain1, ytest1 = train_test_split( Xdf1,ydf1,test_size=0.2, random_state=42 )
network13 = Generalized_NeuralNetwork_Backpropagation([1,10,1],['sigmoid','linear'])
network13.train(Xtrain1,ytrain1,learning_rate=0.01,epochs=300,optimizer='conjugate gradient',batch_size=20)
p1 = network13.prediction(Xtest1)
error1 = ytest1-p1
x = error1[~np.isnan(error1)]
print(np.sum(error1**2))

Xtrain2, Xtest2, ytrain2, ytest2 = train_test_split( Xdf2,ydf2,test_size=0.2, random_state=42 )
network23 = Generalized_NeuralNetwork_Backpropagation([1,10,1],['sigmoid','linear'])
network23.train(Xtrain2,ytrain2,learning_rate=0.01,epochs=300,optimizer='conjugate gradient',batch_size=20)
p2 = network13.prediction(Xtest2)
error2 = ytest2-p2
x = error2[~np.isnan(error2)]
print(np.sum(error2**2))

Xtrain3, Xtest3, ytrain3, ytest3 = train_test_split( Xdf3,ydf3,test_size=0.2, random_state=42 )
network33 = Generalized_NeuralNetwork_Backpropagation([1,10,1],['sigmoid','linear'])
network33.train(Xtrain3,ytrain3,learning_rate=0.01,epochs=300,optimizer='conjugate gradient',batch_size=20)
p3 = network13.prediction(Xtest3)
error3 = ytest3-p3
print(np.mean(error3**2))

Xtrain3, Xtest3, ytrain3, ytest3 = train_test_split( Xdf3,ydf3,test_size=0.2, random_state=42 )
network34 = Generalized_NeuralNetwork_Backpropagation([1,100,1],['sigmoid','linear'])
network34.train(Xtrain3,ytrain3,learning_rate=0.01,epochs=300,optimizer='conjugate gradient',batch_size=20)
p3 = network13.prediction(Xtest3)
error3 = ytest3-p3
print(np.mean(error3**2))
