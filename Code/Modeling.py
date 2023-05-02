import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Final_integration import Generalized_NeuralNetwork_Backpropagation
from sklearn.preprocessing import StandardScaler

d1 = pd.read_csv('Dataset/Dataset(Small).csv',index_col='Unnamed: 0')
d2 = pd.read_csv('Dataset/Dataset(Medium).csv',index_col='Unnamed: 0')
d3 = pd.read_csv('Dataset/Dataset(Large).csv',index_col='Unnamed: 0')

d1.y.dtype
d2.y.dtype
d3.y.dtype
# All regression problems

print(f"Small dataset shape: {d1.shape}")
print(f"Medium dataset shape: {d2.shape}")
print(f"Large dataset shape: {d3.shape}")

# I'll limit the number of rows to 500 rows in each dataset as the comparison to be performed is on the number of features and also for computation reasons.

d1 = d1.sample(n=500,random_state=6202)
d2 = d2.sample(n=500,random_state=6202)
d3 = d3.sample(n=500,random_state=6202)
# Small dataset shape: (500, 8)
# Medium dataset shape: (500, 16)
# Large dataset shape: (500, 32)

d1X = d1.iloc[:,:-1]
d1Y = d1.iloc[:,-1]
d2X = d2.iloc[:,:-1]
d2Y = d2.iloc[:,-1]
d3X = d3.iloc[:,:-1]
d3Y = d3.iloc[:,-1]

# Assuming all cleaning has been done. I'll begin with the train test splits.

X_train1, X_test1, y_train1, y_test1 = train_test_split(d1X,d1Y,test_size=0.2,random_state=6202)
X_train2, X_test2, y_train2, y_test2 = train_test_split(d2X,d2Y,test_size=0.2,random_state=6202)
X_train3, X_test3, y_train3, y_test3 = train_test_split(d3X,d3Y,test_size=0.2,random_state=6202)
X_train1.shape
# -----------------------------------------x-----------------------------------------------------------------------------
# Model-building using custom neural network codes (Simple networks with (R-10-1) architecture - sigmoid and linear activations
# -----------------------------------------x-----------------------------------------------------------------------------

# SGD for small dataset
network1 = Generalized_NeuralNetwork_Backpropagation([6,10,1],['sigmoid','linear'],seed=2345)
# Making sure shapes of train and test are according to custom program
# X_train1.shape
# y_train1.shape
# network1.train(X_train1.values,np.array(y_train1_scaled).reshape(-1,1),learning_rate=0.001,epochs=1000,optimizer='sgd')
# network1.SSE_Epoch()
# Exploding weights. The y_train needs to be standardized as well
scaler = StandardScaler()
y_train1_scaled = scaler.fit_transform(y_train1.values.reshape(-1,1))
y_test1_scaled = scaler.transform(y_test1.values.reshape(-1,1))
network1.train(X_train1.values,np.array(y_train1_scaled).reshape(-1,1),learning_rate=0.001,epochs=1000,optimizer='sgd')
network1.SSE_Epoch()
