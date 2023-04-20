import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Load the Boston housing dataset
boston = load_boston()

# Convert to pandas dataframe
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name='MEDV')

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_pca, y)

# Predict on new data
new_data = [[-1]] # example new data point
new_data_pca = pca.transform(scaler.transform(new_data))
predicted = model.predict(new_data_pca)

print("Predicted MEDV:", predicted[0])

