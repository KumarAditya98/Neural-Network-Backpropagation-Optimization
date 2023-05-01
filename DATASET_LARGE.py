#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from summarytools import dfSummary
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
# In[17]:


df2 = pd.read_csv("cancer_reg.csv")

# In[20]:


df2 = df2.drop(['index','binnedinc','geography'],axis = 1)


# In[21]:


df2.pctsomecol18_24.fillna(df2.pctsomecol18_24.mean(),inplace = True)
df2.pctemployed16_over.fillna(df2.pctemployed16_over.median(),inplace = True)
df2.pctprivatecoveragealone.fillna(df2.pctprivatecoveragealone.mean(),inplace = True)


# In[22]:

df2 = df2.dropna()


print(df2.describe().to_string())


# In[23]:


for i in range(len(df2.columns)):
    sns.boxplot(x = df2[df2.columns[i]])
    plt.title(f"Distribution of {df2.columns[i]}")
    plt.xlabel(df2.columns[i])
    plt.show()


# In[24]:


fig, ax = plt.subplots(figsize=(20, 15))
corr_matrix2 = df2.corr()
sns.heatmap(corr_matrix2, cmap='coolwarm', annot=True, fmt='.2f', ax = ax, linewidths=0.5)
plt.show()

# In[25]:


X2 = df2.drop('target_deathrate', axis=1)
Y2 = df2['target_deathrate']


# In[26]:


# In[15]:

from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X2)


from sklearn.decomposition import PCA
pca1 = PCA(n_components = 30)
X_pca1 = pca1.fit_transform(data_scaled)
X_pca1.shape

# Calculate the explained variance ratio
var_ratio = pca1.explained_variance_ratio_

# Plot the graph of the explained variance ratio
plt.plot(range(1, len(var_ratio)+1), var_ratio, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.title('Scree Plot')
plt.show()


# In[16]:
# Create a new DataFrame with the principal components
cols = ['PC'+str(i) for i in range(1,pca1.n_components_+1)]
pcs_df = pd.DataFrame(X_pca1,columns=cols)
pcs_df["y"] = Y2

# In[27]:


pcs_df.to_csv("Dataset(Large).csv")
