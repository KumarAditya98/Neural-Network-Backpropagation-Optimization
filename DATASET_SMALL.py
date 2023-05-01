#<<<<<<< Updated upstream
#!/usr/bin/env python
# coding: utf-8

# In[1]:

#=======
#>>>>>>> Stashed changes

import numpy as np
import pandas as pd
from summarytools import dfSummary
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

#<<<<<<< Updated upstream
# In[28]:


#=======
#>>>>>>> Stashed changes
df3 = pd.read_csv("Housing.csv")
df3.head()


# In[29]:


dfSummary(df3)


# In[30]:


df3 = df3.drop(['longitude','latitude','ocean_proximity'],axis = 1)
dfSummary(df3)

#<<<<<<< Updated upstream
df3 = df3.dropna()
# In[31]:

#=======

# In[31]:

import numpy as np
import pandas as pd
from summarytools import dfSummary
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

#>>>>>>> Stashed changes

for i in range(len(df3.columns)):
    sns.boxplot(x = df3[df3.columns[i]])
    plt.title(f"Distribution of {df3.columns[i]}")
    plt.xlabel(df3.columns[i])
    plt.show()


# In[32]:


fig, ax = plt.subplots(figsize=(20, 15))
corr_matrix3 = df3.corr()
sns.heatmap(corr_matrix3, cmap='coolwarm', annot=True, fmt='.2f', ax = ax, linewidths=0.5)
#<<<<<<< Updated upstream
plt.show()
#=======

#>>>>>>> Stashed changes

# In[33]:


X3 = df3.drop('median_house_value', axis=1)
Y3 = df3['median_house_value']


# In[34]:


#<<<<<<< Updated upstream
# In[15]:

from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X3)


from sklearn.decomposition import PCA
pca1 = PCA(n_components = 6)
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
pcs_df["y"] = Y3



pcs_df.to_csv("Dataset(Small).csv")
#=======
pca3 = PCA(n_components = 4)
X_pca3 = pca3.fit_transform(X2)
X_pca3.shape


# In[35]:


df3.to_csv("Dataset(Small).csv")
#>>>>>>> Stashed changes

