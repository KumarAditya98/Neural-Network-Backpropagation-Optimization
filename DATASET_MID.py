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


# In[2]:


df1 = pd.read_csv("Spotify_Youtube.csv")
#<<<<<<< Updated upstream
#=======
print(df1.head().to_string())


# In[3]:


print(df1.describe().to_string())


# In[4]:


col = df1.columns
print(col)
#>>>>>>> Stashed changes


# In[5]:

#<<<<<<< Updated upstream

#=======
#>>>>>>> Stashed changes
df1 = df1.drop(['Unnamed: 0', 'Artist', 'Url_spotify', 'Track', 'Album', 'Album_type','Uri','Url_youtube','Title',
                'Channel','Description', 'Licensed', 'official_video'], axis = 1)


#<<<<<<< Updated upstream
#=======
# In[6]:


print(df1.head().to_string())


# In[7]:


#>>>>>>> Stashed changes
df1.Views.fillna(df1.Views.median(),inplace=True)
df1.Likes.fillna(df1.Likes.median(),inplace=True)
df1.Comments.fillna(df1.Comments.median(),inplace=True)
df1.Stream.fillna(df1.Stream.median(),inplace=True)


#<<<<<<< Updated upstream
#=======
# In[8]:


print(df1.describe().to_string())


# In[9]:


#>>>>>>> Stashed changes
df1 = df1.dropna()


# In[10]:


print(df1.describe().to_string())


# In[11]:


for i in range(len(df1.columns)):
#<<<<<<< Updated upstream
    sns.boxplot(x = df1[df1.columns[i]])
    plt.title(f"Distribution of {df1.columns[i]}")
    plt.xlabel(df1.columns[i])
    plt.show()

#>>>>>>> Stashed changes


# In[12]:


fig, ax = plt.subplots(figsize=(20, 15))
corr_matrix1 = df1.corr()
sns.heatmap(corr_matrix1, cmap='coolwarm', annot=True, fmt='.2f', ax = ax, linewidths=0.5)
#<<<<<<< Updated upstream

#=======
plt.show()
#>>>>>>> Stashed changes

# In[13]:


#<<<<<<< Updated upstream
df1.columns
#=======
print(df1.columns)
#>>>>>>> Stashed changes


# In[14]:


X1 = df1[['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness','Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo','Duration_ms','Likes', 'Comments', 'Stream']]
Y1 = df1['Views']


# In[15]:

#<<<<<<< Updated upstream
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X1)


from sklearn.decomposition import PCA
pca1 = PCA(n_components = 14)
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
pcs_df["y"] = Y1

pcs_df.to_csv("Dataset(Medium).csv")
#=======

from sklearn.decomposition import PCA
pca1 = PCA(n_components = 11)
X_pca1 = pca1.fit_transform(X1)
X_pca1.shape


# In[16]:


df1.to_csv("Dataset(Medium).csv")
#>>>>>>> Stashed changes
