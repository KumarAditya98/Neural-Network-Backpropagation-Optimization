#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from summarytools import dfSummary
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)


# In[2]:


df1 = pd.read_csv("Spotify_Youtube.csv")
df1.head(2)


# In[3]:


dfSummary(df1)


# In[4]:


df1.columns


# In[5]:


df1 = df1.drop(['Unnamed: 0', 'Artist', 'Url_spotify', 'Track', 'Album', 'Album_type','Uri','Url_youtube','Title', 
                'Channel','Description', 'Licensed', 'official_video'], axis = 1)


# In[6]:


dfSummary(df1)


# In[7]:


df1.Views.fillna(df1.Views.median(),inplace=True)
df1.Likes.fillna(df1.Likes.median(),inplace=True)
df1.Comments.fillna(df1.Comments.median(),inplace=True)
df1.Stream.fillna(df1.Stream.median(),inplace=True)


# In[8]:


dfSummary(df1)


# In[9]:


df1 = df1.dropna()


# In[10]:


dfSummary(df1)


# In[11]:


for i in range(len(df1.columns)):
    sns.boxplot(x = df1[df1.columns[i]])
    plt.title(f"Distribution of {df1.columns[i]}")
    plt.xlabel(df1.columns[i])
    plt.show()


# In[12]:


fig, ax = plt.subplots(figsize=(20, 15))
corr_matrix1 = df1.corr()
sns.heatmap(corr_matrix1, cmap='coolwarm', annot=True, fmt='.2f', ax = ax, linewidths=0.5)


# In[13]:


df1.columns


# In[14]:


X1 = df1[['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness','Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo','Duration_ms','Likes', 'Comments', 'Stream']]
Y1 = df1['Views']


# In[15]:


from sklearn.decomposition import PCA
pca1 = PCA(n_components = 11)
X_pca1 = pca1.fit_transform(X1)
X_pca1.shape


# In[16]:


df1.to_csv("Dataset(Medium).csv")


# In[17]:


df2 = pd.read_csv("cancer_reg.csv")
df2.head()


# In[18]:


dfSummary(df2)


# In[19]:


df2.columns


# In[20]:


df2 = df2.drop(['index','binnedinc','geography'],axis = 1)


# In[21]:


df2.pctsomecol18_24.fillna(df2.pctsomecol18_24.mean(),inplace = True)
df2.pctemployed16_over.fillna(df2.pctemployed16_over.median(),inplace = True)
df2.pctprivatecoveragealone.fillna(df2.pctprivatecoveragealone.mean(),inplace = True)


# In[22]:


dfSummary(df2)


# In[23]:


for i in range(len(df1.columns)):
    sns.boxplot(x = df2[df2.columns[i]])
    plt.title(f"Distribution of {df2.columns[i]}")
    plt.xlabel(df2.columns[i])
    plt.show()


# In[24]:


fig, ax = plt.subplots(figsize=(20, 15))
corr_matrix2 = df2.corr()
sns.heatmap(corr_matrix2, cmap='coolwarm', annot=True, fmt='.2f', ax = ax, linewidths=0.5)


# In[25]:


X2 = df2.drop('target_deathrate', axis=1)
Y2 = df2['target_deathrate']


# In[26]:


pca2 = PCA(n_components = 17)
X_pca2 = pca2.fit_transform(X2)
X_pca2.shape


# In[27]:


df2.to_csv("Dataset(Large).csv")


# In[28]:


df3 = pd.read_csv("Housing.csv")
df3.head()


# In[29]:


dfSummary(df3)


# In[30]:


df3 = df3.drop(['longitude','latitude','ocean_proximity'],axis = 1)
dfSummary(df3)


# In[31]:


for i in range(len(df3.columns)):
    sns.boxplot(x = df3[df3.columns[i]])
    plt.title(f"Distribution of {df3.columns[i]}")
    plt.xlabel(df3.columns[i])
    plt.show()


# In[32]:


fig, ax = plt.subplots(figsize=(20, 15))
corr_matrix3 = df3.corr()
sns.heatmap(corr_matrix3, cmap='coolwarm', annot=True, fmt='.2f', ax = ax, linewidths=0.5)


# In[33]:


X3 = df3.drop('median_house_value', axis=1)
Y3 = df3['median_house_value']


# In[34]:


pca3 = PCA(n_components = 4)
X_pca3 = pca3.fit_transform(X2)
X_pca3.shape


# In[35]:


df3.to_csv("Dataset(Small).csv")

