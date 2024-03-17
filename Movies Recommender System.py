#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


movies = pd.read_csv ('movies dataset.csv')


# In[4]:


movies.head(10)


# In[5]:


movies.describe()


# In[6]:


movies.info()


# In[7]:


movies.isnull().sum()


# In[8]:


movies.columns


# In[9]:


movies = movies[['id', 'title', 'genre', 'overview']]


# In[10]:


movies


# In[11]:


movies ['tags'] = movies['overview'] + movies['genre']


# In[12]:


movies


# In[13]:


new_data = movies.drop (columns = ['overview', 'genre'])


# In[14]:


new_data


# In[15]:


from sklearn.feature_extraction.text import CountVectorizer


# In[16]:


Cv = CountVectorizer (max_features = 10000, stop_words = 'english')


# In[17]:


Cv


# In[18]:


Vector = Cv.fit_transform(new_data['tags'].values.astype('U')).toarray()


# In[19]:


Vector.shape


# In[20]:


Vector


# In[21]:


from sklearn.metrics.pairwise import cosine_similarity


# In[22]:


similarity = cosine_similarity(Vector)


# In[23]:


similarity


# In[26]:


new_data [new_data['title'] == "The Godfather"]


# In[36]:


distance = sorted (list(enumerate (similarity[2])), reverse = True, key = lambda vector: vector[1])
for i in distance[0:5]:
    print (new_data.iloc[i[0]].title)


# In[37]:


def recommend (movies):
    index = new_data [new_data['title']== movies].index[0]
    distance = sorted (list(enumerate (similarity[index])), reverse = True, key = lambda vector : vector[1])
    for i in distance[0:5]:
        print (new_data.iloc[i[0]].title)


# In[35]:


recommend ("Iron Man")


# In[39]:


recommend ("The Shawshank Redemption")


# In[40]:


recommend ("Captain America")


# In[ ]:




