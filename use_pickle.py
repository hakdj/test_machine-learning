#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pickle


# In[11]:


pickle.load(open('./saves/favorite_save.pkl','rb'))


# In[12]:


favorite_load=pickle.load(open('./saves/favorite_save.pkl','rb'))
print(favorite_load)

# In[13]:


type(favorite_load)


# In[14]:


favorite_load['tiger']


# In[19]:


autompg_lr = pickle.load(open('./saves/autompg_lr.pkl','rb'))
# 뒤에 rb or wb는 read,write, 바이너리 형식임
autompg_lr


# In[18]:


type(autompg_lr)


# In[ ]:

# input from outside
a-3504.0
b=8
import numpy as np;
pre=np.array([[a,b]])
print(autompg_lr.predict(pre))

#print(autompg_lr.predict([[3504.0,8]]))
# 위나 아래나 같음.
