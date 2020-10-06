#!/usr/bin/env python
# coding: utf-8

# # Task -1

# Predict the percentage of marks of a student based on number of study hours

# In[4]:


import pandas as pd
a = pd.read_csv("C:/Users/admin/Desktop/TSF_task-1.csv")


# In[5]:


a.head()


# # Visualizing the data

# In[8]:


import seaborn as sb
sb.pairplot(a)


# In[7]:


sb.scatterplot(a.Hours,a.Scores)


# # Training the Model

# In[9]:


x = a[["Hours"]]
y = a[["Scores"]]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2)


# In[10]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model = lr.fit(xtrain,ytrain)


# # Evaluating the model

# In[12]:


pred = model.predict(xtest)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(ytest.Scores,pred)
print("Mean Absolute Error =",mae)


# # Making Predictions

# In[15]:


import numpy as np
xtrain = np.array(9.25).reshape(1,-1)
new_pred = lr.predict(xtrain)
print("number of hours ={}".format(xtrain))
print("Predicted Scores ={}".format(new_pred[0]))


# In[ ]:




