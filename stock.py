import pandas as pd
import pandas_datareader.data as web
import numpy as np
import datetime as dt
from matplotlib import style
import matplotlib.pyplot as plt
import quandl
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


style.use('ggplot')


# In[19]:


start=dt.datetime(2016,1,1)
end=dt.datetime(2018,9,24)


# In[52]:


df=web.DataReader('SBIN.NS','yahoo',start,end)



# In[53]:


df=df[['Adj Close']]



# In[65]:


forecast_out = int(30) 
df['Prediction'] = df[['Adj Close']].shift(-forecast_out)



# In[85]:

X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)
X.shape


# In[86]:


X_forecast = X[-forecast_out:] # set X_forecast equal to last 30
X = X[:-forecast_out] # remove last 30 from X
X.shape


# In[90]:


y = np.array(df['Prediction'])
y = y[:-forecast_out]
y.shape


# In[91]:


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)


# In[92]:


# Training
clf = LinearRegression()
clf.fit(X_train,y_train)
# Testing
confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)


# In[82]:


forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)


# In[ ]:




