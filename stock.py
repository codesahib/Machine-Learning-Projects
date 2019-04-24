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

style.use('ggplot')

#Select start date and end date of the duration under consideration
start=dt.datetime(2016,1,1)
end=dt.datetime(2019,3,24)

#Read data from website of any stock 
df=web.DataReader('SBIN.NS','yahoo',start,end)

#dataframe contains too many columns, but only 'Adj Close' is required
df=df[['Adj Close']]

#Enter the number of days to forecast for, here 30
forecast_out = int(30) 
df['Prediction'] = df[['Adj Close']].shift(-forecast_out)

#X will be array of actual values   
X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)
X.shape


X_forecast = X[-forecast_out:] # set X_forecast equal to last 30
X = X[:-forecast_out] # remove last 30 from X
X.shape

#Y will be array of predicted values
y = np.array(df['Prediction'])
y = y[:-forecast_out]
y.shape

#Define training and testing data sets 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

# Training
clf = LinearRegression()
clf.fit(X_train,y_train)

# Testing
confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)


forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)
