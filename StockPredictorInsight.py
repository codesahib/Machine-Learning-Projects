
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import datetime as dt
from matplotlib import style
import matplotlib.pyplot as plt
#import quandl
from sklearn.linear_model import LinearRegression
from sklearn import model_selection,preprocessing,svm

%matplotlib inline

style.use('ggplot')

start=dt.datetime(2016,1,1)
end=dt.datetime(2018,9,24)
start

df=web.DataReader('SBIN.NS','yahoo',start,end)
df.shape


#start = dt.datetime.strptime(st.strftime(start),"%d-%m-%Y")
#end = datetime.datetime.strptime(end, "%d-%m-%Y")
#date_generated = [start + dt.timedelta(days=x) for x in range(0, (end-start).days)]

date_generated=date_generated[0:676]
print(len(date_generated))
df['Date']=date_generated
df.columns=df.columns.str.replace(' ','_')
df.columns

plt.plot(df.Date,df.Adj_Close)

df=df[['Adj Close']]
df.shape

forecast_out = int(30) 
df['Prediction'] = df[['Adj Close']].shift(-forecast_out)
df.shape

X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)
X.shape

X_forecast = X[-forecast_out:] # set X_forecast equal to last 30
X = X[:-forecast_out] # remove last 30 from X
X.shape

y = np.array(df['Prediction'])
y = y[:-forecast_out]
y.shape

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

# Training
clf = LinearRegression()
clf.fit(X_train,y_train)
# Testing
confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)

forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)


