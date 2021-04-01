import streamlit as st
import tensorflow as tf
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
st.set_option('deprecation.showPyplotGlobalUse', False)
st.header('EQUITY MARKET PREDICTION AND ANALYSIS')
import pandas_datareader as pdr
import numpy as np
#image = Image.open('image.png')
#st.image(image,use_column_width=True)

radio_list=['Service Company','Product Company']
select_type=st.sidebar.radio('Select Company Type',radio_list)

if select_type=='Service Company':
  stock_list = [ 'INFY','WIT','TCS','CTSH','ORCL','GS','TDC','CAP.PA','LTI.NS']
  select = st.sidebar.selectbox('Select Company',stock_list)
  st.write(select)
  if select == 'INFY':
        model = load_model('infosys.hdf5')
  elif select =='WIT':
      model = load_model('wipro.hdf5')
  elif select =='TCS':
      model = load_model('tcs.hdf5')
  elif select =='CTSH':
      model = load_model('cognizant.hdf5')
  elif select =='ORCL':
      model = load_model('oracle.hdf5')
  elif select =='GS':
      model = load_model('goldmansachs.hdf5')
  elif select =='TDC':
      model = load_model('teradata.hdf5')
  elif select =='CAP.PA':
      model = load_model('capgemini.hdf5')
  elif select =='LTI.NS':
      model = load_model('lti.hdf5')
elif select_type=='Product Company':
  stock_list = ['AAPL','BABA','GOOGL','MSFT','AMZN', 'ADBE','DELL','HP','SNE']
  select = st.sidebar.selectbox('Select IT Product Company',stock_list)
  st.write(select)
  if select == 'AAPL':
        model = load_model('apple.hdf5')
  elif select =='BABA':
      model = load_model('baba.hdf5')
  elif select =='GOOGL':
      model = load_model('Google.hdf5')
  elif select =='MSFT':
      model = load_model('microsoft.hdf5')
  elif select =='AMZN':
      model = load_model('amazon.hdf5')
  elif select =='ADBE':
      model = load_model('adobe.hdf5')
  elif select =='DELL':
      model = load_model('dell.hdf5')
  elif select =='HP':
      model = load_model('hp.hdf5')
  elif select =='SNE':
      model = load_model('sony.hdf5')


df = pdr.DataReader(select, data_source='yahoo',start='2015-05-27', end='2020-05-22')
df1=df.reset_index()['Close']
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)
 
# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

test_predict=model.predict(X_test)
test_predict=scaler.inverse_transform(test_predict)

x_input=test_data[len(test_data)-100:].reshape(1,-1)
#x_input.shape
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array

lst_output=[]
n_steps=100
i=0

days_p = st.sidebar.slider('Days to forecast',1,30)
while(i<days_p):
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1
day_new=np.arange(1,101)
day_pred=np.arange(101,101+days_p)
import matplotlib.pyplot as plt
#visualize the closeing price history
plt.figure(figsize=(16,8))
st.subheader("Close Price History")
#plt.title(,fontsize=18)
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
st.pyplot()
plt.figure(figsize=(16,8))
st.subheader("Test Data -100 Days")
plt.plot(day_new,scaler.inverse_transform(df1[len(train_data)+len(test_data)-100:]))
plt.xlabel('No of Days', fontsize=18)
plt.ylabel(' Price USD ($)', fontsize=18)
plt.plot(day_pred,scaler.inverse_transform(lst_output))
st.pyplot()
plt.figure(figsize=(16,8))
st.subheader("Test Data - 100 Days (Normalized)")
df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[len(train_data)+len(test_data)-100:])
plt.xlabel('No of days',fontsize=18) 
plt.ylabel('Price',fontsize=18)
st.pyplot()
df3=scaler.inverse_transform(df3).tolist()
# #plot the data
data= df.filter(['Close'])
train = data[:training_size+101]
valid = data[training_size+101:]
valid['Predictions']= test_predict
#print(valid)
#visualize the data
plt.figure(figsize=(16,8))
st.subheader("Model")
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
st.pyplot()
st.subheader("Prediction and Close comparison")
st.write(valid)
