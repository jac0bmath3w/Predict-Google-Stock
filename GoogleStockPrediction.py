#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:39:39 2020
Building RNN to predict Google Stock Price Trend in January 2017
Template from the Deep Learning Course on Udemy
@author: jacob
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For Feature Scaling

from sklearn.preprocessing import MinMaxScaler

# Step 1 Import data

RNN_dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
RNN_dataset_train.dtypes

RNN_dataset_train['Close'] = RNN_dataset_train['Close'].str.extract('(\d*\.?\d*)', expand=False).astype(float)
RNN_dataset_train['Volume'] = RNN_dataset_train['Volume'].str.extract('(\d*\.?\d*)', expand=False).astype(float)

training_set = RNN_dataset_train.iloc[:, 1:6].values


# Step 2 Feature Scaling
# Here we are using Normalization


sc = MinMaxScaler(feature_range = (0,1))
sc_open = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
sc_open.fit(training_set[:,0].reshape(-1,1))

# Step 3: Determine the data structure to determine how much RNN would remember.
# Try using 60 time steps before the time we are trying to predict

X_train = []
y_train = []

for i in range(60,training_set_scaled.shape[0]):
    X_train.append(training_set_scaled[i-60:i,:])
    y_train.append(training_set_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Adding a new dimension
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))


# Step 4: Building the RNN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],5)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
#return_sequences = True when you have more LSTM layers to be added


regressor.add(Dense(1))


# Step 5: Compile the RNN

regressor.compile(optimizer= 'adam', loss = 'mean_squared_error')

# Step 6: Fitting the RNN to the training set

regressor.fit(X_train, y_train, epochs = 50, batch_size =32 )


# Step 7: Making the predictions

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
dataset_test.dtypes


#Volumes is object. 
dataset_test['Volume'] = dataset_test['Volume'].str.extract('(\d*\.?\d*)', expand=False).astype(float)


real_stock_price = dataset_test.iloc[:, [1]].values

dataset_total = pd.concat((RNN_dataset_train.iloc[:,1:6], dataset_test.iloc[:,1:6]), axis = 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = sc.transform(inputs)

X_test = []
X_test_old = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0:5])
    X_test_old.append(inputs[i-60:i,0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))

X_test_old = np.array(X_test_old)
X_test_old = np.reshape(X_test_old, (X_test_old.shape[0], X_test_old.shape[1], 1))

# import pickle

#filename = '5_attribute_RNN_model.sav' #This worked properly in single predictions
# pickle.dump(regressor, open(filename, 'wb'))
# loaded_model = pickle.load(open(filename, 'rb'))


predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc_open.inverse_transform(predicted_stock_price)

predicted_stock_price_first = loaded_model.predict(X_test_old)
predicted_stock_price_first = sc_open.inverse_transform(predicted_stock_price_first)

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price New Model')
plt.plot(predicted_stock_price_first, color = 'yellow', label = 'Predicted Google Stock Price Old Model')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

