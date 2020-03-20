#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:39:39 2020
Building RNN 
Following the tutorial in Deep Learning 
@author: jacob
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Step 1 Import data

RNN_dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = RNN_dataset_train.iloc[:, [1]].values

# Step 2 Feature Scaling
# Here we are using Normalization
# if there is a sigmoid activation function in the output layer, then use normalization

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)


# Step 3: Determine the data structure to determine how much RNN would remember.
# Try using 60 time steps before the time we are trying to predict

X_train = []
y_train = []

for i in range(60,training_set_scaled.shape[0]):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Adding a new dimension
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Step 4: Building the RNN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))
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

regressor.fit(X_train, y_train, epochs = 5, batch_size =32 )


# Step 7: Making the predictions

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, [1]].values

dataset_total = pd.concat((RNN_dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

import pickle

filename = 'untuned_RNN_model.sav' #This worked properly in single predictions
pickle.dump(regressor, open(filename, 'wb'))
#tuned_model = pickle.dumps(grid_search)

loaded_model = pickle.load(open(filename, 'rb'))
