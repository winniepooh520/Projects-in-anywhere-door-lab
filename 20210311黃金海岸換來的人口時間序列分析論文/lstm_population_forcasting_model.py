# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 22:13:02 2020

@author: 2070
"""

#%%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from numpy import array
from sklearn.preprocessing import MinMaxScaler

n_steps_in, n_steps_out = 6, 2
scaler = MinMaxScaler()
type_ = 'population'
pop_df = pd.read_excel('E:/Hank/population_prediction/Population_Forcasting/data/{0}_df.xlsx'.format(type_))
c_list=[]
c_list.append('location')
for i in range(98, 115):
    c_list.append(str(i))
    
for i in range(9, 15):
    if i < 10:
        pop_df['10{0}'.format(i)]=None
    else:
        pop_df['1{0}'.format(i)]=None

def split_sequence(sequence, n_steps_in, n_steps_out):
    
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
        
	return array(X), array(y)

pop_df.columns = c_list
df = pop_df
mape_list = []

#%%
for i in range(0, len(df)):
    index = i
    df['108'][index] = None
    city_list = df['location'].tolist()

    
    # covert into input/output
    scaled = scaler.fit_transform(df.iloc[index,1:11].values.transpose().reshape(10, 1)).transpose()
    #%
    X, y = split_sequence(scaled[0].tolist(), n_steps_in, n_steps_out)
    print(X.shape, y.shape)
    # summarize the data
    for i in range(len(X)):
    	print(X[i], y[i])
        
    # the dataset knows the number of features, e.g. 2
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    X_train, X_test, y_train, y_test = X[:int(len(X)*0.8)], X[int(len(X)*0.8):], y[:int(len(y)*0.8)], y[int(len(y)*0.8):]
    #%
    
    model = Sequential()
    model.add(LSTM(1000, activation='relu', kernel_initializer='he_uniform', return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(1000, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(n_steps_out))
    model.summary()
    
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mape'])
    model.fit(X, y, epochs=1000, verbose=1)
    print('MAPE:', round(model.evaluate(X, y)[1], 6))

    y_p = model.predict(X, verbose=1)

    X_plt = list(X[0].flatten())
    for i in range(1, len(X)):
        X_plt.append(X[i][-1][0])
    
    y_plt = list(y[0].flatten())
    for i in range(1, len(y)):
        y_plt.append(y[i][-1])
        
    y_p_plt = list(y_p[0].flatten())
    for i in range(1, len(y_p)):
        y_p_plt.append(y_p[i][-1])

    step = 7
    future_x = X[-1][1:]
    for i in range(step):
        input_f = np.append(np.array(X_plt[-5:]), np.array(y_p_plt[-2])).reshape(1, n_steps_in, 1)
        output_f = model.predict(input_f, verbose=1)
        X_plt.append(output_f[0][0])
        y_p_plt.append(output_f[0][-1])

    output = []
    output.append(X_plt[:6] + y_p_plt[:])
    output_f=[]
    output_f.append(y_p_plt[:])

    y_p_plt = scaler.inverse_transform(np.array(output).transpose()).transpose()[-1].tolist()[6:-7]
    y_f_plt = scaler.inverse_transform(np.array(output).transpose()).transpose()[-1].tolist()[-7:]
    X_plt = df.iloc[index,1:-2].tolist()[:9]
    y_plt = df.iloc[index,1:-2].tolist()[6:]
    
    future = y_f_plt[:]
    
    plt.figure(figsize=(32,8))
    plt.plot(df.iloc[index,1:] , label = 'real', color='gray')
    plt.plot()
    plt.plot([None for _ in X_plt[:6]] + y_p_plt + future, label = 'predict', color='blue', alpha=0.6)
    plt.title('{1} forcasting in {0}'.format(df['location'][index], type_))
    plt.xlabel('time')   
    plt.ylabel('{0}'.format(type_))
    plt.grid()
    plt.legend()
    plt.savefig('E:/Hank/population_prediction/Population_Forcasting/result/{1}/{0}.png'.format('_'.join(df['location'][index].split(' ')), type_))
    
    plt.show()
    
    for i in range(len(future)):
        df.iloc[index,11+i:11+i+1] = future[i]
    mape_list.append(model.evaluate(X, y)[1])

df['MAPE'] = None
for i in range(len(mape_list)):
    df['MAPE'][i] = mape_list[i]
    
df.to_excel('E:/Hank/population_prediction/Population_Forcasting/result/six_city_{0}_forcasting.xlsx'.format(type_), index = False)
#%%%















