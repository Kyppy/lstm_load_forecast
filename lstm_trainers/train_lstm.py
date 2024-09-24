# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, feature_df, batch_size, horizon, window):
        self.batch_size = batch_size
        self.feature_df = feature_df.to_numpy(dtype='float16')
        self.horizon = horizon
        self.window = window

    def __len__(self):
        rows = (len(self.feature_df)-(self.window + self.horizon-1))
        return math.floor(rows / self.batch_size)

    def __getitem__(self, idx):
        X = []
        y = []
        for i in np.arange(idx*self.batch_size, (idx+1)*self.batch_size, 1, dtype='int'):
          row = [r for r in self.feature_df[i:i+self.window]]
          X.append(row)
          label = self.feature_df[i+self.window:i+self.window+self.horizon][:,0]
          y.append(label)
        return np.array(X), np.array(y)

def partition_dataset(feature_dataset, train_portion=0.8, validation_portion=0.2):
    training_len = int(len(feature_dataset) * train_portion)
    training_df = feature_dataset[:training_len]
    validation_df = feature_dataset[training_len:]
    
    return training_df, validation_df

def extract_dataset_features(dataset):
    # initialise dataframe to contain only training features
    feature_df = pd.DataFrame({'Power': dataset['Power']})
    feature_df.index = pd.to_datetime(dataset.index)
    
    #conversion constants for time of day,week and year
    day = 60*60*24
    week = day * 7
    year = day * 365.2425

    #NUMERICALLY ENCODE DATE-TIME LABELS AS SINUSOIDS
    # set column for epoch-seconds
    feature_df['Epoch Seconds'] = feature_df.index.map(pd.Timestamp.timestamp)
    # time of day
    feature_df['Day sin'] = np.sin(feature_df['Epoch Seconds'] * (2 * np.pi / day))
    feature_df['Day cos'] = np.cos(feature_df['Epoch Seconds'] * (2 * np.pi / day))
    # time of the week 
    feature_df['Week sin'] = np.sin(feature_df['Epoch Seconds'] * (2 * np.pi / week))
    feature_df['Week cos'] = np.cos(feature_df['Epoch Seconds'] * (2 * np.pi / week))
    # time of the year 
    feature_df['Year sin'] = np.sin(feature_df['Epoch Seconds'] * (2 * np.pi / year))
    feature_df['Year cos'] = np.cos(feature_df['Epoch Seconds'] * (2 * np.pi / year))
    # remove epoch seconds now that mapping is complete
    feature_df = feature_df.drop('Epoch Seconds', axis=1)
     
    return feature_df, len(feature_df.columns)    

def initialise_lstm(window_size, horizon_length, feature_count, loss_metric, 
                    results_path, neurons=50, droput=0.2, learning_rate=0.0001, 
                    patience=2):
    #STACKED LSTM ARCHITECTURE
    # set model params
    model = Sequential()
    model.add(LSTM((neurons*2), dropout=droput, return_sequences=True, input_shape=(window_size, feature_count)))
    model.add(LSTM(neurons, dropout=droput, return_sequences=True))
    model.add(LSTM(neurons, dropout=droput,))
    model.add(Dense(horizon_length))
    
    if loss_metric == 'mae':
        loss_function = "mean_absolute_error"
    elif loss_metric == 'mse':
        loss_function = "mean_squared_error"
    else:
        loss_function = "mean_absolute_error"    
    model.compile(loss=loss_function, optimizer=Adam(learning_rate=learning_rate), 
                  metrics=[loss_metric], run_eagerly=True)
    cp = ModelCheckpoint("{0}/".format(results_path), save_best_only=True) 
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    
    return model, cp, early_stopping