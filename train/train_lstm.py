# -*- coding: utf-8 -*-
from gc import collect
import math
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from timeit import default_timer as timer

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

def train_lstm(training_dataset, window_size, horizon_length, duration, 
               sample_rate, loss_metric, batch_size, epochs, 
               results_path="results/models", metrics_path="results/metrics"):
    # define containers to save training metrics 
    error_column_names = ['Training Duration', 
                          "Training Error: {0} Minute Sample Rate".format(sample_rate),
                          "Validation Error: {0} Minute Sample Rate".format(sample_rate)]
    timing_column_names = ['Training Duration', 
                           "Training Time: {0} Minute Sample Rate".format(sample_rate)]
    error_df = pd.DataFrame(columns=error_column_names)
    timing_df = pd.DataFrame(columns=timing_column_names)
    
    # create directory to store LSTM models
    model_path = "{0}/{1}/{2}_min_window/{3}_min_horizon/{4}_min_sample_rate/{5}_training_days".format(
        results_path, loss_metric, window_size, horizon_length, sample_rate, duration)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print("Created {0} directory".format(model_path))
        
    model, cp, early_stopping = initialise_lstm(window_size, horizon_length, 
                                                   len(training_dataset[0].columns), 
                                                   loss_metric, model_path)
    # prevent memory build-up from each iteration 
    clear_session()  
    collect()
    # prevents overloading memory when using very large datasets
    train_gen = DataGenerator(training_dataset[0], batch_size, horizon_length, window_size)
    validation_gen = DataGenerator(training_dataset[1], batch_size, horizon_length, window_size)
    # start timer to measure model training time
    training_start = timer()
    # begin model training
    history = model.fit(train_gen, batch_size=batch_size, 
                        validation_data=validation_gen, epochs=epochs, 
                        callbacks=[cp, early_stopping])
    training_end = timer()
    
    #SAVE TRAINING METRICS
    # export to CSV during loop in case of model training failure
    error_df.loc[len(error_df.index)] = ["{0}".format(duration), 
                                         history.history[loss_metric][-1],
                                         history.history["val_{0}".format(loss_metric)][-1]]
    error_df.to_csv("{0}/error_metrics.csv".format(metrics_path), index=False)
    timing_df.loc[len(timing_df.index)] = ["{0}".format(duration), 
                                           (training_end-training_start)]
    timing_df.to_csv("{0}/timing_metrics.csv".format(metrics_path), index=False)
    
    return model_path