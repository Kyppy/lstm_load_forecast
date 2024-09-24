from data import data
from gc import collect
import os
from lstm_trainers import train_lstm as tl
import pandas as pd
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from timeit import default_timer as timer

def import_dataset(data_path, duration):
    dataset_df = pd.read_csv("{0}/{1}_day_training_data.csv".format(data_path, duration), 
                             header=0, infer_datetime_format=True, 
                             parse_dates=['datetime'], index_col='datetime')
    dataset_df.index = pd.to_datetime(dataset_df.index)
    return dataset_df
    
#TRAINING SETTINGS
batch_size = 32
epochs=3
loss_metric = 'mae'
# defined in minutes
horizon_length = 1
window_size = 60 
# set TF memory allocation to growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

#GENERATE TRAINING DATA
# define training data sample rate in seconds
sample_rate = 60
# select training data reference date and durations in days
reference_date = '21/02/08 00:00:00'
durations = [30, 60, 90]
# check if selected training data exists and if not then generate it.
data_path = "data/training_data"
for duration in durations:
    file_path = "{0}/{1}_day_training_data.csv".format(data_path, duration)
    if not os.path.exists(file_path):
        data.generate_training_data(reference_date, durations, sample_rate)
        print("Created {0}-day training dataset".format(duration))
# define container to save model training error metrics
error_column_names = ['Training Days',"{0} Training Error".format(loss_metric.upper()),
                      "{0} Validation Error".format(loss_metric.upper())]
error_df = pd.DataFrame(columns=error_column_names)

#SELECT LSTM TRAINING METHOD
# either 'A' or 'B'
method = 'A'

#TRAIN LSTM
if method =='A':
    #RUN METHOD 'A' MODEL TRANING
    results_path="results/models/method_a"
    metrics_path="results/metrics/method_a"
    
    # define container to save training duration metrics
    timing_column_names = ['Training Days',"{0} Training Time".format(loss_metric.upper())]
    timing_df = pd.DataFrame(columns=timing_column_names)
    
    for duration in durations:
        dataset = import_dataset(data_path, duration)
        # normalise dataset 'Power' column
        dataset['Power'] = (dataset['Power'] - dataset['Power'].min()) / (dataset['Power'].max() - dataset['Power'].min())
        
        feature_df, feature_count = tl.extract_dataset_features(dataset)
        train_df, validation_df = tl.partition_dataset(feature_df)
        
        # create directory to store LSTM models
        model_path = "{0}/{1}/{2}_min_window/{3}_min_horizon/{4}_training_days".format(
            results_path, loss_metric, window_size, horizon_length, duration)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            print("Created {0} directory".format(model_path))
            
        model, cp, early_stopping = tl.initialise_lstm(window_size, horizon_length, 
                                                       feature_count, loss_metric,
                                                       model_path)
        # prevent memory build-up from each iteration 
        clear_session()  
        collect()
        # prevents overloading memory when using very large datasets
        train_gen = tl.DataGenerator(train_df, batch_size, horizon_length, window_size)
        validation_gen = tl.DataGenerator(validation_df, batch_size, horizon_length, window_size)
        # start timer to measure model training time
        training_start = timer()
        # begin model training
        history = model.fit(train_gen, batch_size=batch_size, 
                            validation_data=validation_gen, epochs=epochs, 
                            callbacks=[cp, early_stopping])
        training_end = timer()
        
        #SAVE METHOD 'A' TRAINING METRICS
        timing_df.loc[len(timing_df.index)] = ["{0}".format(duration), 
                                               (training_end-training_start)]
        error_df.loc[len(error_df.index)] = ["{0}".format(duration), 
                                             history.history[loss_metric][-1],
                                             history.history["val_{0}".format(loss_metric)][-1]]
        # export to CSV during loop in case of model training failure
        error_df.to_csv("{0}/method_a/error_metrics.csv".format(metrics_path), index=False)
        timing_df.to_csv("{0}/method_a/timing_metrics.csv".format(metrics_path), index=False)
    

    
    
    
    
    
    