from assess import model_assessment
from data import data
import os
import tensorflow as tf
from train import train_lstm as tl

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

#TRAINING DATA
# define sample rate in minutes
sample_rate = 1
# select training data reference date
reference_date = '21/02/08 00:00:00'
# select dataset length in days
data_length = 30
# check if selected training data exists and if not then generate it.
data_path = "data/training_data"
file_path = "{0}/{1}_day_training_data.csv".format(data_path, data_length)
if not os.path.exists(file_path):
    dataset = data.generate_training_data(reference_date, data_length, sample_rate)
else:
    dataset = data.import_dataset(file_path)
feature_dataset = data.extract_dataset_features(dataset)
training_dataset = data.partition_dataset(feature_dataset)
#TRAIN LSTM 
model_path= tl.train_lstm(training_dataset, window_size, horizon_length, 
                          data_length, sample_rate, loss_metric, batch_size, 
                          epochs)
#ASSESS AND PLOT LSTM MODEL OUTPUT
error = model_assessment.assess_model(model_path, training_dataset[2], 
                                      loss_metric, window_size, horizon_length)
    