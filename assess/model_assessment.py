from matplotlib.dates import DateFormatter
from matplotlib.dates import HourLocator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow as tf

def assess_model(model_path, test_data, loss_metric, window_size, 
                 horizon_length):
    model = load_model("{0}/".format(model_path), compile=False)
    model.compile(loss=tf.keras.losses.mae, optimizer=Adam(learning_rate=0.0001), metrics=[loss_metric], run_eagerly=True)
    # convert dataframe into LSTM-compatible matrix
    inputs, labels = df_to_matrix(test_data, window_size, horizon_length)
    labels = matrix_to_list(labels)
    # get predictions and actuals from test set
    predictions = model.predict(inputs)
    predictions = matrix_to_list(predictions)
    # determine error between labels and predicted values
    prediction_error = mae(labels[:(len(labels)-horizon_length)], 
                           predictions[horizon_length:])
    # define fig settings
    plt.rcParams['figure.dpi'] = 300
    date_format = DateFormatter("%H:%M")
    title_fontsize=20
    label_fontsize=18
    tick_fontsize=17
    assessment_fig, assessment_ax = plt.subplots(1, 1, figsize=(21, 8))
    # get datetime to use as index 
    label_datetimes = test_data[window_size:].index
    # plot labels against model predictions
    title = "Prediction Model Assessment"
    assessment_ax.set_title(title, fontsize=title_fontsize)
    assessment_ax.set_ylabel('Normalised Power', fontsize=title_fontsize)
    assessment_ax.tick_params(axis='x', labelsize=tick_fontsize)
    assessment_ax.tick_params(axis='y', labelsize=tick_fontsize)
    assessment_ax.margins(x=0.01, y=0.02)
    assessment_ax.xaxis.set_major_formatter(date_format)
    assessment_ax.xaxis.set_major_locator(HourLocator(interval=3))
    assessment_ax.plot(label_datetimes[:-(horizon_length)], labels[:(len(labels)-horizon_length)])
    assessment_ax.plot(label_datetimes[:-(horizon_length)], predictions[horizon_length:])
    assessment_ax.legend(['Label', 'Predicted'], loc='upper left', fontsize=label_fontsize)
    return prediction_error
    
def df_to_matrix(df, window_size, horizon_size):
  df_as_np = df.to_numpy()
  limit = (len(df_as_np)-(window_size + horizon_size-1))
  X = []
  y = []
  for i in range(limit):
    row = [r for r in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size:i+window_size+horizon_size][:,0]
    y.append(label)
  return np.array(X), np.array(y)

def matrix_to_list(feature_matrix):
    results = feature_matrix[0].tolist()
    for row in feature_matrix[1:]:
        results.append(row[-1])
    return results