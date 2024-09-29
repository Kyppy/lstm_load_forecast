# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from numpy import nan
from numpy import isnan
from pandas import read_csv
from datetime import datetime
import os.path

def clean_raw_data():
    # load all data
    dataset = read_csv('raw/household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
    # mark all missing values
    dataset.replace('?', nan, inplace=True)
    # make dataset numeric
    dataset = dataset.astype('float32')
    # fill missing dataset values
    fill_missing(dataset.values)
    # save cleaned dataset
    dataset.to_csv('cleaned/household_power_consumption.csv')

def extract_dataset_features(dataset):
    # normalise dataset 'Power' column
    dataset['Power'] = (dataset['Power'] - dataset['Power'].min()) / (dataset['Power'].max() - dataset['Power'].min())
    # initialise dataframe to contain only training features
    feature_df = pd.DataFrame({'Power': dataset['Power']})
    feature_df.index = pd.to_datetime(dataset.index)
    # conversion constants for time of day,week and year
    day = 60*60*24
    week = day * 7
    year = day * 365.2425
    # numerically encode date-time labels
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
    return feature_df
    
def fill_missing(values):
    # fill missing values with a value at the same time one day ago
	one_day = 60 * 24
	for row in range(values.shape[0]):
		for col in range(values.shape[1]):
			if isnan(values[row, col]):
				values[row, col] = values[row - one_day, col]
                
def generate_training_data(ref_date, data_length=30, sample_rate=1,
                           path='cleaned/household_power_consumption.csv'):
    if not os.path.isfile(path):
        clean_raw_data()
    # load cleaned dataset
    dataset = read_csv("{0}".format(path), header=0, infer_datetime_format=True, 
                       parse_dates=['datetime'], index_col='datetime')
    dataset.index = pd.to_datetime(dataset.index)
    dataset.rename(columns={'Global_active_power':'Power'}, inplace=True)
    # determine number of training samples for given training duration
    time_step = dataset.index[1]-dataset.index[0].seconds
    training_data_samples = int((86400 * data_length)/pd.Timedelta(time_step))
    # TODO add logic to check if returned dataframe is empty
    dataset=dataset[dataset.index <= datetime.strptime(ref_date, '%d/%m/%y %H:%M:%S')]
    # TODO refactor to 'try-except' logic
    if len(dataset) > training_data_samples:
        dataset = dataset[-training_data_samples:]
    else:
        dataset = dataset[:]
    if sample_rate != 1:
        dataset.resample("{0}S".format(int(sample_rate*60))).mean()
    dataset.to_csv("training_data/{0}_day_dataset.csv".format(data_length))
    return dataset

def import_dataset(file_path):
    dataset_df = pd.read_csv("{0}".format(file_path), 
                             header=0, infer_datetime_format=True, 
                             parse_dates=['datetime'], index_col='datetime')
    dataset_df.index = pd.to_datetime(dataset_df.index)
    return dataset_df      

def partition_dataset(feature_dataset, test_portion=0.2, train_portion=0.6, 
                      validation_portion=0.2):
    test_len =  int(len(feature_dataset) * test_portion)
    test_data = feature_dataset[-test_len:]
    train_len = int(len(feature_dataset) * train_portion)
    training_data = feature_dataset[:train_len]
    validation_data = feature_dataset[train_len:-test_len]
    return (training_data, validation_data, test_data)